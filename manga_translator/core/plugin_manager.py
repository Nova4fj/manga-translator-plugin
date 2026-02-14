"""GIMP plugin registration and lifecycle management."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import validation (may not be available in all GIMP environments)
try:
    from manga_translator.input_validator import InputValidator, ValidationError
except ImportError:
    InputValidator = None
    ValidationError = None

# Import error recovery (optional)
try:
    from manga_translator.error_recovery import ErrorRecoveryManager
except ImportError:
    ErrorRecoveryManager = None

# Preset configurations: (name, description-dict-key)
QUALITY_PRESETS = {
    0: {  # Fast
        "name": "Fast",
        "ocr_confidence": 0.5,
        "inpaint_quality": 0.3,
        "max_dimension": 2048,
    },
    1: {  # Balanced
        "name": "Balanced",
        "ocr_confidence": 0.7,
        "inpaint_quality": 0.5,
        "max_dimension": 4096,
    },
    2: {  # Quality
        "name": "Quality",
        "ocr_confidence": 0.8,
        "inpaint_quality": 0.7,
        "max_dimension": 8192,
    },
}

# Add plugin directory to path for imports
PLUGIN_DIR = Path(__file__).parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))


def check_gimp_environment() -> bool:
    """Check if we're running inside GIMP."""
    try:
        from gimpfu import pdb  # noqa: F401
        return True
    except ImportError:
        return False


def get_gimp_version() -> str:
    """Get the GIMP version string."""
    try:
        from gimpfu import pdb
        return pdb.gimp_version()
    except (ImportError, Exception):
        return "standalone"


def register_plugin():
    """Register the manga translator as a GIMP Python-Fu plugin.

    This is called from the package __init__.py when loaded by GIMP.
    """
    try:
        from gimpfu import register, main, PF_OPTION, PF_STRING

        register(
            "python-fu-manga-translate",
            "Manga Translator - Translate manga page",
            "Automatically detect speech bubbles, extract text, translate, and typeset",
            "Manga Translator Contributors",
            "GPL v3",
            "2026",
            "<Image>/Filters/Manga/Translate Page...",
            "RGB*, GRAY*",
            [
                (PF_OPTION, "mode", "Translation mode", 0,
                 ["Auto", "Semi-Auto", "Manual"]),
                (PF_STRING, "source_lang", "Source language (ja/zh/ko/auto)", "ja"),
                (PF_STRING, "target_lang", "Target language", "en"),
                (PF_OPTION, "preset", "Quality preset", 1,
                 ["Fast", "Balanced", "Quality"]),
            ],
            [],
            _gimp_translate_page,
        )

        register(
            "python-fu-manga-translate-settings",
            "Manga Translator - Settings",
            "Configure manga translator plugin settings",
            "Manga Translator Contributors",
            "GPL v3",
            "2026",
            "<Image>/Filters/Manga/Settings...",
            "",
            [],
            [],
            _gimp_open_settings,
        )

        logger.info("Manga Translator plugin registered successfully")
        main()

    except ImportError:
        logger.info("Not running inside GIMP, skipping plugin registration")


def _apply_preset(settings, preset_index):
    """Apply a quality preset to the settings object.

    Args:
        settings: PluginSettings instance to modify in-place.
        preset_index: 0=Fast, 1=Balanced, 2=Quality.
    """
    preset = QUALITY_PRESETS.get(preset_index, QUALITY_PRESETS[1])
    logger.info("Applying '%s' quality preset", preset["name"])

    settings.ocr.confidence_threshold = preset["ocr_confidence"]
    settings.inpainting.quality_threshold = preset["inpaint_quality"]
    # max_dimension is used by the pipeline's resize_for_processing call;
    # store it so the progress callback can reference it if needed.
    settings._preset_max_dimension = preset.get("max_dimension", 4096)


def _gimp_translate_page(image, drawable, mode=0, source_lang="ja",
                         target_lang="en", preset=1):
    """GIMP callback for translating a manga page.

    Args:
        image: GIMP image object.
        drawable: Active drawable.
        mode: Translation mode index (0=Auto, 1=Semi-Auto, 2=Manual).
        source_lang: Source language code.
        target_lang: Target language code.
        preset: Quality preset index (0=Fast, 1=Balanced, 2=Quality).
    """
    from gimpfu import pdb, gimp

    mode_names = ["auto", "semi_auto", "manual"]
    mode_str = mode_names[mode] if mode < len(mode_names) else "auto"
    preset_name = QUALITY_PRESETS.get(preset, QUALITY_PRESETS[1])["name"]

    gimp.progress_init(f"Manga Translator: Initializing ({preset_name} mode)...")

    try:
        from manga_translator.manga_translator import MangaTranslationPipeline
        from manga_translator.config.settings import SettingsManager
        from manga_translator.core.layer_manager import GimpLayerAdapter

        # Load settings and apply preset
        gimp.progress_init("Manga Translator: Loading settings...")
        settings_mgr = SettingsManager()
        settings = settings_mgr.get_settings()
        settings.workflow_mode = mode_str
        settings.translation.source_language = source_lang
        settings.translation.target_language = target_lang
        _apply_preset(settings, preset)

        gimp.progress_update(0.05)

        # Get image data from GIMP
        gimp.progress_init("Manga Translator: Reading image data from GIMP...")
        width = pdb.gimp_image_width(image)
        height = pdb.gimp_image_height(image)

        # Flatten to get pixel data
        flat = pdb.gimp_image_flatten(image)
        pixel_region = flat.get_pixel_rgn(0, 0, width, height, False, False)
        import numpy as np
        pixels = np.frombuffer(pixel_region[:, :], dtype=np.uint8)
        channels = pdb.gimp_drawable_bpp(flat)
        img_array = pixels.reshape((height, width, channels))

        # Convert RGB to BGR for OpenCV
        if channels >= 3:
            img_array = img_array[:, :, :3][:, :, ::-1].copy()

        gimp.progress_update(0.08)

        # Validate the image array
        if InputValidator is not None:
            gimp.progress_init("Manga Translator: Validating image...")
            try:
                InputValidator.validate_image_array(img_array)
            except (ValidationError, Exception) as ve:
                pdb.gimp_message(f"Manga Translator: Image validation failed — {ve}")
                return
        gimp.progress_update(0.1)

        # Run translation pipeline
        gimp.progress_init("Manga Translator: Starting translation pipeline...")
        pipeline = MangaTranslationPipeline(settings)

        step_labels = {
            0: "Detecting speech bubbles...",
            1: "Extracting text (OCR)...",
            2: "Translating text...",
            3: "Removing original text (inpainting)...",
            4: "Typesetting translated text...",
            5: "Assembling final image...",
        }

        def progress_callback(step, total, message):
            label = step_labels.get(step, message)
            progress = 0.1 + 0.8 * (step / max(total, 1))
            gimp.progress_update(progress)
            gimp.progress_init(f"Manga Translator: {label}")

        # Wrap pipeline in error recovery if available
        if ErrorRecoveryManager is not None:
            recovery = ErrorRecoveryManager()
            try:
                result = pipeline.translate_page(
                    img_array, progress_callback=progress_callback,
                )
            except Exception as pipeline_err:
                logger.warning(
                    "Pipeline failed, attempting recovery: %s", pipeline_err,
                )
                recovery.report.add(
                    "pipeline", str(pipeline_err),
                    "Full pipeline error — attempting graceful degradation", False,
                )
                # Re-raise; the outer except will handle user notification
                raise
            finally:
                report_summary = recovery.report.summary()
                if recovery.report.total_recoveries > 0:
                    logger.info("Recovery report:\n%s", report_summary)
        else:
            result = pipeline.translate_page(
                img_array, progress_callback=progress_callback,
            )

        gimp.progress_update(0.9)
        gimp.progress_init("Manga Translator: Creating output layer...")

        # Create output layer in GIMP
        layer_adapter = GimpLayerAdapter(image)
        if layer_adapter.is_gimp_available:
            layer_adapter.create_layer("Translated", result.final_image)

        gimp.progress_update(1.0)
        gimp.progress_init("Manga Translator: Done!")
        pdb.gimp_displays_flush()
        pdb.gimp_image_clean_all(image)

        # Report summary
        n_bubbles = len(result.bubbles)
        success_pct = result.success_rate * 100
        logger.info(
            "GIMP translation complete: %d bubbles, %.0f%% success, preset=%s",
            n_bubbles, success_pct, preset_name,
        )

    except Exception as e:
        logger.error("Translation failed: %s", e, exc_info=True)
        pdb.gimp_message(f"Manga Translator Error: {e}")


def _gimp_open_settings(image, drawable):
    """GIMP callback for opening the settings dialog."""
    try:
        from gimpfu import pdb
        pdb.gimp_message("Settings dialog — use config file at ~/.config/manga-translator/settings.json")
    except Exception as e:
        logger.error("Settings dialog failed: %s", e)
