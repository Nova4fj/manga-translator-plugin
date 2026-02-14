"""GIMP plugin registration and lifecycle management."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _gimp_translate_page(image, drawable, mode=0, source_lang="ja", target_lang="en"):
    """GIMP callback for translating a manga page."""
    from gimpfu import pdb, gimp

    mode_names = ["auto", "semi_auto", "manual"]
    mode_str = mode_names[mode] if mode < len(mode_names) else "auto"

    gimp.progress_init("Manga Translator: Starting...")

    try:
        from manga_translator.manga_translator import MangaTranslationPipeline
        from manga_translator.config.settings import SettingsManager
        from manga_translator.core.image_processor import pil_to_numpy, numpy_to_pil
        from manga_translator.core.layer_manager import GimpLayerAdapter

        # Load settings
        settings_mgr = SettingsManager()
        settings = settings_mgr.get_settings()
        settings.workflow_mode = mode_str
        settings.translation.source_language = source_lang
        settings.translation.target_language = target_lang

        # Get image data from GIMP
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

        gimp.progress_update(0.1)

        # Run translation pipeline
        pipeline = MangaTranslationPipeline(settings)

        def progress_callback(step, total, message):
            gimp.progress_update(0.1 + 0.8 * (step / total))
            gimp.progress_init(f"Manga Translator: {message}")

        result = pipeline.translate_page(img_array, progress_callback=progress_callback)

        gimp.progress_update(0.9)

        # Create output layer in GIMP
        layer_adapter = GimpLayerAdapter(image)
        if layer_adapter.is_gimp_available:
            layer_adapter.create_layer("Translated", result.final_image)

        gimp.progress_update(1.0)
        pdb.gimp_displays_flush()
        pdb.gimp_image_clean_all(image)

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
