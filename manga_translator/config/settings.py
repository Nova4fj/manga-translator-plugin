"""Configuration and settings management for Manga Translator Plugin."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


@dataclass
class TranslationSettings:
    """Translation engine configuration."""
    primary_engine: str = "deepl"  # deepl, openai, argos
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    deepl_api_key: str = ""
    argos_installed: bool = False
    source_language: str = "ja"  # auto, ja, zh, ko
    target_language: str = "en"
    context_prompt: str = "Translate this manga dialogue naturally into English."


@dataclass
class OCRSettings:
    """OCR engine configuration."""
    primary_engine: str = "manga-ocr"  # manga-ocr, paddleocr, tesseract
    confidence_threshold: float = 0.7
    tesseract_path: str = ""
    language_hint: str = "ja"


@dataclass
class DetectionSettings:
    """Bubble detection configuration."""
    min_bubble_area: int = 1000  # pixels
    max_bubble_area: int = 500000
    edge_sensitivity: int = 100  # Canny threshold
    contour_approx_epsilon: float = 0.02
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0


@dataclass
class InpaintingSettings:
    """Inpainting configuration."""
    method: str = "opencv_telea"  # opencv_telea, opencv_ns, blur
    inpaint_radius: int = 5
    blur_kernel_size: int = 15
    mask_dilation: int = 5


@dataclass
class TypesettingSettings:
    """Typesetting configuration."""
    default_font: str = "Comic Sans MS"
    font_size_ratio: float = 0.7  # relative to bubble height
    min_font_size: int = 10
    max_font_size: int = 72
    text_color: tuple = (0, 0, 0)
    outline_color: tuple = (255, 255, 255)
    outline_width: int = 0
    line_spacing: float = 1.2
    padding_ratio: float = 0.1  # padding inside bubble


@dataclass
class PluginSettings:
    """Top-level plugin settings container."""
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    inpainting: InpaintingSettings = field(default_factory=InpaintingSettings)
    typesetting: TypesettingSettings = field(default_factory=TypesettingSettings)
    workflow_mode: str = "auto"  # auto, semi_auto, manual
    gpu_enabled: bool = True
    log_level: str = "INFO"
    model_cache_dir: str = ""


# Maps environment variable names to (section, key) paths in settings.
_ENV_VAR_MAP: Dict[str, Tuple[str, str]] = {
    "DEEPL_API_KEY": ("translation", "deepl_api_key"),
    "DEEPL_AUTH_KEY": ("translation", "deepl_api_key"),
    "OPENAI_API_KEY": ("translation", "openai_api_key"),
}

# Maps service name shortcuts used with get_api_key() to env var / settings paths.
_SERVICE_KEY_MAP: Dict[str, Tuple[str, str, List[str]]] = {
    # (section, settings_field, [env_var_names_to_check])
    "deepl": ("translation", "deepl_api_key", ["DEEPL_API_KEY", "DEEPL_AUTH_KEY"]),
    "openai": ("translation", "openai_api_key", ["OPENAI_API_KEY"]),
}

# Dataclass type registry for nested reconstruction from dicts.
_SECTION_CLASSES: Dict[str, type] = {
    "translation": TranslationSettings,
    "ocr": OCRSettings,
    "detection": DetectionSettings,
    "inpainting": InpaintingSettings,
    "typesetting": TypesettingSettings,
}

# Fields in TypesettingSettings that must be stored/restored as tuples.
_TUPLE_FIELDS = {"text_color", "outline_color"}


def _serialize_settings(settings: PluginSettings) -> Dict[str, Any]:
    """Convert a PluginSettings instance to a JSON-safe dictionary.

    Tuples are serialized as ``{"__tuple__": true, "items": [...]}}`` so they
    survive the round-trip through JSON (which only has arrays).
    """

    def _convert(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return {"__tuple__": True, "items": list(obj)}
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    raw = asdict(settings)
    return _convert(raw)


def _deserialize_value(value: Any) -> Any:
    """Recursively restore tagged tuples from a deserialized JSON structure."""
    if isinstance(value, dict):
        if value.get("__tuple__") is True:
            return tuple(_deserialize_value(v) for v in value["items"])
        return {k: _deserialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    return value


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Instantiate a dataclass *cls* from *data*, ignoring unknown keys and
    restoring tagged tuples."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in data.items():
        if k in field_names:
            filtered[k] = _deserialize_value(v)
    return cls(**filtered)


class SettingsManager:
    """Manages loading, saving, and accessing plugin settings.

    Settings are persisted as JSON at ``~/.config/manga-translator/settings.json``.
    API keys can also be supplied through environment variables which always
    take precedence over the on-disk values.
    """

    CONFIG_DIR: Path = Path.home() / ".config" / "manga-translator"
    CONFIG_FILE_NAME: str = "settings.json"

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir is not None:
            self.CONFIG_DIR = Path(config_dir)
        self._config_path: Path = self.CONFIG_DIR / self.CONFIG_FILE_NAME
        self._settings: PluginSettings = PluginSettings()
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> PluginSettings:
        """Load settings from the JSON file on disk.

        If the file does not exist or is corrupt the current (default) settings
        are kept and a warning is printed to stderr.
        """
        if not self._config_path.is_file():
            return self._settings

        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            import sys
            print(
                f"[manga-translator] Warning: could not load settings from "
                f"{self._config_path}: {exc}",
                file=sys.stderr,
            )
            return self._settings

        self._apply_raw_dict(raw)
        return self._settings

    def save(self) -> Path:
        """Persist current settings to the JSON file.

        Creates the config directory if it does not exist.  Returns the path
        to the written file.
        """
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = _serialize_settings(self._settings)
        with open(self._config_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return self._config_path

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_settings(self) -> PluginSettings:
        """Return the current ``PluginSettings`` instance."""
        return self._settings

    def update_settings(self, partial: Dict[str, Any]) -> PluginSettings:
        """Merge a (possibly partial) dictionary into the current settings.

        *partial* may contain top-level scalar keys (e.g. ``workflow_mode``)
        and/or nested section dicts (e.g. ``{"translation": {"source_language": "zh"}}``).

        Unknown keys are silently ignored.  Returns the updated settings.
        """
        self._apply_raw_dict(partial)
        return self._settings

    def reset_to_defaults(self) -> PluginSettings:
        """Reset every setting to its default value (does **not** delete the
        file on disk -- call :meth:`save` afterwards if desired)."""
        self._settings = PluginSettings()
        return self._settings

    # ------------------------------------------------------------------
    # API key helpers
    # ------------------------------------------------------------------

    def get_api_key(self, service: str) -> str:
        """Return the API key for *service* (``"deepl"`` or ``"openai"``).

        Environment variables are checked **first** and take precedence over
        the value stored in the settings file.  Returns an empty string if no
        key is found anywhere.
        """
        service_lower = service.lower()
        info = _SERVICE_KEY_MAP.get(service_lower)
        if info is None:
            return ""

        _section, _field, env_vars = info

        # 1. Environment variables have priority.
        for env_name in env_vars:
            value = os.environ.get(env_name, "").strip()
            if value:
                return value

        # 2. Fall back to saved settings.
        section_obj = getattr(self._settings, _section, None)
        if section_obj is not None:
            return str(getattr(section_obj, _field, ""))

        return ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the current settings and return a list of human-readable
        error strings.  An empty list means everything looks good."""
        errors: List[str] = []
        s = self._settings

        # -- Translation engine requires an API key (except argos) ----------
        engine = s.translation.primary_engine
        if engine == "deepl" and not self.get_api_key("deepl"):
            errors.append(
                "DeepL is selected as the translation engine but no API key is "
                "configured. Set DEEPL_API_KEY or provide it in settings."
            )
        if engine == "openai" and not self.get_api_key("openai"):
            errors.append(
                "OpenAI is selected as the translation engine but no API key is "
                "configured. Set OPENAI_API_KEY or provide it in settings."
            )

        # -- OCR: tesseract needs a path if selected ------------------------
        if s.ocr.primary_engine == "tesseract" and not s.ocr.tesseract_path:
            errors.append(
                "Tesseract is selected as the OCR engine but tesseract_path is "
                "not set."
            )

        # -- Detection sanity checks ----------------------------------------
        det = s.detection
        if det.min_bubble_area >= det.max_bubble_area:
            errors.append(
                "detection.min_bubble_area must be less than max_bubble_area."
            )
        if not (0.0 < det.contour_approx_epsilon < 1.0):
            errors.append(
                "detection.contour_approx_epsilon must be between 0 and 1."
            )
        if det.min_aspect_ratio >= det.max_aspect_ratio:
            errors.append(
                "detection.min_aspect_ratio must be less than max_aspect_ratio."
            )

        # -- Inpainting -----------------------------------------------------
        if s.inpainting.method not in ("opencv_telea", "opencv_ns", "blur"):
            errors.append(
                f"Unknown inpainting method: {s.inpainting.method!r}. "
                "Allowed values: opencv_telea, opencv_ns, blur."
            )

        # -- Typesetting ----------------------------------------------------
        ts = s.typesetting
        if ts.min_font_size > ts.max_font_size:
            errors.append(
                "typesetting.min_font_size must not exceed max_font_size."
            )
        if not (0.0 < ts.font_size_ratio <= 2.0):
            errors.append(
                "typesetting.font_size_ratio should be between 0 and 2."
            )
        if not (0.0 <= ts.padding_ratio < 0.5):
            errors.append(
                "typesetting.padding_ratio should be between 0 and 0.5."
            )
        for color_name in ("text_color", "outline_color"):
            color = getattr(ts, color_name)
            if (
                not isinstance(color, (tuple, list))
                or len(color) != 3
                or not all(isinstance(c, int) and 0 <= c <= 255 for c in color)
            ):
                errors.append(
                    f"typesetting.{color_name} must be an (R, G, B) tuple with "
                    "values in 0..255."
                )

        # -- Workflow mode --------------------------------------------------
        if s.workflow_mode not in ("auto", "semi_auto", "manual"):
            errors.append(
                f"Unknown workflow_mode: {s.workflow_mode!r}. "
                "Allowed values: auto, semi_auto, manual."
            )

        # -- Log level ------------------------------------------------------
        if s.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(
                f"Unknown log_level: {s.log_level!r}."
            )

        # -- model_cache_dir (if set, must be a writable directory) ---------
        if s.model_cache_dir:
            cache = Path(s.model_cache_dir)
            if cache.exists() and not cache.is_dir():
                errors.append(
                    f"model_cache_dir '{s.model_cache_dir}' exists but is not "
                    "a directory."
                )

        return errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_raw_dict(self, raw: Dict[str, Any]) -> None:
        """Merge *raw* (a plain dict, e.g. from JSON) into ``self._settings``.

        Nested section dicts are merged field-by-field rather than replacing
        the whole section, so callers can provide partial updates.
        """
        import dataclasses

        for key, value in raw.items():
            # Nested section?
            if key in _SECTION_CLASSES and isinstance(value, dict):
                cls = _SECTION_CLASSES[key]
                current_section = getattr(self._settings, key)
                # Merge each provided field into the existing section instance.
                for fld in dataclasses.fields(cls):
                    if fld.name in value:
                        setattr(
                            current_section,
                            fld.name,
                            _deserialize_value(value[fld.name]),
                        )
            elif hasattr(self._settings, key):
                setattr(self._settings, key, _deserialize_value(value))
