"""Manga Translator Plugin for GIMP.

Automated manga/comic translation: bubble detection, OCR, translation,
inpainting, and typesetting.

Quick start::

    from manga_translator import translate_file
    result = translate_file("page.png")

    # Or use the pipeline directly:
    from manga_translator import MangaTranslationPipeline
    pipeline = MangaTranslationPipeline()
    result = pipeline.translate_page(image_array)
"""

__version__ = "0.5.0"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API — lazy imports to avoid loading OpenCV at package import time
def __getattr__(name):
    if name == "MangaTranslationPipeline":
        from manga_translator.manga_translator import MangaTranslationPipeline
        return MangaTranslationPipeline
    if name == "translate_file":
        from manga_translator.manga_translator import translate_file
        return translate_file
    if name == "SettingsManager":
        from manga_translator.config.settings import SettingsManager
        return SettingsManager
    if name == "UnifiedConfig":
        from manga_translator.config.unified_config import UnifiedConfig
        return UnifiedConfig
    raise AttributeError(f"module 'manga_translator' has no attribute {name!r}")

__all__ = [
    "MangaTranslationPipeline",
    "translate_file",
    "SettingsManager",
    "UnifiedConfig",
    "__version__",
]

# Auto-register with GIMP if running inside it
try:
    from gimpfu import register  # noqa: F401
    from manga_translator.core.plugin_manager import register_plugin
    register_plugin()
except ImportError:
    pass  # Running standalone, not inside GIMP
