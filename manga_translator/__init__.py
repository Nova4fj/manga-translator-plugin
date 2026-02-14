"""Manga Translator Plugin for GIMP.

Automated manga/comic translation: bubble detection, OCR, translation,
inpainting, and typesetting.
"""

__version__ = "0.1.0"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Auto-register with GIMP if running inside it
try:
    from gimpfu import register  # noqa: F401
    from manga_translator.core.plugin_manager import register_plugin
    register_plugin()
except ImportError:
    pass  # Running standalone, not inside GIMP
