"""Intelligent font selection based on bubble type and text characteristics.

Selects appropriate fonts for translated text based on bubble classification
(speech, thought, shout, narration, caption, SFX).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from manga_translator.components.bubble_classifier import BubbleType

logger = logging.getLogger(__name__)


@dataclass
class FontProfile:
    """Metadata for a font."""
    name: str
    path: Optional[str] = None
    style: str = "sans"  # serif, sans, comic, gothic, display
    weight: str = "regular"  # light, regular, bold, heavy
    suitable_for: List[str] = field(default_factory=list)  # BubbleType values


# Default font profiles for each bubble type
_DEFAULT_PROFILES: Dict[BubbleType, FontProfile] = {
    BubbleType.SPEECH: FontProfile(
        name="default-speech", style="comic", weight="regular",
        suitable_for=["speech"],
    ),
    BubbleType.THOUGHT: FontProfile(
        name="default-thought", style="serif", weight="light",
        suitable_for=["thought"],
    ),
    BubbleType.SHOUT: FontProfile(
        name="default-shout", style="gothic", weight="heavy",
        suitable_for=["shout"],
    ),
    BubbleType.NARRATION: FontProfile(
        name="default-narration", style="serif", weight="regular",
        suitable_for=["narration"],
    ),
    BubbleType.CAPTION: FontProfile(
        name="default-caption", style="sans", weight="regular",
        suitable_for=["caption"],
    ),
    BubbleType.UNKNOWN: FontProfile(
        name="default-fallback", style="sans", weight="regular",
        suitable_for=["unknown"],
    ),
}


class FontMatcher:
    """Select appropriate fonts based on bubble type and text characteristics."""

    def __init__(
        self,
        fonts_dir: Optional[str] = None,
        fallback_font: Optional[str] = None,
    ):
        """
        Args:
            fonts_dir: Directory to scan for custom fonts.
            fallback_font: Path to fallback font file.
        """
        self._custom_fonts: Dict[str, FontProfile] = {}
        self._type_overrides: Dict[BubbleType, FontProfile] = {}
        self.fallback_font = fallback_font

        if fonts_dir:
            self._scan_fonts_dir(fonts_dir)

    def register_font(self, profile: FontProfile) -> None:
        """Register a custom font profile.

        If the profile has suitable_for entries, it overrides the default
        for those bubble types.
        """
        self._custom_fonts[profile.name] = profile
        for type_name in profile.suitable_for:
            try:
                bt = BubbleType(type_name)
                self._type_overrides[bt] = profile
            except ValueError:
                logger.warning("Unknown bubble type in suitable_for: %s", type_name)

    def match_font(
        self,
        bubble_type: BubbleType = BubbleType.UNKNOWN,
        text: str = "",
        language: str = "en",
    ) -> FontProfile:
        """Select the best font for a bubble.

        Args:
            bubble_type: Classified bubble type.
            text: The text to render (for future style analysis).
            language: Target language code.

        Returns:
            FontProfile for the best matching font.
        """
        # Check custom overrides first
        if bubble_type in self._type_overrides:
            return self._type_overrides[bubble_type]

        # Check custom fonts that match the type
        for profile in self._custom_fonts.values():
            if bubble_type.value in profile.suitable_for:
                return profile

        # Fall back to defaults
        if bubble_type in _DEFAULT_PROFILES:
            return _DEFAULT_PROFILES[bubble_type]

        return _DEFAULT_PROFILES[BubbleType.UNKNOWN]

    def match_font_batch(
        self,
        bubble_types: List[BubbleType],
        texts: Optional[List[str]] = None,
        language: str = "en",
    ) -> List[FontProfile]:
        """Match fonts for multiple bubbles."""
        texts = texts or [""] * len(bubble_types)
        return [
            self.match_font(bt, t, language)
            for bt, t in zip(bubble_types, texts)
        ]

    def list_fonts(self) -> Dict[str, FontProfile]:
        """Return all registered custom fonts."""
        return dict(self._custom_fonts)

    def _scan_fonts_dir(self, fonts_dir: str) -> None:
        """Scan a directory for font files and register them."""
        path = Path(fonts_dir)
        if not path.is_dir():
            logger.warning("Fonts directory not found: %s", fonts_dir)
            return

        font_extensions = {".ttf", ".otf", ".woff", ".woff2"}
        for font_file in path.iterdir():
            if font_file.suffix.lower() in font_extensions:
                name = font_file.stem
                profile = FontProfile(
                    name=name,
                    path=str(font_file),
                    style=self._guess_style(name),
                    weight=self._guess_weight(name),
                )
                self._custom_fonts[name] = profile
                logger.debug("Registered font: %s from %s", name, font_file)

    @staticmethod
    def _guess_style(name: str) -> str:
        """Guess font style from filename."""
        lower = name.lower()
        if any(w in lower for w in ("comic", "manga", "bubble")):
            return "comic"
        if any(w in lower for w in ("gothic", "black")):
            return "gothic"
        if any(w in lower for w in ("serif", "times", "garamond")):
            return "serif"
        if any(w in lower for w in ("display", "impact", "poster")):
            return "display"
        return "sans"

    @staticmethod
    def _guess_weight(name: str) -> str:
        """Guess font weight from filename."""
        lower = name.lower()
        if any(w in lower for w in ("heavy", "black", "extra-bold", "extrabold")):
            return "heavy"
        if any(w in lower for w in ("bold", "strong")):
            return "bold"
        if any(w in lower for w in ("light", "thin")):
            return "light"
        return "regular"
