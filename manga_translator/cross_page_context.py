"""Cross-page context accumulator for multi-page translation consistency.

Carries dialogue history, character names, and glossary terms across
pages so that batch translations maintain coherent terminology and tone.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Pattern for detecting character names: Title Case sequences of 2-3 words
# Excludes common English words that are often title-cased at sentence start
_TITLE_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b")
_COMMON_WORDS = frozenset({
    "The", "This", "That", "These", "Those", "Here", "There",
    "What", "When", "Where", "Which", "Who", "How", "Why",
    "Yes", "No", "Not", "But", "And", "So", "If", "Or",
    "Just", "Well", "Now", "Then", "Also", "Still", "Even",
    "Please", "Thank", "Thanks", "Sorry", "Hello", "Hey",
    "Right", "Left", "Good", "Great", "Fine", "Okay",
    "Let", "Come", "Look", "Wait", "Stop", "Run", "Go",
    "All", "Our", "Your", "His", "Her", "Its", "My",
})


@dataclass
class PageDialogue:
    """Dialogue extracted from a single page."""
    page_num: int
    source_texts: List[str] = field(default_factory=list)
    translated_texts: List[str] = field(default_factory=list)


@dataclass
class CrossPageContext:
    """Accumulates translation context across multiple manga pages.

    Tracks recent dialogue, character names, and glossary terms so that
    subsequent pages can receive richer translation prompts.

    Args:
        page_window: Number of previous pages to keep in dialogue history.
        max_dialogue_lines: Maximum dialogue lines returned by get_dialogue_summary().
    """

    dialogue_history: List[PageDialogue] = field(default_factory=list)
    character_names: Dict[str, str] = field(default_factory=dict)
    glossary: Dict[str, str] = field(default_factory=dict)
    pages_processed: int = 0
    page_window: int = 2
    max_dialogue_lines: int = 10

    # --- Update methods ---

    def update_from_page(
        self,
        page_num: int,
        source_texts: List[str],
        translated_texts: List[str],
    ) -> None:
        """Record dialogue from a completed page.

        Appends the dialogue and trims history to ``page_window`` pages.
        Also runs heuristic name detection on the translations.

        Args:
            page_num: Page number (0-based or 1-based, caller decides).
            source_texts: OCR source texts from the page.
            translated_texts: Translated texts (same order as source_texts).
        """
        page = PageDialogue(
            page_num=page_num,
            source_texts=list(source_texts),
            translated_texts=list(translated_texts),
        )
        self.dialogue_history.append(page)

        # Trim to page_window
        if len(self.dialogue_history) > self.page_window:
            self.dialogue_history = self.dialogue_history[-self.page_window:]

        self.pages_processed += 1

        # Auto-detect character names from translations
        detected = self.detect_names_from_translations(translated_texts)
        for name in detected:
            if name not in self.character_names.values():
                self.character_names[name] = name
                logger.debug("Auto-detected character name: %s", name)

    # --- Query methods ---

    def get_dialogue_summary(self, max_lines: Optional[int] = None) -> str:
        """Format recent dialogue as a prompt-friendly summary.

        Args:
            max_lines: Override for maximum lines. Uses self.max_dialogue_lines if None.

        Returns:
            Formatted string with recent dialogue, or empty string if no history.
        """
        limit = max_lines if max_lines is not None else self.max_dialogue_lines
        if not self.dialogue_history:
            return ""

        lines: List[str] = []
        for page in self.dialogue_history:
            for src, tgt in zip(page.source_texts, page.translated_texts):
                if src.strip() and tgt.strip():
                    lines.append(f"  {src} -> {tgt}")

        if not lines:
            return ""

        # Take last N lines
        lines = lines[-limit:]
        return "Previous dialogue:\n" + "\n".join(lines)

    def get_character_map(self) -> Dict[str, str]:
        """Return accumulated character name mappings."""
        return dict(self.character_names)

    def get_glossary(self) -> Dict[str, str]:
        """Return accumulated glossary terms."""
        return dict(self.glossary)

    # --- Mutators ---

    def add_character_name(self, source: str, translated: str) -> None:
        """Register a character name mapping.

        Args:
            source: Name in source language.
            translated: Name in target language.
        """
        self.character_names[source] = translated

    def add_glossary_term(self, source: str, translated: str) -> None:
        """Register a glossary term.

        Args:
            source: Term in source language.
            translated: Term in target language.
        """
        self.glossary[source] = translated

    # --- Name detection ---

    def detect_names_from_translations(self, translations: List[str]) -> List[str]:
        """Heuristic: extract likely character names from translated text.

        Looks for Title Case sequences (2-3 words) that don't match common
        English words. Returns deduplicated list of detected names.

        Args:
            translations: List of translated texts to scan.

        Returns:
            List of likely character names found.
        """
        names: List[str] = []
        seen = set()

        for text in translations:
            matches = _TITLE_CASE_RE.findall(text)
            for match in matches:
                words = match.split()
                # Strip leading common words
                while words and words[0] in _COMMON_WORDS:
                    words = words[1:]
                if not words:
                    continue
                cleaned = " ".join(words)
                # Require at least 2 words for higher confidence
                if len(words) >= 2 and cleaned not in seen:
                    seen.add(cleaned)
                    names.append(cleaned)

        return names

    def check_name_consistency(self, translations: List[str]) -> List[str]:
        """Check new translations for character name inconsistencies.

        Compares names in the new translations against the accumulated
        character_names map and flags any mismatches.

        Args:
            translations: Newly translated texts to check.

        Returns:
            List of warning messages (empty if consistent).
        """
        warnings: List[str] = []
        known_names = set(self.character_names.values())

        if not known_names:
            return warnings

        new_names = self.detect_names_from_translations(translations)
        for name in new_names:
            # Check for near-duplicates (same first word, different full form)
            first_word = name.split()[0]
            for known in known_names:
                known_first = known.split()[0]
                if first_word == known_first and name != known:
                    warnings.append(
                        f"Possible name inconsistency: '{name}' vs previously seen '{known}'"
                    )

        return warnings
