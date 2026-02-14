"""Context-aware translation enhancement.

Enriches translation requests with surrounding context, terminology,
and page-level information to improve translation consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class TranslationContext:
    """Context information for a single translation request."""
    text: str
    index: int  # Position on the page (0-based)
    prev_texts: List[str] = field(default_factory=list)  # Previous bubbles
    next_texts: List[str] = field(default_factory=list)  # Following bubbles
    character_name: str = ""  # Speaker name if known
    glossary: Dict[str, str] = field(default_factory=dict)  # term -> translation


@dataclass
class PageContext:
    """Context for an entire page of translations."""
    entries: List[TranslationContext] = field(default_factory=list)
    series_name: str = ""
    chapter: str = ""
    page_num: int = 0


class ContextBuilder:
    """Builds translation context from page-level information.

    Provides surrounding bubble text as context to help translators
    maintain consistency in tone, character voice, and terminology.

    Args:
        context_window: Number of surrounding bubbles to include as context.
        glossary: Dict mapping source terms to expected translations.
    """

    def __init__(
        self,
        context_window: int = 2,
        glossary: Optional[Dict[str, str]] = None,
    ):
        self.context_window = context_window
        self.glossary = glossary or {}

    def build_page_context(
        self,
        texts: List[str],
        series_name: str = "",
        chapter: str = "",
        page_num: int = 0,
    ) -> PageContext:
        """Build context for all texts on a page.

        Args:
            texts: List of source texts (in reading order).
            series_name: Series name for glossary filtering.
            chapter: Chapter identifier.
            page_num: Page number.

        Returns:
            PageContext with surrounding text context for each entry.
        """
        entries = []
        for i, text in enumerate(texts):
            prev_start = max(0, i - self.context_window)
            next_end = min(len(texts), i + self.context_window + 1)

            entry = TranslationContext(
                text=text,
                index=i,
                prev_texts=texts[prev_start:i],
                next_texts=texts[i + 1:next_end],
                glossary=self._filter_glossary(text),
            )
            entries.append(entry)

        return PageContext(
            entries=entries,
            series_name=series_name,
            chapter=chapter,
            page_num=page_num,
        )

    def format_prompt_context(self, ctx: TranslationContext) -> str:
        """Format a TranslationContext into a prompt supplement.

        Returns a string that can be prepended to the translation prompt
        to provide context information.
        """
        parts = []

        if ctx.prev_texts:
            prev = " | ".join(ctx.prev_texts[-2:])  # Last 2
            parts.append(f"[Previous dialogue: {prev}]")

        if ctx.next_texts:
            nxt = " | ".join(ctx.next_texts[:2])  # Next 2
            parts.append(f"[Following dialogue: {nxt}]")

        if ctx.character_name:
            parts.append(f"[Speaker: {ctx.character_name}]")

        if ctx.glossary:
            terms = ", ".join(f"{k}={v}" for k, v in ctx.glossary.items())
            parts.append(f"[Glossary: {terms}]")

        return "\n".join(parts)

    def format_page_prompt(self, page_ctx: PageContext) -> str:
        """Format entire page context for batch translation.

        Creates a single context block for translating all bubbles together.
        """
        parts = []

        if page_ctx.series_name:
            parts.append(f"Series: {page_ctx.series_name}")

        if page_ctx.entries and any(e.glossary for e in page_ctx.entries):
            all_terms = {}
            for entry in page_ctx.entries:
                all_terms.update(entry.glossary)
            if all_terms:
                terms = ", ".join(f"{k} = {v}" for k, v in all_terms.items())
                parts.append(f"Terminology: {terms}")

        return "\n".join(parts)

    def _filter_glossary(self, text: str) -> Dict[str, str]:
        """Return glossary entries relevant to the given text."""
        relevant = {}
        text_lower = text.lower()
        for term, translation in self.glossary.items():
            if term.lower() in text_lower:
                relevant[term] = translation
        return relevant

    def add_character_names(
        self,
        page_ctx: PageContext,
        character_map: Dict[int, str],
    ) -> PageContext:
        """Assign character names to specific bubble indices.

        Args:
            page_ctx: The page context to update.
            character_map: Dict mapping bubble index to character name.

        Returns:
            Updated PageContext.
        """
        for entry in page_ctx.entries:
            if entry.index in character_map:
                entry.character_name = character_map[entry.index]
        return page_ctx
