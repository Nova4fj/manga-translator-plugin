"""Tests for the context-aware translation module."""

import pytest

from manga_translator.translation_context import (
    ContextBuilder,
    PageContext,
    TranslationContext,
)


class TestTranslationContext:
    """Tests for the TranslationContext dataclass."""

    def test_defaults(self):
        """Empty context has empty lists and defaults."""
        ctx = TranslationContext(text="hello", index=0)
        assert ctx.prev_texts == []
        assert ctx.next_texts == []
        assert ctx.character_name == ""
        assert ctx.glossary == {}

    def test_with_prev_next(self):
        """Check prev/next populated correctly."""
        ctx = TranslationContext(
            text="middle",
            index=1,
            prev_texts=["first"],
            next_texts=["third"],
            character_name="Goku",
            glossary={"ki": "energy"},
        )
        assert ctx.prev_texts == ["first"]
        assert ctx.next_texts == ["third"]
        assert ctx.character_name == "Goku"
        assert ctx.glossary == {"ki": "energy"}


class TestContextBuilder:
    """Tests for ContextBuilder.build_page_context."""

    def test_build_page_context_basic(self):
        """3 texts with default window=2, verify context for each."""
        builder = ContextBuilder()
        texts = ["one", "two", "three"]
        page = builder.build_page_context(texts)

        assert len(page.entries) == 3
        # First entry
        assert page.entries[0].prev_texts == []
        assert page.entries[0].next_texts == ["two", "three"]
        # Middle entry
        assert page.entries[1].prev_texts == ["one"]
        assert page.entries[1].next_texts == ["three"]
        # Last entry
        assert page.entries[2].prev_texts == ["one", "two"]
        assert page.entries[2].next_texts == []

    def test_context_window_size(self):
        """Custom window=1, verify only 1 neighbor each side."""
        builder = ContextBuilder(context_window=1)
        texts = ["a", "b", "c", "d", "e"]
        page = builder.build_page_context(texts)

        mid = page.entries[2]  # "c"
        assert mid.prev_texts == ["b"]
        assert mid.next_texts == ["d"]

    def test_first_bubble_no_prev(self):
        """First bubble has empty prev_texts."""
        builder = ContextBuilder()
        texts = ["first", "second", "third"]
        page = builder.build_page_context(texts)

        assert page.entries[0].prev_texts == []
        assert page.entries[0].text == "first"

    def test_last_bubble_no_next(self):
        """Last bubble has empty next_texts."""
        builder = ContextBuilder()
        texts = ["first", "second", "third"]
        page = builder.build_page_context(texts)

        assert page.entries[2].next_texts == []
        assert page.entries[2].text == "third"

    def test_single_bubble(self):
        """Only one text, no prev/next."""
        builder = ContextBuilder()
        texts = ["solo"]
        page = builder.build_page_context(texts)

        assert len(page.entries) == 1
        assert page.entries[0].prev_texts == []
        assert page.entries[0].next_texts == []
        assert page.entries[0].text == "solo"

    def test_glossary_filtering(self):
        """Only matching terms included in entry glossary."""
        builder = ContextBuilder(glossary={
            "nakama": "comrade",
            "sensei": "teacher",
            "baka": "fool",
        })
        texts = ["nakama wa daiji", "sensei ga", "hello world"]
        page = builder.build_page_context(texts)

        assert page.entries[0].glossary == {"nakama": "comrade"}
        assert page.entries[1].glossary == {"sensei": "teacher"}
        assert page.entries[2].glossary == {}

    def test_empty_texts(self):
        """Empty list returns empty PageContext."""
        builder = ContextBuilder()
        page = builder.build_page_context([])

        assert page.entries == []
        assert page.series_name == ""


class TestFormatPromptContext:
    """Tests for ContextBuilder.format_prompt_context."""

    def test_format_with_all_fields(self):
        """All context fields present in output."""
        builder = ContextBuilder()
        ctx = TranslationContext(
            text="main line",
            index=1,
            prev_texts=["prev1", "prev2"],
            next_texts=["next1", "next2"],
            character_name="Naruto",
            glossary={"jutsu": "technique"},
        )
        result = builder.format_prompt_context(ctx)

        assert "[Previous dialogue: prev1 | prev2]" in result
        assert "[Following dialogue: next1 | next2]" in result
        assert "[Speaker: Naruto]" in result
        assert "[Glossary: jutsu=technique]" in result

    def test_format_minimal(self):
        """Only text, no extras produces empty string."""
        builder = ContextBuilder()
        ctx = TranslationContext(text="just text", index=0)
        result = builder.format_prompt_context(ctx)

        assert result == ""

    def test_format_with_glossary(self):
        """Glossary terms appear in output."""
        builder = ContextBuilder()
        ctx = TranslationContext(
            text="test",
            index=0,
            glossary={"ki": "energy", "dojo": "training hall"},
        )
        result = builder.format_prompt_context(ctx)

        assert "[Glossary:" in result
        assert "ki=energy" in result
        assert "dojo=training hall" in result


class TestFormatPagePrompt:
    """Tests for ContextBuilder.format_page_prompt."""

    def test_page_prompt_with_series(self):
        """Series name included in page prompt."""
        builder = ContextBuilder()
        page = PageContext(series_name="One Piece")
        result = builder.format_page_prompt(page)

        assert "Series: One Piece" in result

    def test_page_prompt_with_terms(self):
        """Terminology block included when glossary entries exist."""
        builder = ContextBuilder()
        entries = [
            TranslationContext(text="a", index=0, glossary={"nakama": "comrade"}),
            TranslationContext(text="b", index=1, glossary={"sensei": "teacher"}),
        ]
        page = PageContext(entries=entries)
        result = builder.format_page_prompt(page)

        assert "Terminology:" in result
        assert "nakama = comrade" in result
        assert "sensei = teacher" in result


class TestAddCharacterNames:
    """Tests for ContextBuilder.add_character_names."""

    def test_assign_characters(self):
        """Character names assigned to correct indices."""
        builder = ContextBuilder()
        texts = ["line 1", "line 2", "line 3"]
        page = builder.build_page_context(texts)

        character_map = {0: "Luffy", 2: "Zoro"}
        page = builder.add_character_names(page, character_map)

        assert page.entries[0].character_name == "Luffy"
        assert page.entries[1].character_name == ""
        assert page.entries[2].character_name == "Zoro"
