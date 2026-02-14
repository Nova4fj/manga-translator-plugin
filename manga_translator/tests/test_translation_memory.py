"""Tests for translation memory system."""

import json
import os
import pytest

from manga_translator.translation_memory import (
    TranslationMemory,
    TMEntry,
    TMMatch,
    TermEntry,
)


@pytest.fixture
def tm(tmp_path):
    """Create a fresh translation memory for testing."""
    db_path = str(tmp_path / "test_tm.db")
    return TranslationMemory(db_path=db_path)


class TestTranslationMemoryInit:
    def test_creates_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        tm = TranslationMemory(db_path=db_path)
        assert os.path.exists(db_path)

    def test_default_threshold(self, tm):
        assert tm.fuzzy_threshold == 0.85

    def test_custom_threshold(self, tmp_path):
        tm = TranslationMemory(
            db_path=str(tmp_path / "t.db"),
            fuzzy_threshold=0.7,
        )
        assert tm.fuzzy_threshold == 0.7


class TestAddAndLookup:
    def test_add_entry(self, tm):
        entry = tm.add_entry("こんにちは", "Hello")
        assert entry.id is not None
        assert entry.source_text == "こんにちは"
        assert entry.target_text == "Hello"

    def test_exact_lookup(self, tm):
        tm.add_entry("テスト", "Test")
        result = tm.lookup_exact("テスト")
        assert result is not None
        assert result.target_text == "Test"

    def test_exact_lookup_miss(self, tm):
        result = tm.lookup_exact("nonexistent")
        assert result is None

    def test_update_existing(self, tm):
        tm.add_entry("テスト", "Test v1")
        tm.add_entry("テスト", "Test v2")
        result = tm.lookup_exact("テスト")
        assert result.target_text == "Test v2"
        assert tm.entry_count() == 1

    def test_different_languages(self, tm):
        tm.add_entry("Hello", "Bonjour", "en", "fr")
        tm.add_entry("Hello", "Hallo", "en", "de")
        fr = tm.lookup_exact("Hello", "en", "fr")
        de = tm.lookup_exact("Hello", "en", "de")
        assert fr.target_text == "Bonjour"
        assert de.target_text == "Hallo"

    def test_entry_count(self, tm):
        assert tm.entry_count() == 0
        tm.add_entry("a", "A")
        tm.add_entry("b", "B")
        assert tm.entry_count() == 2


class TestFuzzyMatching:
    def test_fuzzy_match(self, tm):
        tm.add_entry("お腹がすいた", "I'm hungry")
        matches = tm.lookup_fuzzy("お腹がすいたよ")
        assert len(matches) >= 1
        assert matches[0].similarity > 0.8

    def test_fuzzy_below_threshold(self, tm):
        tm.add_entry("完全に違う文", "Completely different")
        matches = tm.lookup_fuzzy("こんにちは")
        assert len(matches) == 0

    def test_fuzzy_threshold(self, tmp_path):
        tm = TranslationMemory(
            db_path=str(tmp_path / "t.db"),
            fuzzy_threshold=0.5,
        )
        tm.add_entry("Hello World", "こんにちは世界", source_lang="en", target_lang="ja")
        matches = tm.lookup_fuzzy("Hello World!", "en", "ja")
        assert len(matches) >= 1


class TestContextLookup:
    def test_combined_lookup(self, tm):
        tm.add_entry("行くぞ", "Let's go!", context="Naruto")
        matches = tm.lookup("行くぞ", context="Naruto")
        assert len(matches) >= 1
        assert matches[0].match_type == "exact"

    def test_context_boost(self, tm):
        tm.add_entry("行くぞ", "Let's go!", context="Naruto")
        tm.add_entry("行くぞう", "Going!")
        matches = tm.lookup("行くぞ", context="Naruto")
        # Exact match with context should be ranked first
        assert matches[0].entry.target_text == "Let's go!"

    def test_usage_tracking(self, tm):
        tm.add_entry("テスト", "Test")
        tm.lookup("テスト")  # triggers usage increment
        entry = tm.lookup_exact("テスト")
        assert entry.usage_count >= 1


class TestTerminology:
    def test_add_term(self, tm):
        term = tm.add_term("ナルト", "Naruto", category="character_name")
        assert term.id is not None
        assert term.term == "ナルト"

    def test_lookup_term(self, tm):
        tm.add_term("ナルト", "Naruto", category="character_name")
        result = tm.lookup_term("ナルト")
        assert result is not None
        assert result.translation == "Naruto"

    def test_lookup_term_miss(self, tm):
        result = tm.lookup_term("nonexistent")
        assert result is None

    def test_lookup_term_by_series(self, tm):
        tm.add_term("先生", "Sensei", series="Naruto")
        tm.add_term("先生", "Teacher", series="Other")
        result = tm.lookup_term("先生", series="Naruto")
        assert result.translation == "Sensei"

    def test_list_terms(self, tm):
        tm.add_term("ナルト", "Naruto")
        tm.add_term("サスケ", "Sasuke")
        terms = tm.list_terms()
        assert len(terms) == 2

    def test_list_terms_by_series(self, tm):
        tm.add_term("ナルト", "Naruto", series="Naruto")
        tm.add_term("ルフィ", "Luffy", series="One Piece")
        naruto_terms = tm.list_terms(series="Naruto")
        assert len(naruto_terms) == 1
        assert naruto_terms[0].term == "ナルト"

    def test_term_count(self, tm):
        assert tm.term_count() == 0
        tm.add_term("a", "A")
        assert tm.term_count() == 1


class TestImportExport:
    def test_export_json(self, tm, tmp_path):
        tm.add_entry("テスト", "Test")
        tm.add_term("ナルト", "Naruto")
        path = str(tmp_path / "export.json")
        count = tm.export_json(path)
        assert count == 2
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data["entries"]) == 1
        assert len(data["terms"]) == 1

    def test_import_json(self, tmp_path):
        data = {
            "entries": [
                {"source_text": "テスト", "target_text": "Test"},
                {"source_text": "漫画", "target_text": "Manga"},
            ],
            "terms": [
                {"term": "ナルト", "translation": "Naruto", "category": "character_name"},
            ],
        }
        path = str(tmp_path / "import.json")
        with open(path, "w") as f:
            json.dump(data, f)

        tm = TranslationMemory(db_path=str(tmp_path / "import.db"))
        count = tm.import_json(path)
        assert count == 3
        assert tm.entry_count() == 2
        assert tm.term_count() == 1

    def test_roundtrip(self, tm, tmp_path):
        tm.add_entry("こんにちは", "Hello", context="greeting")
        tm.add_entry("さようなら", "Goodbye")
        tm.add_term("先生", "Sensei", category="honorific")

        export_path = str(tmp_path / "roundtrip.json")
        tm.export_json(export_path)

        tm2 = TranslationMemory(db_path=str(tmp_path / "roundtrip.db"))
        tm2.import_json(export_path)

        assert tm2.entry_count() == 2
        assert tm2.term_count() == 1
        result = tm2.lookup_exact("こんにちは")
        assert result.target_text == "Hello"
