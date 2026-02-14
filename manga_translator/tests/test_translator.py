"""Tests for translation engine."""

import pytest

from manga_translator.components.translator import (
    TranslationManager,
    TranslationResult,
    ArgosEngine,
    OpenAIEngine,
    DeepLEngine,
)


class TestTranslationManager:
    def test_init(self):
        manager = TranslationManager()
        assert manager is not None

    def test_available_engines(self):
        manager = TranslationManager()
        engines = manager.available_engines
        assert isinstance(engines, list)

    def test_translate_empty_text(self):
        manager = TranslationManager()
        result = manager.translate("", "ja", "en")
        assert isinstance(result, TranslationResult)
        assert result.translated_text == ""

    def test_translate_returns_result(self):
        manager = TranslationManager()
        result = manager.translate("テスト", "ja", "en")
        assert isinstance(result, TranslationResult)
        assert result.source_text == "テスト"
        # May fail if no engine available, but should still return a result
        assert result.engine_used is not None

    def test_batch_translate_empty(self):
        manager = TranslationManager()
        results = manager.translate_batch([], "ja", "en")
        assert results == []

    def test_engine_availability_check(self):
        openai = OpenAIEngine(api_key="")
        assert openai.is_available() is False  # No API key

        deepl = DeepLEngine(api_key="")
        assert deepl.is_available() is False  # No API key

    def test_argos_availability(self):
        argos = ArgosEngine()
        # Returns True only if argostranslate is installed
        assert isinstance(argos.is_available(), bool)
