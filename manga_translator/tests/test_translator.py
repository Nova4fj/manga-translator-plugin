"""Tests for translation engine."""

import pytest
from unittest.mock import patch, MagicMock

from manga_translator.components.translator import (
    TranslationManager,
    TranslationResult,
    ArgosEngine,
    OpenAIEngine,
    DeepLEngine,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def good_func():
            return "ok"
        assert good_func() == "ok"

    def test_retries_on_failure(self):
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("temporary error")
            return "success"

        assert flaky_func() == "success"
        assert call_count[0] == 3

    def test_exhausts_retries(self):
        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def bad_func():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            bad_func()

    def test_rate_limit_delay(self):
        call_count = [0]

        @retry_with_backoff(max_retries=1, base_delay=0.01, max_delay=0.05)
        def rate_limited():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("429 rate limit exceeded")
            return "ok"

        assert rate_limited() == "ok"


class TestTranslationResult:
    def test_basic_fields(self):
        r = TranslationResult(
            source_text="hello",
            translated_text="こんにちは",
            source_language="en",
            target_language="ja",
            engine_used="test",
            confidence=0.9,
        )
        assert r.source_text == "hello"
        assert r.confidence == 0.9
        assert r.error is None


class TestOpenAIEngine:
    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        engine = OpenAIEngine(api_key="")
        assert engine.is_available() is False

    def test_not_available_without_package(self):
        engine = OpenAIEngine(api_key="fake-key")
        # Depends on whether openai is installed
        assert isinstance(engine.is_available(), bool)

    @patch("manga_translator.components.translator.OpenAIEngine._get_client")
    def test_translate_success(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_client.return_value.chat.completions.create.return_value = mock_response

        engine = OpenAIEngine(api_key="fake")
        result = engine.translate("こんにちは", "ja", "en")
        assert result.translated_text == "Hello"
        assert result.confidence == 0.9

    @patch("manga_translator.components.translator.OpenAIEngine._get_client")
    def test_translate_failure(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = RuntimeError("API error")

        engine = OpenAIEngine(api_key="fake")
        result = engine.translate("テスト", "ja", "en")
        assert result.error is not None
        assert result.confidence == 0.0

    @patch("manga_translator.components.translator.OpenAIEngine._get_client")
    def test_batch_translate(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[1] Hello\n[2] World"
        mock_client.return_value.chat.completions.create.return_value = mock_response

        engine = OpenAIEngine(api_key="fake")
        results = engine.translate_batch(["こんにちは", "世界"], "ja", "en")
        assert len(results) == 2
        assert results[0].translated_text == "Hello"
        assert results[1].translated_text == "World"

    @patch("manga_translator.components.translator.OpenAIEngine._get_client")
    def test_batch_single_item(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_client.return_value.chat.completions.create.return_value = mock_response

        engine = OpenAIEngine(api_key="fake")
        results = engine.translate_batch(["こんにちは"], "ja", "en")
        assert len(results) == 1

    @patch("manga_translator.components.translator.OpenAIEngine._get_client")
    def test_batch_translate_failure_fallback(self, mock_client):
        # Batch fails, falls back to individual
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("batch failed")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "translated"
            return mock_resp

        mock_client.return_value.chat.completions.create.side_effect = side_effect
        engine = OpenAIEngine(api_key="fake")
        results = engine.translate_batch(["text1", "text2"], "ja", "en")
        assert len(results) == 2


class TestDeepLEngine:
    def test_not_available_without_key(self):
        engine = DeepLEngine(api_key="")
        assert engine.is_available() is False

    def test_lang_map(self):
        engine = DeepLEngine()
        assert engine.LANG_MAP["ja"] == "JA"
        assert engine.LANG_MAP["en"] == "EN-US"

    def test_translate_when_unavailable(self):
        engine = DeepLEngine(api_key="")
        result = engine.translate("test", "ja", "en")
        # Should fail since no API key
        assert isinstance(result, TranslationResult)

    @patch("manga_translator.components.translator.DeepLEngine._get_translator")
    @patch("manga_translator.components.translator.DeepLEngine.is_available", return_value=True)
    def test_translate_success(self, mock_avail, mock_translator):
        mock_result = MagicMock()
        mock_result.text = "Hello"
        mock_translator.return_value.translate_text.return_value = mock_result

        engine = DeepLEngine(api_key="fake")
        result = engine.translate("こんにちは", "ja", "en")
        assert result.translated_text == "Hello"
        assert result.confidence == 0.85

    @patch("manga_translator.components.translator.DeepLEngine._get_translator")
    @patch("manga_translator.components.translator.DeepLEngine.is_available", return_value=True)
    def test_translate_failure(self, mock_avail, mock_translator):
        mock_translator.return_value.translate_text.side_effect = RuntimeError("API error")

        engine = DeepLEngine(api_key="fake")
        result = engine.translate("test", "ja", "en")
        assert result.error is not None

    @patch("manga_translator.components.translator.DeepLEngine._get_translator")
    @patch("manga_translator.components.translator.DeepLEngine.is_available", return_value=True)
    def test_batch_translate(self, mock_avail, mock_translator):
        mock_results = [MagicMock(text="Hello"), MagicMock(text="World")]
        mock_translator.return_value.translate_text.return_value = mock_results

        engine = DeepLEngine(api_key="fake")
        results = engine.translate_batch(["こんにちは", "世界"], "ja", "en")
        assert len(results) == 2
        assert results[0].translated_text == "Hello"

    @patch("manga_translator.components.translator.DeepLEngine._get_translator")
    @patch("manga_translator.components.translator.DeepLEngine.is_available", return_value=True)
    def test_batch_failure_fallback(self, mock_avail, mock_translator):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if len(args) > 0 and isinstance(args[0], list):
                raise RuntimeError("batch failed")
            mock_r = MagicMock()
            mock_r.text = "translated"
            return mock_r

        mock_translator.return_value.translate_text.side_effect = side_effect

        engine = DeepLEngine(api_key="fake")
        results = engine.translate_batch(["a", "b"], "ja", "en")
        assert len(results) == 2


class TestArgosEngine:
    def test_availability(self):
        engine = ArgosEngine()
        assert isinstance(engine.is_available(), bool)


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

    def test_translate_whitespace_only(self):
        manager = TranslationManager()
        result = manager.translate("   ", "ja", "en")
        assert result.translated_text == ""

    def test_same_language_passthrough(self):
        manager = TranslationManager()
        result = manager.translate("Hello World", "en", "en")
        assert result.translated_text == "Hello World"
        assert result.engine_used == "passthrough"
        assert result.confidence == 1.0

    def test_translate_returns_result(self):
        manager = TranslationManager()
        result = manager.translate("テスト", "ja", "en")
        assert isinstance(result, TranslationResult)
        assert result.source_text == "テスト"
        assert result.engine_used is not None

    def test_batch_translate_empty(self):
        manager = TranslationManager()
        results = manager.translate_batch([], "ja", "en")
        assert results == []

    def test_no_engines_available(self):
        manager = TranslationManager()
        # Force all engines unavailable
        for e in manager._engines.values():
            if hasattr(e, '_api_key'):
                e._api_key = ""
            if hasattr(e, '_available'):
                e._available = False
        # Monkey-patch is_available
        for e in manager._engines.values():
            e.is_available = lambda: False
        result = manager.translate("test", "ja", "en")
        assert result.error is not None

    def test_engine_order(self):
        manager = TranslationManager(primary_engine="openai")
        order = manager._get_engine_order()
        assert isinstance(order, list)

    def test_batch_no_engines(self):
        manager = TranslationManager()
        for e in manager._engines.values():
            e.is_available = lambda: False
        results = manager.translate_batch(["a", "b"], "ja", "en")
        assert len(results) == 2
        assert all(r.error is not None for r in results)
