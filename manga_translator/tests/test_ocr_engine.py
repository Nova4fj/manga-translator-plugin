"""Tests for OCR engine."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from manga_translator.components.ocr_engine import (
    OCREngine,
    OCRResult,
    MangaOCREngine,
    TesseractEngine,
    PaddleOCREngine,
)


class TestOCRResult:
    def test_is_valid_with_text(self):
        r = OCRResult(text="hello", confidence=0.9, language="en", engine_used="test")
        assert r.is_valid is True

    def test_is_valid_empty(self):
        r = OCRResult(text="", confidence=0.0, language="en", engine_used="test")
        assert r.is_valid is False

    def test_is_valid_with_error(self):
        r = OCRResult(text="hello", confidence=0.9, language="en", engine_used="test", error="oops")
        assert r.is_valid is False

    def test_empty_factory(self):
        r = OCRResult.empty("myengine", error="fail")
        assert r.text == ""
        assert r.confidence == 0.0
        assert r.engine_used == "myengine"
        assert r.error == "fail"


class TestMangaOCREngine:
    def test_not_available_without_install(self):
        engine = MangaOCREngine()
        # Just check it returns a bool
        assert isinstance(engine.is_available(), bool)

    def test_extract_when_unavailable(self):
        engine = MangaOCREngine()
        engine._available = False
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.error is not None

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_extract_grayscale(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock(return_value="テスト")
        img = np.zeros((50, 50), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.text == "テスト"
        assert result.confidence == 0.9

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_extract_rgba(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock(return_value="hello")
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.text == "hello"

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_extract_rgb(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock(return_value="word")
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.text == "word"

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_extract_empty_result(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock(return_value="")
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.confidence == 0.0

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_extract_exception(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock(side_effect=RuntimeError("model crash"))
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.error is not None

    @patch("manga_translator.components.ocr_engine.MangaOCREngine.is_available", return_value=True)
    def test_unsupported_shape(self, mock_avail):
        engine = MangaOCREngine()
        engine._model = MagicMock()
        img = np.zeros((50, 50, 5), dtype=np.uint8)  # 5 channels - unsupported
        result = engine.extract_text(img)
        assert result.error is not None


class TestTesseractEngine:
    def test_availability_check(self):
        engine = TesseractEngine()
        assert isinstance(engine.is_available(), bool)

    def test_extract_when_unavailable(self):
        engine = TesseractEngine()
        engine._available = False
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.error is not None

    def test_extract_real_text(self):
        """Test real Tesseract extraction if available."""
        import cv2
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract not available")
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Hello", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        result = engine.extract_text(img, language_hint="en")
        assert result.text.strip()  # Should find some text
        assert result.confidence > 0

    def test_extract_grayscale(self):
        """Test with grayscale image."""
        import cv2
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract not available")
        img = np.full((100, 300), 255, dtype=np.uint8)
        cv2.putText(img, "Test", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        result = engine.extract_text(img, language_hint="en")
        assert isinstance(result, OCRResult)

    def test_extract_rgba(self):
        """Test with RGBA image."""
        import cv2
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract not available")
        img = np.full((100, 300, 4), 255, dtype=np.uint8)
        cv2.putText(img, "Test", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 3)
        result = engine.extract_text(img, language_hint="en")
        assert isinstance(result, OCRResult)


class TestPaddleOCREngine:
    def test_not_available(self):
        engine = PaddleOCREngine()
        # Just check it returns a bool
        assert isinstance(engine.is_available(), bool)

    def test_extract_when_unavailable(self):
        engine = PaddleOCREngine()
        engine._available = False
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.error is not None


class TestOCREngine:
    def test_init(self):
        engine = OCREngine()
        assert engine is not None

    def test_extract_text_returns_result(self, white_bubble_region):
        engine = OCREngine(primary_engine="tesseract")
        result = engine.extract_text(white_bubble_region, language_hint="en")
        assert isinstance(result, OCRResult)
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_image(self):
        engine = OCREngine()
        empty = np.zeros((10, 10, 3), dtype=np.uint8)
        result = engine.extract_text(empty)
        assert isinstance(result, OCRResult)

    def test_engine_availability(self):
        OCREngine()
        tesseract = TesseractEngine()
        assert isinstance(tesseract.is_available(), bool)

    def test_batch_extraction(self, white_bubble_region):
        engine = OCREngine(primary_engine="tesseract")
        regions = [white_bubble_region, white_bubble_region]
        results = engine.extract_text_batch(regions, language_hint="en")
        assert len(results) == 2
        for r in results:
            assert isinstance(r, OCRResult)

    def test_no_engines_available(self):
        engine = OCREngine()
        # Force all engines unavailable
        for e in engine._engines.values():
            e._available = False
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = engine.extract_text(img)
        assert result.error is not None

    def test_available_engines_property(self):
        engine = OCREngine()
        avail = engine.available_engines
        assert isinstance(avail, list)

    def test_fallback_order(self):
        engine = OCREngine(primary_engine="tesseract")
        order = engine._fallback_order()
        assert isinstance(order, list)


class TestGuessLanguage:
    def test_guess_japanese(self):
        # Text with hiragana
        result = OCREngine._guess_cjk_language("こんにちは世界")
        assert result == "ja"

    def test_guess_korean(self):
        result = OCREngine._guess_cjk_language("안녕하세요")
        assert result == "ko"

    def test_guess_chinese(self):
        result = OCREngine._guess_cjk_language("你好世界")
        assert result == "zh"

    def test_guess_english(self):
        result = OCREngine._guess_cjk_language("Hello World")
        assert result == "en"

    def test_guess_empty(self):
        result = OCREngine._guess_cjk_language("")
        assert result == "unknown"
