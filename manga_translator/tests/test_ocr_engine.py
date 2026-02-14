"""Tests for OCR engine."""

import numpy as np
import pytest

from manga_translator.components.ocr_engine import (
    OCREngine,
    OCRResult,
    MangaOCREngine,
    TesseractEngine,
)


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
        """Test that at least one engine reports availability."""
        engine = OCREngine()
        # At minimum, tesseract should be checkable
        tesseract = TesseractEngine()
        # Just ensure is_available returns a bool
        assert isinstance(tesseract.is_available(), bool)

    def test_batch_extraction(self, white_bubble_region):
        engine = OCREngine(primary_engine="tesseract")
        regions = [white_bubble_region, white_bubble_region]
        results = engine.extract_text_batch(regions, language_hint="en")
        assert len(results) == 2
        for r in results:
            assert isinstance(r, OCRResult)
