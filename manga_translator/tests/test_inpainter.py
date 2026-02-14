"""Tests for inpainting component."""

import numpy as np
from unittest.mock import patch

from manga_translator.components.inpainter import Inpainter, InpaintResult


class TestInpainter:
    def test_init(self):
        inpainter = Inpainter()
        assert inpainter is not None

    def test_create_text_mask(self, white_bubble_region):
        inpainter = Inpainter()
        bubble_mask = np.ones(white_bubble_region.shape[:2], dtype=np.uint8) * 255
        text_mask = inpainter.create_text_mask(white_bubble_region, bubble_mask)
        assert text_mask.shape == white_bubble_region.shape[:2]
        assert text_mask.dtype == np.uint8

    def test_remove_text_telea(self, white_bubble_region):
        inpainter = Inpainter(method="opencv_telea")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        # Mark center area as text
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text(white_bubble_region, mask)
        assert isinstance(result, InpaintResult)
        assert result.image.shape == white_bubble_region.shape
        assert result.method_used == "opencv_telea"

    def test_remove_text_ns(self, white_bubble_region):
        inpainter = Inpainter(method="opencv_ns")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text(white_bubble_region, mask)
        assert isinstance(result, InpaintResult)

    def test_remove_text_blur(self, white_bubble_region):
        inpainter = Inpainter(method="blur")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text(white_bubble_region, mask)
        assert isinstance(result, InpaintResult)

    def test_quality_score(self, white_bubble_region):
        inpainter = Inpainter()
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text(white_bubble_region, mask)
        assert 0.0 <= result.quality_score <= 1.0

    def test_background_complexity(self):
        inpainter = Inpainter()
        # Simple white background
        simple = np.full((100, 100, 3), 255, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        complexity = inpainter.analyze_background_complexity(simple, mask)
        assert 0.0 <= complexity <= 1.0

    def test_quality_has_texture_component(self):
        """Quality assessment should include texture consistency."""
        inpainter = Inpainter()
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = inpainter.remove_text(img, mask)
        # Uniform image should score high
        assert result.quality_score > 0.5


class TestRemoveTextWithFallback:
    def test_good_quality_no_fallback(self, white_bubble_region):
        """High quality result should not trigger fallback."""
        inpainter = Inpainter(method="opencv_telea")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text_with_fallback(
            white_bubble_region, mask, quality_threshold=0.3
        )
        assert isinstance(result, InpaintResult)
        assert result.quality_score >= 0.0

    def test_low_threshold_no_fallback(self, white_bubble_region):
        """Very low threshold should accept any result."""
        inpainter = Inpainter(method="blur")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text_with_fallback(
            white_bubble_region, mask, quality_threshold=0.0
        )
        assert isinstance(result, InpaintResult)

    @patch("manga_translator.components.inpainter.Inpainter.is_neural_available", return_value=False)
    def test_fallback_telea_to_ns(self, mock_neural, white_bubble_region):
        """If telea quality is poor, should try ns."""
        inpainter = Inpainter(method="opencv_telea")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text_with_fallback(
            white_bubble_region, mask, quality_threshold=1.0  # impossible to meet
        )
        assert isinstance(result, InpaintResult)
        # Should have tried fallback
        assert result.method_used in ("opencv_telea", "opencv_ns")

    def test_fallback_returns_best(self, white_bubble_region):
        """Fallback should return the higher-quality result."""
        inpainter = Inpainter(method="blur")
        mask = np.zeros(white_bubble_region.shape[:2], dtype=np.uint8)
        mask[40:80, 30:170] = 255
        result = inpainter.remove_text_with_fallback(
            white_bubble_region, mask, quality_threshold=1.0
        )
        assert isinstance(result, InpaintResult)
        assert 0.0 <= result.quality_score <= 1.0
