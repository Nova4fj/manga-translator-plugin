"""Tests for inpainting component."""

import numpy as np
import pytest

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
