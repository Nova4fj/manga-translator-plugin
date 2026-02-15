"""Tests for inpainting quality on real manga images.

These tests verify that the text removal pipeline:
1. Removes text from speech bubbles (no residual dark strokes)
2. Does not create white smudges that bleed over bubble outlines
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from manga_translator.manga_translator import MangaTranslationPipeline, PageTranslationResult
from manga_translator.components.ocr_engine import OCRResult
from manga_translator.components.translator import TranslationResult

# ---------------------------------------------------------------------------
# Paths to real test images
# ---------------------------------------------------------------------------
_TEST_DIR = Path(__file__).parent
_IMAGE_P0 = _TEST_DIR / "141005506_p0_master1200.jpg"
_IMAGE_P1 = _TEST_DIR / "141005506_p1_master1200.jpg"

# ---------------------------------------------------------------------------
# Mock helpers (avoid real OCR / translation API calls)
# ---------------------------------------------------------------------------
_OCR_PATCH = "manga_translator.components.ocr_engine.OCREngine.extract_text"
_TRANS_PATCH = "manga_translator.components.translator.TranslationManager.translate_batch"


def _mock_ocr(region, language_hint=None):
    return OCRResult(
        text="テスト",
        confidence=0.95,
        language=language_hint or "ja",
        engine_used="mock",
    )


def _mock_translate_batch(texts, source_lang, target_lang):
    return [
        TranslationResult(
            source_text=t,
            translated_text=f"translated_{t}",
            source_language=source_lang,
            target_language=target_lang,
            engine_used="mock",
            confidence=0.9,
        )
        for t in texts
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[_IMAGE_P0, _IMAGE_P1], ids=["p0", "p1"])
def manga_image(request):
    """Load a real manga test image; skip if file is missing."""
    path = request.param
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"Failed to read image: {path}")
    return img


@pytest.fixture
def cleanup_result(manga_image):
    """Run the full cleanup pipeline (mocked OCR/translation) and return
    (original, cleaned, result) tuple."""
    with patch(_OCR_PATCH, side_effect=_mock_ocr), \
         patch(_TRANS_PATCH, side_effect=_mock_translate_batch):
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(
            manga_image,
            source_lang="ja",
            target_lang="en",
            cleanup_only=True,
        )
    return manga_image, result.cleaned_image, result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInpaintingQuality:
    """Verify text removal and artifact-free inpainting on real images."""

    def test_pipeline_succeeds(self, cleanup_result):
        """Pipeline completes without errors on real manga."""
        original, cleaned, result = cleanup_result
        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) >= 1, "Should detect at least one bubble"

    def test_cleaned_image_differs(self, cleanup_result):
        """Cleaned image should differ from original (text was removed)."""
        original, cleaned, result = cleanup_result
        assert cleaned is not None, "cleaned_image should not be None"
        assert not np.array_equal(original, cleaned)

    def test_no_residual_text_in_bubbles(self, cleanup_result):
        """Bubble interiors should have very few dark pixels after cleanup.

        For each detected bubble, extract the interior region from the
        cleaned image and check that the fraction of dark pixels is low.
        """
        original, cleaned, result = cleanup_result
        if cleaned is None:
            pytest.skip("No cleaned image produced")

        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        for bt in result.bubbles:
            bubble = bt.bubble
            if bubble.mask is None:
                continue
            x, y, w, h = bubble.bbox
            local_mask = bubble.mask[y : y + h, x : x + w]
            local_gray = gray[y : y + h, x : x + w]

            # Erode mask to stay away from outlines
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            interior = cv2.erode(local_mask, erode_k, iterations=1)
            interior_pixels = local_gray[interior > 0]

            if len(interior_pixels) == 0:
                continue

            # Dark pixels (< 100) inside the bubble indicate residual text
            dark_ratio = np.count_nonzero(interior_pixels < 100) / len(interior_pixels)
            assert dark_ratio < 0.15, (
                f"Bubble {bubble.id}: {dark_ratio:.1%} dark pixels in interior "
                f"(expected < 15%) — possible residual text"
            )

    def test_new_bubble_has_clean_outline(self, cleanup_result):
        """The redrawn bubble should have a visible dark outline.

        The pipeline draws a new ellipse with a black contour.
        Verify that a wide band around the bubble region contains
        dark outline pixels (the ellipse may not align exactly with
        the original mask boundary).
        """
        original, cleaned, result = cleanup_result
        if cleaned is None:
            pytest.skip("No cleaned image produced")

        clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        for bt in result.bubbles:
            bubble = bt.bubble
            if bubble.contour is None or len(bubble.contour) < 5:
                continue
            x, y, w, h = bubble.bbox
            local_mask = bubble.mask[y : y + h, x : x + w]

            # Wide band around the bubble (the new ellipse outline
            # may be shifted vs the original mask boundary).
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            band = cv2.subtract(
                cv2.dilate(local_mask, dk, iterations=1),
                cv2.erode(local_mask, ek, iterations=1),
            )

            band_pixels = clean_gray[y : y + h, x : x + w][band > 0]
            if len(band_pixels) == 0:
                continue

            # There should be some dark pixels in this band
            # (the drawn black contour of the new ellipse).
            dark_ratio = np.count_nonzero(band_pixels < 80) / len(band_pixels)
            assert dark_ratio > 0.001, (
                f"Bubble {bubble.id}: no dark outline pixels found "
                f"({dark_ratio:.1%}) — new bubble may be missing its outline"
            )

    def test_new_bubble_interior_is_uniform(self, cleanup_result):
        """The redrawn bubble interior should be mostly uniform
        (filled with a single background colour, no text remnants).
        """
        original, cleaned, result = cleanup_result
        if cleaned is None:
            pytest.skip("No cleaned image produced")

        clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        for bt in result.bubbles:
            bubble = bt.bubble
            if bubble.mask is None:
                continue
            x, y, w, h = bubble.bbox
            local_mask = bubble.mask[y : y + h, x : x + w]

            # Erode well inside to avoid outline pixels
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            interior = cv2.erode(local_mask, ek, iterations=1)
            interior_pixels = clean_gray[y : y + h, x : x + w][interior > 0]

            if len(interior_pixels) == 0:
                continue

            # Standard deviation should be low (uniform fill)
            std = np.std(interior_pixels.astype(np.float64))
            assert std < 15, (
                f"Bubble {bubble.id}: interior std={std:.1f} "
                f"(expected < 15) — fill may not be uniform"
            )
