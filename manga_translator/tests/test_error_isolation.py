"""Tests for per-bubble error isolation in the translation pipeline."""

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from manga_translator.components.bubble_detector import BubbleRegion
from manga_translator.components.ocr_engine import OCRResult
from manga_translator.components.translator import TranslationResult
from manga_translator.components.inpainter import InpaintResult
from manga_translator.components.typesetter import TypesetResult
from manga_translator.components.text_region_filter import TextRegionScore
from manga_translator.manga_translator import MangaTranslationPipeline
from manga_translator.config.settings import PluginSettings


def _make_bubble(bid, x=10, y=10, w=50, h=50):
    """Helper to create a BubbleRegion with a valid contour."""
    contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
    return BubbleRegion(
        id=bid,
        contour=contour,
        bbox=(x, y, w, h),
        center=(x + w // 2, y + h // 2),
        area=float(w * h),
        confidence=0.9,
        shape_type="rectangle",
        mask=None,
    )


def _make_image(width=200, height=200):
    """Create a simple test image."""
    return np.full((height, width, 3), 200, dtype=np.uint8)


def _good_ocr_result(text="hello", lang="ja"):
    return OCRResult(text=text, confidence=0.9, language=lang, engine_used="mock")


def _good_translation(source="hello", translated="hi", src="ja", tgt="en"):
    return TranslationResult(
        source_text=source,
        translated_text=translated,
        source_language=src,
        target_language=tgt,
        engine_used="mock",
        confidence=0.9,
    )


def _good_inpaint_result(region):
    return InpaintResult(image=region.copy(), method_used="mock", quality_score=0.9)


def _text_region_score():
    return TextRegionScore(
        has_text=True, confidence=0.9, edge_density=0.5,
        variance=100.0, text_line_score=0.8, region_type="text",
    )


@pytest.fixture
def pipeline():
    """Create a pipeline with default settings and no optional services."""
    settings = PluginSettings()
    return MangaTranslationPipeline(settings, perf_monitor=None, quality_checker=None, translation_memory=None)


class TestSingleBubbleOCRFailure:
    """OCR failure on one bubble should not prevent others from being processed."""

    def test_single_bubble_ocr_failure(self, pipeline):
        image = _make_image(300, 300)
        bubbles = [_make_bubble(0, 10, 10, 50, 50),
                   _make_bubble(1, 80, 80, 50, 50),
                   _make_bubble(2, 160, 160, 50, 50)]

        # OCR: succeed on bubble 0 and 2, raise on bubble 1
        ocr_call_count = [0]
        def mock_extract_text(region, language_hint=None):
            idx = ocr_call_count[0]
            ocr_call_count[0] += 1
            if idx == 1:
                raise RuntimeError("OCR engine crashed on bubble 1")
            return _good_ocr_result(text=f"text_{idx}", lang="ja")

        pipeline.detector.detect_bubbles = MagicMock(return_value=bubbles)
        pipeline.ocr.extract_text = MagicMock(side_effect=mock_extract_text)
        pipeline.text_filter.analyze_region = MagicMock(return_value=_text_region_score())
        pipeline.translator.translate_batch = MagicMock(
            side_effect=lambda texts, src, tgt: [
                _good_translation(source=t, translated=f"translated_{t}", src=src, tgt=tgt)
                for t in texts
            ]
        )
        pipeline.inpainter.create_text_mask = MagicMock(
            return_value=np.ones((50, 50), dtype=np.uint8) * 255
        )
        pipeline.inpainter.remove_text_with_fallback = MagicMock(
            side_effect=lambda region, mask, quality_threshold=0.5: _good_inpaint_result(region)
        )
        pipeline.typesetter.typeset_text = MagicMock(
            side_effect=lambda img, text, bbox, **kw: TypesetResult(
                image=img.copy(),
                text_mask=np.zeros(img.shape[:2], dtype=np.uint8),
                layout=MagicMock(),
            )
        )

        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        # Pipeline did not crash
        assert result is not None
        # Errors were recorded for bubble 1
        ocr_errors = [e for e in result.errors if "OCR failed for bubble 1" in e]
        assert len(ocr_errors) == 1
        # Bubbles 0 and 2 still got valid translations (only valid OCR results pass through)
        assert len(result.bubbles) == 2
        for bt in result.bubbles:
            assert bt.ocr_result.text.startswith("text_")
            assert bt.translation.translated_text.startswith("translated_")


class TestSingleBubbleInpaintFailure:
    """Inpainting failure on one bubble should not prevent others."""

    def test_single_bubble_inpaint_failure(self, pipeline):
        image = _make_image(300, 300)
        bubbles = [_make_bubble(0, 10, 10, 50, 50),
                   _make_bubble(1, 80, 80, 50, 50),
                   _make_bubble(2, 160, 160, 50, 50)]

        pipeline.detector.detect_bubbles = MagicMock(return_value=bubbles)
        pipeline.ocr.extract_text = MagicMock(
            side_effect=lambda region, language_hint=None: _good_ocr_result()
        )
        pipeline.text_filter.analyze_region = MagicMock(return_value=_text_region_score())
        pipeline.translator.translate_batch = MagicMock(
            side_effect=lambda texts, src, tgt: [_good_translation() for _ in texts]
        )

        # Inpainting: fail on 2nd call (bubble index 1)
        inpaint_call_count = [0]
        def mock_remove_text(region, mask, quality_threshold=0.5):
            idx = inpaint_call_count[0]
            inpaint_call_count[0] += 1
            if idx == 1:
                raise RuntimeError("Inpainting crashed on bubble 1")
            return _good_inpaint_result(region)

        pipeline.inpainter.create_text_mask = MagicMock(
            return_value=np.ones((50, 50), dtype=np.uint8) * 255
        )
        pipeline.inpainter.remove_text_with_fallback = MagicMock(side_effect=mock_remove_text)
        pipeline.typesetter.typeset_text = MagicMock(
            side_effect=lambda img, text, bbox, **kw: TypesetResult(
                image=img.copy(),
                text_mask=np.zeros(img.shape[:2], dtype=np.uint8),
                layout=MagicMock(),
            )
        )

        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert result is not None
        inpaint_errors = [e for e in result.errors if "Inpainting failed for bubble" in e]
        assert len(inpaint_errors) == 1
        # All 3 bubbles still present in result
        assert len(result.bubbles) == 3
        # The failed bubble has None for inpaint_result
        assert result.bubbles[1].inpaint_result is None
        # The others have actual results
        assert result.bubbles[0].inpaint_result is not None
        assert result.bubbles[2].inpaint_result is not None


class TestSingleBubbleTypesetFailure:
    """Typesetting failure on one bubble should not prevent others."""

    def test_single_bubble_typeset_failure(self, pipeline):
        image = _make_image(300, 300)
        bubbles = [_make_bubble(0, 10, 10, 50, 50),
                   _make_bubble(1, 80, 80, 50, 50),
                   _make_bubble(2, 160, 160, 50, 50)]

        pipeline.detector.detect_bubbles = MagicMock(return_value=bubbles)
        pipeline.ocr.extract_text = MagicMock(
            side_effect=lambda region, language_hint=None: _good_ocr_result()
        )
        pipeline.text_filter.analyze_region = MagicMock(return_value=_text_region_score())
        pipeline.translator.translate_batch = MagicMock(
            side_effect=lambda texts, src, tgt: [_good_translation() for _ in texts]
        )
        pipeline.inpainter.create_text_mask = MagicMock(
            return_value=np.ones((50, 50), dtype=np.uint8) * 255
        )
        pipeline.inpainter.remove_text_with_fallback = MagicMock(
            side_effect=lambda region, mask, quality_threshold=0.5: _good_inpaint_result(region)
        )

        # Typesetting: fail on 2nd call (bubble index 1)
        typeset_call_count = [0]
        def mock_typeset(img, text, bbox, **kw):
            idx = typeset_call_count[0]
            typeset_call_count[0] += 1
            if idx == 1:
                raise RuntimeError("Font rendering crashed on bubble 1")
            return TypesetResult(
                image=img.copy(),
                text_mask=np.zeros(img.shape[:2], dtype=np.uint8),
                layout=MagicMock(),
            )

        pipeline.typesetter.typeset_text = MagicMock(side_effect=mock_typeset)

        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert result is not None
        typeset_errors = [e for e in result.errors if "Typesetting failed for bubble" in e]
        assert len(typeset_errors) == 1
        assert len(result.bubbles) == 3
        # The failed bubble has None for typeset_result
        assert result.bubbles[1].typeset_result is None
        # The others have actual results
        assert result.bubbles[0].typeset_result is not None
        assert result.bubbles[2].typeset_result is not None


class TestAllBubblesFailGracefully:
    """All OCR failures should not crash the pipeline."""

    def test_all_bubbles_fail_gracefully(self, pipeline):
        image = _make_image(300, 300)
        bubbles = [_make_bubble(0, 10, 10, 50, 50),
                   _make_bubble(1, 80, 80, 50, 50),
                   _make_bubble(2, 160, 160, 50, 50)]

        pipeline.detector.detect_bubbles = MagicMock(return_value=bubbles)
        pipeline.text_filter.analyze_region = MagicMock(return_value=_text_region_score())
        # Every OCR call raises
        pipeline.ocr.extract_text = MagicMock(
            side_effect=RuntimeError("OCR engine unavailable")
        )

        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert result is not None
        # All 3 bubbles failed OCR
        ocr_errors = [e for e in result.errors if "OCR failed for bubble" in e]
        assert len(ocr_errors) == 3
        # No valid text detected, so pipeline returns early with no bubble translations
        assert len(result.bubbles) == 0
        # But no crash — we get a result with errors
        assert "No text detected in bubbles" in result.errors or len(ocr_errors) == 3
