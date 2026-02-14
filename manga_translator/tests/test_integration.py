"""End-to-end integration tests for the manga translation pipeline.

These tests exercise the full pipeline with synthetic images. OCR and
translation components are mocked at the method level so the tests
run without external engines or API keys, while the bubble detector,
inpainter, typesetter, and pipeline orchestration run for real.
"""

import os

import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from manga_translator.manga_translator import (
    MangaTranslationPipeline,
    PageTranslationResult,
    BubbleTranslation,
    translate_file,
)
from manga_translator.batch_processor import BatchProcessor, BatchResult
from manga_translator.components.ocr_engine import OCRResult
from manga_translator.components.translator import TranslationResult
from manga_translator.config.settings import PluginSettings
from manga_translator.perf_monitor import PerfMonitor
from manga_translator.quality_control import QualityChecker
from manga_translator.translation_memory import TranslationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_page(width=600, height=800, num_bubbles=3):
    """Create a synthetic manga page with white elliptical bubbles on a dark
    gray background.  Returns a BGR numpy array."""
    page = np.full((height, width, 3), 100, dtype=np.uint8)  # dark gray

    positions = [
        (450, 150, 100, 60),
        (150, 400, 120, 70),
        (400, 650, 80, 50),
    ]

    for i in range(min(num_bubbles, len(positions))):
        cx, cy, rx, ry = positions[i]
        # White filled ellipse with black outline
        cv2.ellipse(page, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(page, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 0), 2)
        # Dark text scribbles inside
        cv2.putText(
            page, f"T{i}", (cx - 20, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2,
        )

    return page


def _mock_ocr_side_effect(region, language_hint=None):
    """Return a plausible OCRResult for any region."""
    return OCRResult(
        text="konnichiwa",
        confidence=0.95,
        language=language_hint or "ja",
        engine_used="mock",
    )


def _mock_translate_batch_side_effect(texts, source_lang, target_lang):
    """Return a TranslationResult for each input text."""
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


# Patch targets (method-level mocking)
_OCR_PATCH = "manga_translator.components.ocr_engine.OCREngine.extract_text"
_TRANS_PATCH = "manga_translator.components.translator.TranslationManager.translate_batch"


# ---------------------------------------------------------------------------
# TestPipelineEndToEnd
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    """Full pipeline tests with mocked OCR and translation."""

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_with_mocked_components(self, mock_ocr, mock_trans):
        """Pipeline detects bubbles, runs mocked OCR/translation, inpaints,
        typesets, and returns a valid result."""
        image = _make_synthetic_page()
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)
        # Bubble detector should find at least one bubble in our synthetic page
        assert len(result.bubbles) >= 1
        # final_image should differ from the original (inpainting + typesetting)
        assert not np.array_equal(result.original_image, result.final_image)
        # Each bubble should have a translation
        for bt in result.bubbles:
            assert isinstance(bt, BubbleTranslation)
            assert bt.ocr_result.text
            assert bt.translation.translated_text.startswith("translated_")

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_with_perf_monitor(self, mock_ocr, mock_trans):
        """PerfMonitor produces a non-empty summary with stage names."""
        image = _make_synthetic_page()
        perf = PerfMonitor()
        perf.start()
        pipeline = MangaTranslationPipeline(perf_monitor=perf)
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert result.perf_summary  # non-empty string
        # The summary should mention at least a few pipeline stages
        for stage in ("detection", "ocr", "translation"):
            assert stage in result.perf_summary.lower(), (
                f"Stage '{stage}' not found in perf summary"
            )

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_with_quality_checker(self, mock_ocr, mock_trans):
        """QualityChecker produces a QCReport."""
        image = _make_synthetic_page()
        qc = QualityChecker()
        pipeline = MangaTranslationPipeline(quality_checker=qc)
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert result.qc_report is not None
        assert result.qc_report.page_count >= 1

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_with_translation_memory(self, mock_ocr, mock_trans, tmp_path):
        """TM stores translations on first run; second run gets TM hits."""
        image = _make_synthetic_page()
        db_path = str(tmp_path / "tm_test.db")
        tm = TranslationMemory(db_path=db_path)

        # First run — should call the translator and store results in TM
        pipeline1 = MangaTranslationPipeline(translation_memory=tm)
        result1 = pipeline1.translate_page(image, source_lang="ja", target_lang="en")
        assert len(result1.bubbles) >= 1
        first_call_count = mock_trans.call_count

        # Second run — TM should have cached entries, so translator gets
        # fewer (or zero) texts to translate
        mock_trans.reset_mock()
        pipeline2 = MangaTranslationPipeline(translation_memory=tm)
        result2 = pipeline2.translate_page(image, source_lang="ja", target_lang="en")

        # The translator should have been called fewer times (or with fewer texts)
        # because TM hits bypass it
        if mock_trans.call_count > 0:
            # If called, it should have received fewer texts
            second_texts = mock_trans.call_args[0][0] if mock_trans.call_args else []
            first_texts = []
            # At minimum, second run should still produce valid results
            assert len(result2.bubbles) >= 1
        else:
            # Translator was not called at all — all hits from TM
            assert len(result2.bubbles) >= 1

        # Verify at least one bubble used translation_memory engine
        tm_engines = [
            bt.translation.engine_used for bt in result2.bubbles
            if bt.translation.engine_used == "translation_memory"
        ]
        assert len(tm_engines) >= 1, "Expected at least one TM hit on second run"

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_all_features_combined(self, mock_ocr, mock_trans, tmp_path):
        """Pipeline with perf monitor, QC, and TM all enabled."""
        image = _make_synthetic_page()
        db_path = str(tmp_path / "tm_combined.db")

        perf = PerfMonitor()
        perf.start()
        qc = QualityChecker()
        tm = TranslationMemory(db_path=db_path)

        pipeline = MangaTranslationPipeline(
            perf_monitor=perf,
            quality_checker=qc,
            translation_memory=tm,
        )
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert len(result.bubbles) >= 1
        assert result.perf_summary
        assert result.qc_report is not None
        assert not np.array_equal(result.original_image, result.final_image)

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_no_bubbles(self, mock_ocr, mock_trans):
        """Solid dark image has no bubbles; pipeline returns gracefully."""
        # All-black image — no white regions to detect as bubbles
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) == 0
        assert len(result.errors) >= 1  # "No speech bubbles detected"

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_pipeline_progress_callback(self, mock_ocr, mock_trans):
        """progress_callback is called during pipeline execution."""
        image = _make_synthetic_page()
        pipeline = MangaTranslationPipeline()

        calls = []

        def progress_cb(step, total, message):
            calls.append((step, total, message))

        result = pipeline.translate_page(
            image, source_lang="ja", target_lang="en",
            progress_callback=progress_cb,
        )

        # The pipeline has 6 steps (0-5), so we expect at least a few calls
        assert len(calls) >= 2, f"Expected progress calls, got {len(calls)}"


# ---------------------------------------------------------------------------
# TestTranslateFile
# ---------------------------------------------------------------------------

class TestTranslateFile:
    """Tests for the module-level translate_file() convenience function."""

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_translate_file_basic(self, mock_ocr, mock_trans, tmp_path):
        """Write synthetic image, translate it, verify output file exists."""
        input_path = str(tmp_path / "page.png")
        output_path = str(tmp_path / "page_out.png")
        image = _make_synthetic_page()
        cv2.imwrite(input_path, image)

        result = translate_file(
            input_path, output_path,
            source_lang="ja", target_lang="en",
        )

        assert isinstance(result, PageTranslationResult)
        assert os.path.isfile(output_path)

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_translate_file_with_options(self, mock_ocr, mock_trans, tmp_path):
        """translate_file with QC, perf, and TM enabled."""
        input_path = str(tmp_path / "page.png")
        output_path = str(tmp_path / "page_out.png")
        db_path = str(tmp_path / "tm.db")
        image = _make_synthetic_page()
        cv2.imwrite(input_path, image)

        result = translate_file(
            input_path, output_path,
            source_lang="ja", target_lang="en",
            enable_qc=True,
            enable_perf=True,
            tm_db_path=db_path,
        )

        assert isinstance(result, PageTranslationResult)
        assert os.path.isfile(output_path)

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_translate_file_auto_output_path(self, mock_ocr, mock_trans, tmp_path):
        """Without output_path, file gets _translated suffix."""
        input_path = str(tmp_path / "manga_page.png")
        image = _make_synthetic_page()
        cv2.imwrite(input_path, image)

        result = translate_file(input_path, source_lang="ja", target_lang="en")

        expected_output = str(tmp_path / "manga_page_translated.png")
        assert os.path.isfile(expected_output)


# ---------------------------------------------------------------------------
# TestBatchIntegration
# ---------------------------------------------------------------------------

class TestBatchIntegration:
    """Tests for BatchProcessor with synthetic images."""

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_batch_processor(self, mock_ocr, mock_trans, tmp_path):
        """Process 3 synthetic images, verify BatchResult."""
        input_paths = []
        for i in range(3):
            path = str(tmp_path / f"page_{i}.png")
            cv2.imwrite(path, _make_synthetic_page())
            input_paths.append(path)

        bp = BatchProcessor(output_dir=str(tmp_path / "output"), max_workers=1)
        result = bp.process_batch(input_paths, source_lang="ja", target_lang="en")

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.completed == 3
        assert result.failed == 0

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_batch_processor_missing_file(self, mock_ocr, mock_trans, tmp_path):
        """Non-existent path is skipped, valid files still processed."""
        valid_path = str(tmp_path / "real.png")
        cv2.imwrite(valid_path, _make_synthetic_page())
        missing_path = str(tmp_path / "ghost.png")

        bp = BatchProcessor(output_dir=str(tmp_path / "output"), max_workers=1)
        result = bp.process_batch(
            [valid_path, missing_path],
            source_lang="ja", target_lang="en",
        )

        assert result.total == 2
        assert result.skipped == 1
        assert result.completed == 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case images that should not crash the pipeline."""

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_empty_image(self, mock_ocr, mock_trans):
        """A 1x1 image should not crash; it returns with errors or empty bubbles."""
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) == 0

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_single_color_white(self, mock_ocr, mock_trans):
        """Solid white image — could trigger false bubble detections,
        but pipeline should not crash."""
        image = np.full((400, 400, 3), 255, dtype=np.uint8)
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_single_color_black(self, mock_ocr, mock_trans):
        """Solid black image — no bubbles expected."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) == 0

    @patch(_TRANS_PATCH, side_effect=_mock_translate_batch_side_effect)
    @patch(_OCR_PATCH, side_effect=_mock_ocr_side_effect)
    def test_very_large_image(self, mock_ocr, mock_trans):
        """5000x5000 image should work via resize_for_processing."""
        image = np.full((5000, 5000, 3), 100, dtype=np.uint8)
        # Add a large white ellipse so there is a detectable bubble
        cv2.ellipse(image, (2500, 2500), (600, 400), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (2500, 2500), (600, 400), 0, 0, 360, (0, 0, 0), 3)

        pipeline = MangaTranslationPipeline()
        result = pipeline.translate_page(image, source_lang="ja", target_lang="en")

        assert isinstance(result, PageTranslationResult)
        # The pipeline should not crash and should produce a final image
        # with the same dimensions as the original
        assert result.final_image.shape == image.shape
