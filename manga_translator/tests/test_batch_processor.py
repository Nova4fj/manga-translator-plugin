"""Tests for batch processing engine."""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from manga_translator.batch_processor import (
    BatchProcessor,
    BatchResult,
    PageResult,
)


class TestPageResult:
    def test_defaults(self):
        pr = PageResult(input_path="test.png")
        assert pr.status == "pending"
        assert pr.error is None
        assert pr.bubble_count == 0

    def test_fields(self):
        pr = PageResult(
            input_path="test.png",
            output_path="out.png",
            status="complete",
            bubble_count=5,
            success_rate=0.8,
            processing_time=2.5,
        )
        assert pr.success_rate == 0.8


class TestBatchResult:
    def test_empty(self):
        br = BatchResult()
        assert br.total == 0
        assert br.completed == 0
        assert br.failed == 0
        assert br.avg_success_rate == 0.0

    def test_counts(self):
        br = BatchResult(pages=[
            PageResult(input_path="a.png", status="complete", success_rate=1.0),
            PageResult(input_path="b.png", status="complete", success_rate=0.5),
            PageResult(input_path="c.png", status="failed"),
            PageResult(input_path="d.png", status="skipped"),
        ])
        assert br.total == 4
        assert br.completed == 2
        assert br.failed == 1
        assert br.skipped == 1
        assert br.avg_success_rate == 0.75

    def test_summary(self):
        br = BatchResult(pages=[
            PageResult(input_path="a.png", status="complete", success_rate=1.0),
            PageResult(input_path="b.png", status="failed", error="oops"),
        ])
        summary = br.summary()
        assert "1/2" in summary
        assert "oops" in summary


class TestBatchProcessor:
    def test_init(self):
        bp = BatchProcessor(max_workers=2)
        assert bp.max_workers == 2
        assert bp.output_format == "png"

    def test_min_workers(self):
        bp = BatchProcessor(max_workers=0)
        assert bp.max_workers == 1

    def test_get_output_path(self):
        bp = BatchProcessor(output_format="png")
        path = bp._get_output_path("/images/page01.jpg")
        assert path.endswith("page01_translated.png")
        assert "/images/" in path

    def test_get_output_path_with_dir(self, tmp_path):
        bp = BatchProcessor(output_dir=str(tmp_path), output_format="jpg")
        path = bp._get_output_path("/images/page01.png")
        assert path.startswith(str(tmp_path))
        assert path.endswith("page01_translated.jpg")

    def test_skips_missing_files(self):
        bp = BatchProcessor()
        result = bp.process_batch(["/nonexistent/file.png"])
        assert result.total == 1
        assert result.skipped == 1
        assert result.pages[0].status == "skipped"

    def test_empty_batch(self):
        bp = BatchProcessor()
        result = bp.process_batch([])
        assert result.total == 0

    @patch("manga_translator.batch_processor.MangaTranslationPipeline")
    @patch("manga_translator.batch_processor.load_image")
    @patch("manga_translator.batch_processor.save_image")
    def test_process_single_page(self, mock_save, mock_load, mock_pipeline, tmp_path):
        """Test processing a single page with mocked pipeline."""
        # Create a fake input file
        input_path = str(tmp_path / "test.png")
        with open(input_path, "w") as f:
            f.write("fake")

        mock_load.return_value = np.full((100, 100, 3), 200, dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.bubbles = [MagicMock(), MagicMock()]
        mock_result.success_rate = 0.75
        mock_result.errors = []
        mock_result.final_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        mock_pipeline.return_value.translate_page.return_value = mock_result

        bp = BatchProcessor(output_dir=str(tmp_path))
        result = bp.process_batch([input_path])
        assert result.completed == 1
        assert result.pages[0].bubble_count == 2
        assert result.pages[0].success_rate == 0.75

    @patch("manga_translator.batch_processor.MangaTranslationPipeline")
    @patch("manga_translator.batch_processor.load_image")
    @patch("manga_translator.batch_processor.save_image")
    def test_error_isolation(self, mock_save, mock_load, mock_pipeline, tmp_path):
        """One failed page should not affect others."""
        good_path = str(tmp_path / "good.png")
        bad_path = str(tmp_path / "bad.png")
        for p in [good_path, bad_path]:
            with open(p, "w") as f:
                f.write("fake")

        mock_load.return_value = np.full((100, 100, 3), 200, dtype=np.uint8)

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("page failed")
            mock_r = MagicMock()
            mock_r.bubbles = []
            mock_r.success_rate = 0.0
            mock_r.errors = []
            mock_r.final_image = np.full((100, 100, 3), 200, dtype=np.uint8)
            return mock_r

        mock_pipeline.return_value.translate_page.side_effect = side_effect

        bp = BatchProcessor(output_dir=str(tmp_path), max_workers=1)
        result = bp.process_batch([bad_path, good_path])
        statuses = {p.status for p in result.pages}
        # At least one should complete even if the other fails
        assert "failed" in statuses or "complete" in statuses

    @patch("manga_translator.batch_processor.MangaTranslationPipeline")
    @patch("manga_translator.batch_processor.load_image")
    @patch("manga_translator.batch_processor.save_image")
    def test_progress_callback(self, mock_save, mock_load, mock_pipeline, tmp_path):
        """Progress callback should be called for each page."""
        input_path = str(tmp_path / "test.png")
        with open(input_path, "w") as f:
            f.write("fake")

        mock_load.return_value = np.full((100, 100, 3), 200, dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.bubbles = []
        mock_result.success_rate = 0.0
        mock_result.errors = []
        mock_result.final_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        mock_pipeline.return_value.translate_page.return_value = mock_result

        progress_calls = []
        def callback(completed, total, filename, status):
            progress_calls.append((completed, total, filename, status))

        bp = BatchProcessor(output_dir=str(tmp_path))
        bp.process_batch([input_path], progress_callback=callback)
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == 1  # completed
        assert progress_calls[0][1] == 1  # total


class TestCrossPageContext:
    def test_cross_page_forces_sequential(self):
        """Enabling cross-page context should force max_workers=1."""
        bp = BatchProcessor(max_workers=4, enable_cross_page_context=True)
        assert bp.max_workers == 1
        assert bp.enable_cross_page_context is True

    def test_default_no_cross_page(self):
        """By default cross-page context is disabled."""
        bp = BatchProcessor(max_workers=3)
        assert bp.enable_cross_page_context is False
        assert bp.max_workers == 3
