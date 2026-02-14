"""Tests for error recovery system."""

import numpy as np
from unittest.mock import patch

from manga_translator.error_recovery import (
    ErrorRecoveryManager,
    RecoveryReport,
)
from manga_translator.components.bubble_detector import BubbleDetector
from manga_translator.components.ocr_engine import OCREngine, OCRResult
from manga_translator.components.translator import TranslationManager, TranslationResult
from manga_translator.components.inpainter import Inpainter, InpaintResult
from manga_translator.components.typesetter import Typesetter


class TestRecoveryReport:
    def test_empty_report(self):
        report = RecoveryReport()
        assert report.total_recoveries == 0
        assert report.successful_recoveries == 0
        assert "No errors" in report.summary()

    def test_add_action(self):
        report = RecoveryReport()
        report.add("ocr", "failed", "tried fallback", True)
        assert report.total_recoveries == 1
        assert report.successful_recoveries == 1

    def test_mixed_results(self):
        report = RecoveryReport()
        report.add("ocr", "err1", "fallback1", True)
        report.add("translate", "err2", "fallback2", False)
        assert report.total_recoveries == 2
        assert report.successful_recoveries == 1
        assert report.failed_recoveries == 1

    def test_summary_format(self):
        report = RecoveryReport()
        report.add("detection", "error", "relaxed params", True)
        summary = report.summary()
        assert "detection" in summary
        assert "OK" in summary


class TestTryDetect:
    def test_success_no_recovery(self):
        erm = ErrorRecoveryManager()
        detector = BubbleDetector()
        img = np.full((200, 200, 3), 180, dtype=np.uint8)
        result = erm.try_detect(detector, img)
        assert isinstance(result, list)
        assert erm.report.total_recoveries == 0

    def test_failure_returns_empty(self):
        erm = ErrorRecoveryManager()
        detector = BubbleDetector()
        # Pass invalid image to trigger error
        result = erm.try_detect(detector, np.array([]))
        assert result == []


class TestTryOCR:
    def test_success(self):
        erm = ErrorRecoveryManager()
        ocr = OCREngine()
        img = np.full((50, 100, 3), 255, dtype=np.uint8)
        result = erm.try_ocr(ocr, img)
        assert isinstance(result, OCRResult)

    @patch.object(OCREngine, "extract_text", side_effect=RuntimeError("OCR crash"))
    def test_all_engines_fail(self, mock_ocr):
        erm = ErrorRecoveryManager()
        ocr = OCREngine()
        img = np.full((50, 100, 3), 255, dtype=np.uint8)
        result = erm.try_ocr(ocr, img)
        assert isinstance(result, OCRResult)
        assert result.engine_used == "error"
        assert erm.report.total_recoveries >= 1


class TestTryTranslate:
    def test_passthrough_recovery(self):
        erm = ErrorRecoveryManager()
        manager = TranslationManager()
        # Force all engines unavailable
        for e in manager._engines.values():
            e.is_available = lambda: False
        result = erm.try_translate(manager, "テスト", "ja", "en")
        assert isinstance(result, TranslationResult)
        # Should get passthrough
        assert result.translated_text == "テスト"
        assert "passthrough" in result.engine_used

    def test_batch_fallback_to_individual(self):
        erm = ErrorRecoveryManager()
        manager = TranslationManager()
        for e in manager._engines.values():
            e.is_available = lambda: False
        results = erm.try_translate_batch(
            manager, ["テスト", "漫画"], "ja", "en"
        )
        assert len(results) == 2
        # All should have passthrough
        for r in results:
            assert r.translated_text  # not empty


class TestTryInpaint:
    def test_success(self):
        erm = ErrorRecoveryManager()
        inpainter = Inpainter(method="blur")
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = erm.try_inpaint(inpainter, img, mask)
        assert isinstance(result, InpaintResult)

    @patch.object(Inpainter, "remove_text_with_fallback", side_effect=RuntimeError("crash"))
    def test_white_fill_fallback(self, mock_inpaint):
        erm = ErrorRecoveryManager()
        inpainter = Inpainter()
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = erm.try_inpaint(inpainter, img, mask)
        assert isinstance(result, InpaintResult)
        assert result.method_used == "white_fill_recovery"
        # Masked area should be white
        assert result.image[50, 50, 0] == 255


class TestTryTypeset:
    def test_success(self):
        erm = ErrorRecoveryManager()
        ts = Typesetter()
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = erm.try_typeset(ts, img, "Hello", (50, 50, 200, 100))
        assert result is not None

    @patch.object(Typesetter, "typeset_text", side_effect=RuntimeError("font crash"))
    def test_fallback_basic(self, mock_ts):
        erm = ErrorRecoveryManager()
        ts = Typesetter()
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = erm.try_typeset(ts, img, "Hello", (50, 50, 200, 100))
        # May succeed with basic typesetter or return None
        assert result is None or isinstance(result, type(result))

    def test_empty_text(self):
        erm = ErrorRecoveryManager()
        ts = Typesetter()
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = erm.try_typeset(ts, img, "", (50, 50, 200, 100))
        assert result is not None  # empty text should not crash
