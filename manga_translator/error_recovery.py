"""Comprehensive error recovery for every pipeline stage.

Provides graceful degradation with structured fallback strategies,
so the pipeline continues even when individual components fail.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any

import numpy as np

from manga_translator.components.bubble_detector import BubbleDetector, BubbleRegion
from manga_translator.components.ocr_engine import OCREngine, OCRResult
from manga_translator.components.translator import TranslationManager, TranslationResult
from manga_translator.components.inpainter import Inpainter, InpaintResult
from manga_translator.components.typesetter import Typesetter, TypesetResult

logger = logging.getLogger(__name__)


@dataclass
class RecoveryAction:
    """Record of a recovery action taken."""
    stage: str
    error: str
    action: str
    success: bool


@dataclass
class RecoveryReport:
    """Summary of all recovery actions taken during pipeline execution."""
    actions: List[RecoveryAction] = field(default_factory=list)

    @property
    def total_recoveries(self) -> int:
        return len(self.actions)

    @property
    def successful_recoveries(self) -> int:
        return sum(1 for a in self.actions if a.success)

    @property
    def failed_recoveries(self) -> int:
        return sum(1 for a in self.actions if not a.success)

    def add(self, stage: str, error: str, action: str, success: bool):
        self.actions.append(RecoveryAction(
            stage=stage, error=error, action=action, success=success,
        ))

    def summary(self) -> str:
        if not self.actions:
            return "No errors encountered."
        lines = [f"Recovery Report: {self.total_recoveries} actions taken"]
        for a in self.actions:
            status = "OK" if a.success else "FAILED"
            lines.append(f"  [{status}] {a.stage}: {a.action}")
        return "\n".join(lines)


class ErrorRecoveryManager:
    """Wraps pipeline stages with fallback strategies.

    Each ``try_*`` method attempts the primary operation, catches failures,
    and applies fallback strategies in order. Returns the best available
    result along with recovery information.
    """

    def __init__(self):
        self.report = RecoveryReport()

    def try_detect(
        self,
        detector: BubbleDetector,
        image: np.ndarray,
    ) -> List[BubbleRegion]:
        """Attempt bubble detection with fallback strategies.

        Fallbacks:
        1. Retry with relaxed parameters (lower min_area, higher max_area)
        2. Return empty list (skip detection entirely)
        """
        try:
            return detector.detect_bubbles(image)
        except Exception as e:
            logger.warning("Primary detection failed: %s", e)
            self.report.add("detection", str(e), "Trying relaxed parameters", True)

        # Fallback 1: Relaxed parameters
        try:
            relaxed = BubbleDetector(
                min_area=max(detector.min_area // 2, 100),
                max_area=min(detector.max_area * 2, 1000000),
                edge_sensitivity=max(detector.edge_sensitivity - 30, 30),
            )
            result = relaxed.detect_bubbles(image)
            self.report.add("detection", "relaxed params", "Relaxed detection succeeded", True)
            return result
        except Exception as e2:
            self.report.add("detection", str(e2), "Relaxed detection also failed", False)

        return []

    def try_ocr(
        self,
        ocr: OCREngine,
        region: np.ndarray,
        language_hint: str = "ja",
    ) -> OCRResult:
        """Attempt OCR with fallback strategies.

        Fallbacks:
        1. Try with different engine
        2. Return empty result
        """
        try:
            return ocr.extract_text(region, language_hint=language_hint)
        except Exception as e:
            self.report.add("ocr", str(e), "Primary OCR failed, trying fallback", True)

        # Fallback: try with tesseract specifically (most widely available)
        try:
            fallback_ocr = OCREngine(
                primary_engine="tesseract",
                confidence_threshold=0.3,
            )
            result = fallback_ocr.extract_text(region, language_hint=language_hint)
            self.report.add("ocr", "fallback", "Tesseract fallback succeeded", True)
            return result
        except Exception as e2:
            self.report.add("ocr", str(e2), "All OCR engines failed", False)

        return OCRResult(
            text="", confidence=0.0,
            language=language_hint, engine_used="error",
        )

    def try_translate(
        self,
        translator: TranslationManager,
        text: str,
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> TranslationResult:
        """Attempt translation with fallback strategies.

        Fallbacks:
        1. Manager already has multi-engine fallback built in
        2. Return passthrough (original text) as last resort
        """
        try:
            result = translator.translate(text, source_lang, target_lang)
            if result.translated_text:
                return result
            # Got empty translation — treat as failure
            raise RuntimeError(f"Empty translation: {result.error or 'unknown'}")
        except Exception as e:
            self.report.add(
                "translation", str(e),
                "Translation failed, using passthrough", True,
            )

        # Last resort: return the original text
        return TranslationResult(
            source_text=text,
            translated_text=text,  # passthrough
            source_language=source_lang,
            target_language=target_lang,
            engine_used="passthrough_recovery",
            confidence=0.1,
            error=f"All engines failed, using original text",
        )

    def try_translate_batch(
        self,
        translator: TranslationManager,
        texts: List[str],
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> List[TranslationResult]:
        """Batch translation with individual fallbacks."""
        try:
            results = translator.translate_batch(texts, source_lang, target_lang)
            if results and any(r.translated_text for r in results):
                return results
        except Exception as e:
            self.report.add("translation_batch", str(e), "Batch failed, trying individual", True)

        # Fall back to individual translations
        return [
            self.try_translate(translator, t, source_lang, target_lang)
            for t in texts
        ]

    def try_inpaint(
        self,
        inpainter: Inpainter,
        image: np.ndarray,
        text_mask: np.ndarray,
        quality_threshold: float = 0.5,
    ) -> InpaintResult:
        """Attempt inpainting with fallback strategies.

        Fallbacks:
        1. Quality-based fallback (built into remove_text_with_fallback)
        2. Simple white fill as absolute last resort
        """
        try:
            return inpainter.remove_text_with_fallback(
                image, text_mask, quality_threshold=quality_threshold,
            )
        except Exception as e:
            self.report.add("inpainting", str(e), "All inpainting methods failed", True)

        # Fallback: simple white fill
        try:
            result_image = image.copy()
            mask_bool = text_mask > 127
            if result_image.ndim == 3:
                result_image[mask_bool] = (255, 255, 255)
            else:
                result_image[mask_bool] = 255
            self.report.add("inpainting", "white fill", "Using simple white fill", True)
            return InpaintResult(
                image=result_image,
                method_used="white_fill_recovery",
                quality_score=0.2,
            )
        except Exception as e2:
            self.report.add("inpainting", str(e2), "White fill also failed", False)
            return InpaintResult(
                image=image.copy(),
                method_used="none",
                quality_score=0.0,
            )

    def try_typeset(
        self,
        typesetter: Typesetter,
        image: np.ndarray,
        text: str,
        bbox: tuple,
        bubble_mask=None,
        orientation: str = "auto",
        source_lang: str = "",
    ) -> Optional[TypesetResult]:
        """Attempt typesetting with fallback strategies.

        Fallbacks:
        1. Try with default font and simpler settings
        2. Skip typesetting (return None)
        """
        try:
            return typesetter.typeset_text(
                image, text, bbox,
                bubble_mask=bubble_mask,
                orientation=orientation,
                source_lang=source_lang,
            )
        except Exception as e:
            self.report.add("typesetting", str(e), "Trying basic typesetting", True)

        # Fallback: basic typesetter with minimal settings
        try:
            basic = Typesetter(
                default_font="DejaVuSans.ttf",
                min_font_size=8,
                max_font_size=36,
                outline_width=0,
            )
            result = basic.typeset_text(
                image, text, bbox,
                orientation="horizontal",
            )
            self.report.add("typesetting", "basic", "Basic typesetting succeeded", True)
            return result
        except Exception as e2:
            self.report.add("typesetting", str(e2), "All typesetting failed, skipping", False)
            return None
