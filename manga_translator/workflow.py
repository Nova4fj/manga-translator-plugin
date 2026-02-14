"""Semi-auto and manual workflow modes for manga translation.

Provides step-by-step pipeline execution with review checkpoints,
allowing users to review, adjust, and approve at each stage.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Callable, Any

import numpy as np

from manga_translator.components.bubble_detector import BubbleDetector, BubbleRegion
from manga_translator.components.ocr_engine import OCREngine, OCRResult
from manga_translator.components.translator import TranslationManager, TranslationResult
from manga_translator.components.inpainter import Inpainter, InpaintResult
from manga_translator.components.typesetter import Typesetter, TypesetResult
from manga_translator.config.settings import PluginSettings, SettingsManager
from manga_translator.core.image_processor import resize_for_processing, scale_bbox

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Pipeline steps in order."""
    DETECT = "detect"
    REVIEW_BUBBLES = "review_bubbles"
    OCR = "ocr"
    REVIEW_TEXT = "review_text"
    TRANSLATE = "translate"
    REVIEW_TRANSLATION = "review_translation"
    INPAINT = "inpaint"
    TYPESET = "typeset"
    COMPLETE = "complete"


# Step order for iteration
_STEP_ORDER = list(WorkflowStep)


@dataclass
class BubbleData:
    """Mutable data for a single bubble through the workflow."""
    bubble: BubbleRegion
    accepted: bool = True  # user can reject individual bubbles
    ocr_result: Optional[OCRResult] = None
    ocr_edited: bool = False  # whether user edited the OCR text
    translation: Optional[TranslationResult] = None
    translation_edited: bool = False
    inpaint_result: Optional[InpaintResult] = None
    typeset_result: Optional[TypesetResult] = None


@dataclass
class WorkflowState:
    """Complete state of the workflow at any point."""
    image: np.ndarray
    source_lang: str = "ja"
    target_lang: str = "en"
    current_step: WorkflowStep = WorkflowStep.DETECT
    bubbles: List[BubbleData] = field(default_factory=list)
    cleaned_image: Optional[np.ndarray] = None
    final_image: Optional[np.ndarray] = None
    errors: List[str] = field(default_factory=list)
    history: List[WorkflowStep] = field(default_factory=list)

    @property
    def accepted_bubbles(self) -> List[BubbleData]:
        return [b for b in self.bubbles if b.accepted]

    @property
    def progress_percent(self) -> float:
        if self.current_step == WorkflowStep.COMPLETE:
            return 100.0
        idx = _STEP_ORDER.index(self.current_step)
        return (idx / (len(_STEP_ORDER) - 1)) * 100.0


class SemiAutoWorkflow:
    """Step-by-step translation workflow with review checkpoints.

    Usage::

        workflow = SemiAutoWorkflow(settings)
        state = workflow.start(image, "ja", "en")

        # Run detection
        state = workflow.execute_step(state)
        # state.current_step is now REVIEW_BUBBLES

        # User reviews and adjusts bubbles...
        state.bubbles[0].accepted = False  # reject a bubble

        # Advance to OCR
        state = workflow.advance(state)
        state = workflow.execute_step(state)
        # ... and so on

        # Or run to completion
        state = workflow.run_to_completion(state)
    """

    # Steps that pause for user review (not auto-executed)
    REVIEW_STEPS = {
        WorkflowStep.REVIEW_BUBBLES,
        WorkflowStep.REVIEW_TEXT,
        WorkflowStep.REVIEW_TRANSLATION,
    }

    def __init__(self, settings: Optional[PluginSettings] = None):
        if settings is None:
            settings = SettingsManager().get_settings()
        self.settings = settings

        self.detector = BubbleDetector(
            min_area=settings.detection.min_bubble_area,
            max_area=settings.detection.max_bubble_area,
        )
        self.ocr = OCREngine(
            primary_engine=settings.ocr.primary_engine,
            confidence_threshold=settings.ocr.confidence_threshold,
        )
        self.translator = TranslationManager(
            primary_engine=settings.translation.primary_engine,
            openai_api_key=settings.translation.openai_api_key,
            deepl_api_key=settings.translation.deepl_api_key,
        )
        self.inpainter = Inpainter(
            method=settings.inpainting.method,
            lama_model_dir=settings.model_cache_dir or None,
            lama_device="auto" if settings.gpu_enabled else "cpu",
        )
        self.typesetter = Typesetter(
            default_font=settings.typesetting.default_font,
            alignment=settings.typesetting.alignment,
        )

    def start(
        self, image: np.ndarray, source_lang: str = "ja", target_lang: str = "en"
    ) -> WorkflowState:
        """Initialize a new workflow."""
        return WorkflowState(
            image=image.copy(),
            source_lang=source_lang,
            target_lang=target_lang,
            current_step=WorkflowStep.DETECT,
        )

    def execute_step(self, state: WorkflowState) -> WorkflowState:
        """Execute the current step and advance to the next."""
        step = state.current_step

        if step == WorkflowStep.DETECT:
            state = self._step_detect(state)
        elif step == WorkflowStep.OCR:
            state = self._step_ocr(state)
        elif step == WorkflowStep.TRANSLATE:
            state = self._step_translate(state)
        elif step == WorkflowStep.INPAINT:
            state = self._step_inpaint(state)
        elif step == WorkflowStep.TYPESET:
            state = self._step_typeset(state)
        elif step in self.REVIEW_STEPS:
            # Review steps are no-ops — user reviews state externally
            pass
        elif step == WorkflowStep.COMPLETE:
            pass

        state.history.append(step)
        state = self.advance(state)
        return state

    def advance(self, state: WorkflowState) -> WorkflowState:
        """Move to the next step in the pipeline."""
        idx = _STEP_ORDER.index(state.current_step)
        if idx < len(_STEP_ORDER) - 1:
            state.current_step = _STEP_ORDER[idx + 1]
        return state

    def go_back(self, state: WorkflowState) -> WorkflowState:
        """Move back to the previous step."""
        idx = _STEP_ORDER.index(state.current_step)
        if idx > 0:
            state.current_step = _STEP_ORDER[idx - 1]
        return state

    def run_to_completion(self, state: WorkflowState) -> WorkflowState:
        """Run all remaining steps, skipping review checkpoints."""
        while state.current_step != WorkflowStep.COMPLETE:
            if state.current_step in self.REVIEW_STEPS:
                # Auto-advance past review steps
                state = self.advance(state)
            else:
                state = self.execute_step(state)
        return state

    def update_bubble_text(
        self, state: WorkflowState, bubble_idx: int, new_text: str
    ) -> WorkflowState:
        """User edits OCR text for a specific bubble."""
        if 0 <= bubble_idx < len(state.bubbles):
            bd = state.bubbles[bubble_idx]
            if bd.ocr_result is not None:
                bd.ocr_result = OCRResult(
                    text=new_text,
                    confidence=bd.ocr_result.confidence,
                    language=bd.ocr_result.language,
                    engine_used="user_edit",
                )
                bd.ocr_edited = True
        return state

    def update_bubble_translation(
        self, state: WorkflowState, bubble_idx: int, new_text: str
    ) -> WorkflowState:
        """User edits translation for a specific bubble."""
        if 0 <= bubble_idx < len(state.bubbles):
            bd = state.bubbles[bubble_idx]
            if bd.translation is not None:
                bd.translation = TranslationResult(
                    source_text=bd.translation.source_text,
                    translated_text=new_text,
                    source_language=bd.translation.source_language,
                    target_language=bd.translation.target_language,
                    engine_used="user_edit",
                    confidence=1.0,
                )
                bd.translation_edited = True
        return state

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_detect(self, state: WorkflowState) -> WorkflowState:
        """Run bubble detection."""
        try:
            processed, scale = resize_for_processing(state.image, max_dimension=4096)
            bubbles = self.detector.detect_bubbles(processed)

            if scale != 1.0:
                for b in bubbles:
                    b.bbox = scale_bbox(b.bbox, scale)

            state.bubbles = [BubbleData(bubble=b) for b in bubbles]
            logger.info("Detected %d bubbles", len(bubbles))
        except Exception as e:
            state.errors.append(f"Detection failed: {e}")
            logger.error("Detection failed: %s", e)
        return state

    def _step_ocr(self, state: WorkflowState) -> WorkflowState:
        """Run OCR on accepted bubbles."""
        for bd in state.accepted_bubbles:
            try:
                x, y, w, h = bd.bubble.bbox
                region = state.image[y:y+h, x:x+w]
                if region.size == 0:
                    bd.ocr_result = OCRResult(
                        text="", confidence=0.0,
                        language=state.source_lang, engine_used="none",
                    )
                    continue
                bd.ocr_result = self.ocr.extract_text(
                    region, language_hint=state.source_lang,
                )
            except Exception as e:
                state.errors.append(f"OCR failed for bubble {bd.bubble.id}: {e}")
                bd.ocr_result = OCRResult(
                    text="", confidence=0.0,
                    language=state.source_lang, engine_used="error",
                )
        return state

    def _step_translate(self, state: WorkflowState) -> WorkflowState:
        """Translate OCR'd text."""
        texts_to_translate = []
        bubble_indices = []

        for i, bd in enumerate(state.accepted_bubbles):
            if bd.ocr_result and bd.ocr_result.text.strip():
                texts_to_translate.append(bd.ocr_result.text)
                bubble_indices.append(i)

        if not texts_to_translate:
            return state

        try:
            results = self.translator.translate_batch(
                texts_to_translate, state.source_lang, state.target_lang,
            )
            accepted = state.accepted_bubbles
            for idx, result in zip(bubble_indices, results):
                accepted[idx].translation = result
        except Exception as e:
            state.errors.append(f"Translation failed: {e}")
        return state

    def _step_inpaint(self, state: WorkflowState) -> WorkflowState:
        """Remove text from accepted bubbles."""
        cleaned = state.image.copy()

        for bd in state.accepted_bubbles:
            if not (bd.ocr_result and bd.ocr_result.text.strip()):
                continue
            try:
                x, y, w, h = bd.bubble.bbox
                region = cleaned[y:y+h, x:x+w]
                if region.size == 0:
                    continue

                local_mask = bd.bubble.mask
                if local_mask is not None:
                    local_mask = local_mask[y:y+h, x:x+w]
                else:
                    local_mask = np.ones((h, w), dtype=np.uint8) * 255

                text_mask = self.inpainter.create_text_mask(region, local_mask)
                result = self.inpainter.remove_text_with_fallback(
                    region, text_mask,
                    quality_threshold=self.settings.inpainting.quality_threshold,
                )
                cleaned[y:y+h, x:x+w] = result.image
                bd.inpaint_result = result
            except Exception as e:
                state.errors.append(f"Inpaint failed for bubble {bd.bubble.id}: {e}")

        state.cleaned_image = cleaned
        return state

    def _step_typeset(self, state: WorkflowState) -> WorkflowState:
        """Render translated text onto cleaned image."""
        if state.cleaned_image is None:
            state.cleaned_image = state.image.copy()

        final = state.cleaned_image.copy()

        for bd in state.accepted_bubbles:
            if not (bd.translation and bd.translation.translated_text):
                continue
            try:
                result = self.typesetter.typeset_text(
                    final,
                    bd.translation.translated_text,
                    bd.bubble.bbox,
                    bubble_mask=bd.bubble.mask,
                    orientation=self.settings.typesetting.orientation,
                    source_lang=state.source_lang,
                )
                final = result.image
                bd.typeset_result = result
            except Exception as e:
                state.errors.append(f"Typeset failed for bubble {bd.bubble.id}: {e}")

        state.final_image = final
        state.current_step = WorkflowStep.COMPLETE
        return state
