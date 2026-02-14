"""Tests for semi-auto workflow mode."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from manga_translator.workflow import (
    SemiAutoWorkflow,
    WorkflowState,
    WorkflowStep,
    BubbleData,
)
from manga_translator.components.ocr_engine import OCRResult
from manga_translator.components.translator import TranslationResult


@pytest.fixture
def simple_image():
    return np.full((200, 300, 3), 200, dtype=np.uint8)


class TestWorkflowState:
    def test_initial_state(self, simple_image):
        state = WorkflowState(image=simple_image)
        assert state.current_step == WorkflowStep.DETECT
        assert state.bubbles == []
        assert state.progress_percent == 0.0

    def test_complete_progress(self, simple_image):
        state = WorkflowState(image=simple_image)
        state.current_step = WorkflowStep.COMPLETE
        assert state.progress_percent == 100.0

    def test_accepted_bubbles(self, simple_image):
        state = WorkflowState(image=simple_image)
        mock_bubble = MagicMock()
        state.bubbles = [
            BubbleData(bubble=mock_bubble, accepted=True),
            BubbleData(bubble=mock_bubble, accepted=False),
            BubbleData(bubble=mock_bubble, accepted=True),
        ]
        assert len(state.accepted_bubbles) == 2


class TestSemiAutoWorkflow:
    def test_start(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image, "ja", "en")
        assert state.current_step == WorkflowStep.DETECT
        assert state.source_lang == "ja"
        assert state.target_lang == "en"

    def test_advance(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.advance(state)
        assert state.current_step == WorkflowStep.REVIEW_BUBBLES

    def test_go_back(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.advance(state)  # -> REVIEW_BUBBLES
        state = workflow.go_back(state)
        assert state.current_step == WorkflowStep.DETECT

    def test_go_back_at_start(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.go_back(state)
        assert state.current_step == WorkflowStep.DETECT  # stays at start

    def test_execute_detect_step(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.execute_step(state)
        # After detect, should advance to REVIEW_BUBBLES
        assert state.current_step == WorkflowStep.REVIEW_BUBBLES
        assert isinstance(state.bubbles, list)

    def test_run_to_completion(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.run_to_completion(state)
        assert state.current_step == WorkflowStep.COMPLETE
        assert state.progress_percent == 100.0

    def test_update_bubble_text(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        mock_bubble = MagicMock()
        state.bubbles = [
            BubbleData(
                bubble=mock_bubble,
                ocr_result=OCRResult(
                    text="original", confidence=0.9,
                    language="ja", engine_used="test",
                ),
            )
        ]
        state = workflow.update_bubble_text(state, 0, "edited text")
        assert state.bubbles[0].ocr_result.text == "edited text"
        assert state.bubbles[0].ocr_edited is True
        assert state.bubbles[0].ocr_result.engine_used == "user_edit"

    def test_update_bubble_text_invalid_index(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        # Should not crash on invalid index
        state = workflow.update_bubble_text(state, 99, "text")

    def test_update_bubble_translation(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        mock_bubble = MagicMock()
        state.bubbles = [
            BubbleData(
                bubble=mock_bubble,
                translation=TranslationResult(
                    source_text="テスト",
                    translated_text="test",
                    source_language="ja",
                    target_language="en",
                    engine_used="openai",
                    confidence=0.9,
                ),
            )
        ]
        state = workflow.update_bubble_translation(state, 0, "corrected")
        assert state.bubbles[0].translation.translated_text == "corrected"
        assert state.bubbles[0].translation_edited is True
        assert state.bubbles[0].translation.confidence == 1.0

    def test_review_steps_are_noops(self, simple_image):
        """Review steps should just advance without executing anything."""
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        # Manually set to a review step
        state.current_step = WorkflowStep.REVIEW_BUBBLES
        state = workflow.execute_step(state)
        assert state.current_step == WorkflowStep.OCR

    def test_history_tracking(self, simple_image):
        workflow = SemiAutoWorkflow()
        state = workflow.start(simple_image)
        state = workflow.execute_step(state)  # detect -> review_bubbles
        state = workflow.execute_step(state)  # review_bubbles -> ocr
        assert WorkflowStep.DETECT in state.history
        assert WorkflowStep.REVIEW_BUBBLES in state.history


class TestWorkflowSteps:
    def test_step_order(self):
        """Steps should be in logical pipeline order."""
        from manga_translator.workflow import _STEP_ORDER
        assert _STEP_ORDER[0] == WorkflowStep.DETECT
        assert _STEP_ORDER[-1] == WorkflowStep.COMPLETE
        assert _STEP_ORDER.index(WorkflowStep.OCR) < _STEP_ORDER.index(WorkflowStep.TRANSLATE)
        assert _STEP_ORDER.index(WorkflowStep.TRANSLATE) < _STEP_ORDER.index(WorkflowStep.INPAINT)
        assert _STEP_ORDER.index(WorkflowStep.INPAINT) < _STEP_ORDER.index(WorkflowStep.TYPESET)
