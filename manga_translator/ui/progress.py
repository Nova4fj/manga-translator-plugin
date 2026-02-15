"""Progress tracking for the translation pipeline."""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A step in the translation pipeline."""

    name: str
    weight: float = 1.0  # relative weight for progress calculation
    started: float = 0.0
    completed: float = 0.0
    status: str = "pending"  # pending, running, completed, failed


class ProgressTracker:
    """Tracks progress across the translation pipeline steps."""

    PIPELINE_STEPS = [
        PipelineStep("Bubble Detection", weight=2.0),
        PipelineStep("Text Extraction (OCR)", weight=2.0),
        PipelineStep("Translation", weight=1.5),
        PipelineStep("Text Removal (Inpainting)", weight=2.0),
        PipelineStep("Typesetting", weight=1.5),
        PipelineStep("Layer Assembly", weight=1.0),
    ]

    def __init__(self, callback: Optional[Callable[[int, int, str], None]] = None):
        self.steps = [
            PipelineStep(s.name, s.weight) for s in self.PIPELINE_STEPS
        ]
        self._callback = callback
        self._total_weight = sum(s.weight for s in self.steps)
        self._current_step_idx = -1

    def start_step(self, step_index: int) -> None:
        """Mark a step as started."""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            step.status = "running"
            step.started = time.time()
            self._current_step_idx = step_index
            logger.info("Pipeline step started: %s", step.name)
            self._notify()

    def complete_step(self, step_index: int) -> None:
        """Mark a step as completed."""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            step.status = "completed"
            step.completed = time.time()
            logger.info(
                "Pipeline step completed: %s (%.1fs)",
                step.name,
                step.completed - step.started if step.started else 0,
            )
            self._notify()

    def skip_step(self, step_index: int) -> None:
        """Mark a step as skipped (excluded from progress calculation)."""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            step.status = "skipped"
            self._total_weight -= step.weight
            self._notify()

    def fail_step(self, step_index: int, error: str = "") -> None:
        """Mark a step as failed."""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            step.status = "failed"
            step.completed = time.time()
            logger.error("Pipeline step failed: %s — %s", step.name, error)
            self._notify()

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress as 0.0 to 1.0."""
        completed_weight = sum(
            s.weight for s in self.steps if s.status == "completed"
        )
        return completed_weight / self._total_weight if self._total_weight > 0 else 0.0

    @property
    def current_step_name(self) -> str:
        """Get the name of the currently running step."""
        if 0 <= self._current_step_idx < len(self.steps):
            return self.steps[self._current_step_idx].name
        return ""

    def _notify(self) -> None:
        """Notify the callback of progress change."""
        if self._callback:
            completed = sum(1 for s in self.steps if s.status == "completed")
            total = len(self.steps)
            self._callback(completed, total, self.current_step_name)

    def get_summary(self) -> str:
        """Get a text summary of pipeline progress."""
        lines = []
        for i, step in enumerate(self.steps):
            icon = {"pending": "⬜", "running": "🔄", "completed": "✅", "failed": "❌", "skipped": "⏭️"}
            elapsed = ""
            if step.completed and step.started:
                elapsed = f" ({step.completed - step.started:.1f}s)"
            lines.append(f"  {icon.get(step.status, '?')} {step.name}{elapsed}")
        pct = int(self.overall_progress * 100)
        lines.insert(0, f"Progress: {pct}%")
        return "\n".join(lines)
