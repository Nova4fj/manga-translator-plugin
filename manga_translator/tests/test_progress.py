"""Tests for progress tracking."""

from manga_translator.ui.progress import ProgressTracker


class TestProgressTracker:
    def test_init(self):
        tracker = ProgressTracker()
        assert tracker is not None

    def test_with_callback(self):
        steps = []

        def cb(step, total, msg):
            steps.append((step, total, msg))

        tracker = ProgressTracker(callback=cb)
        tracker.start_step(0)
        tracker.complete_step(0)
        assert len(steps) > 0

    def test_all_steps(self):
        steps = []

        def cb(step, total, msg):
            steps.append((step, total, msg))

        tracker = ProgressTracker(callback=cb)
        for i in range(6):
            tracker.start_step(i)
            tracker.complete_step(i)
        assert len(steps) >= 6

    def test_summary(self):
        tracker = ProgressTracker()
        for i in range(6):
            tracker.start_step(i)
            tracker.complete_step(i)
        summary = tracker.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_no_callback_works(self):
        tracker = ProgressTracker(callback=None)
        tracker.start_step(0)
        tracker.complete_step(0)
        # No crash
