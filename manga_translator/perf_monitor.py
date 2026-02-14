"""Lightweight performance monitoring for the translation pipeline.

Provides timing instrumentation via context managers and decorators,
plus memory-usage snapshots and summary reporting.
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """A single timing measurement."""
    name: str
    duration: float  # seconds
    timestamp: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerfReport:
    """Aggregated performance report."""
    records: List[TimingRecord] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def stage_totals(self) -> Dict[str, float]:
        """Sum durations by stage name."""
        totals: Dict[str, float] = {}
        for r in self.records:
            totals[r.name] = totals.get(r.name, 0.0) + r.duration
        return totals

    @property
    def stage_counts(self) -> Dict[str, int]:
        """Count invocations by stage name."""
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.name] = counts.get(r.name, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [f"Performance Report (total: {self.total_time:.2f}s)"]
        lines.append(f"{'Stage':<25} {'Count':>6} {'Total':>8} {'Avg':>8} {'%':>6}")
        lines.append("-" * 57)

        totals = self.stage_totals
        counts = self.stage_counts

        for name in sorted(totals, key=totals.get, reverse=True):
            total = totals[name]
            count = counts[name]
            avg = total / count if count else 0
            pct = (total / self.total_time * 100) if self.total_time > 0 else 0
            lines.append(
                f"{name:<25} {count:>6} {total:>7.2f}s {avg:>7.3f}s {pct:>5.1f}%"
            )

        return "\n".join(lines)


class PerfMonitor:
    """Collects timing data across pipeline stages.

    Usage::

        monitor = PerfMonitor()
        with monitor.track("detection"):
            detect_bubbles(image)
        with monitor.track("ocr"):
            run_ocr(regions)
        print(monitor.report().summary())

    Or as a decorator::

        @monitor.timed("detection")
        def detect_bubbles(image):
            ...
    """

    def __init__(self):
        self._records: List[TimingRecord] = []
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Mark the start of the overall pipeline."""
        self._start_time = time.time()

    def stop(self) -> float:
        """Mark the end and return total elapsed time."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return elapsed

    @contextmanager
    def track(self, name: str, **metadata):
        """Context manager to time a code block.

        Args:
            name: Stage name (e.g. "detection", "ocr", "inpainting").
            **metadata: Extra key-value pairs to attach to the record.
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._records.append(TimingRecord(
                name=name, duration=duration,
                timestamp=start,
                metadata={k: str(v) for k, v in metadata.items()},
            ))
            logger.debug("%s completed in %.3fs", name, duration)

    def timed(self, name: str) -> Callable:
        """Decorator to time a function."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.track(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record(self, name: str, duration: float, **metadata) -> None:
        """Manually add a timing record."""
        self._records.append(TimingRecord(
            name=name, duration=duration,
            timestamp=time.time(),
            metadata={k: str(v) for k, v in metadata.items()},
        ))

    def report(self) -> PerfReport:
        """Generate a performance report from collected records."""
        total = self.stop() if self._start_time else sum(r.duration for r in self._records)
        return PerfReport(records=list(self._records), total_time=total)

    def reset(self) -> None:
        """Clear all collected records."""
        self._records.clear()
        self._start_time = None

    @property
    def record_count(self) -> int:
        return len(self._records)
