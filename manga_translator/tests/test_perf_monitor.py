"""Tests for performance monitor."""

import time

from manga_translator.perf_monitor import (
    TimingRecord,
    PerfReport,
    PerfMonitor,
)


class TestTimingRecord:
    def test_fields(self):
        r = TimingRecord(name="test", duration=1.5)
        assert r.name == "test"
        assert r.duration == 1.5

    def test_metadata(self):
        r = TimingRecord(name="test", duration=1.0, metadata={"pages": "5"})
        assert r.metadata["pages"] == "5"


class TestPerfReport:
    def test_empty(self):
        r = PerfReport()
        assert r.stage_totals == {}
        assert r.stage_counts == {}

    def test_stage_totals(self):
        r = PerfReport(records=[
            TimingRecord(name="detect", duration=1.0),
            TimingRecord(name="detect", duration=2.0),
            TimingRecord(name="ocr", duration=0.5),
        ], total_time=3.5)
        assert r.stage_totals["detect"] == 3.0
        assert r.stage_totals["ocr"] == 0.5
        assert r.stage_counts["detect"] == 2

    def test_summary(self):
        r = PerfReport(
            records=[TimingRecord(name="detect", duration=1.0)],
            total_time=1.0,
        )
        s = r.summary()
        assert "detect" in s
        assert "1.00s" in s


class TestPerfMonitor:
    def test_track_context_manager(self):
        m = PerfMonitor()
        with m.track("test"):
            time.sleep(0.01)
        assert m.record_count == 1
        report = m.report()
        assert report.records[0].name == "test"
        assert report.records[0].duration >= 0.01

    def test_track_metadata(self):
        m = PerfMonitor()
        with m.track("detect", pages="5"):
            pass
        assert m.report().records[0].metadata["pages"] == "5"

    def test_timed_decorator(self):
        m = PerfMonitor()

        @m.timed("my_func")
        def do_work():
            return 42

        result = do_work()
        assert result == 42
        assert m.record_count == 1
        assert m.report().records[0].name == "my_func"

    def test_manual_record(self):
        m = PerfMonitor()
        m.record("external", 2.5, source="api")
        assert m.record_count == 1
        assert m.report().records[0].metadata["source"] == "api"

    def test_start_stop(self):
        m = PerfMonitor()
        m.start()
        time.sleep(0.01)
        elapsed = m.stop()
        assert elapsed >= 0.01

    def test_report_total_from_start(self):
        m = PerfMonitor()
        m.start()
        with m.track("a"):
            time.sleep(0.01)
        report = m.report()
        assert report.total_time >= 0.01

    def test_report_total_without_start(self):
        m = PerfMonitor()
        m.record("a", 1.0)
        m.record("b", 2.0)
        report = m.report()
        assert report.total_time == 3.0

    def test_reset(self):
        m = PerfMonitor()
        m.start()
        m.record("a", 1.0)
        m.reset()
        assert m.record_count == 0
        assert m.stop() == 0.0

    def test_multiple_stages(self):
        m = PerfMonitor()
        m.start()
        with m.track("detect"):
            time.sleep(0.01)
        with m.track("ocr"):
            time.sleep(0.01)
        with m.track("detect"):
            time.sleep(0.01)

        report = m.report()
        assert report.stage_counts["detect"] == 2
        assert report.stage_counts["ocr"] == 1
        assert report.total_time >= 0.03
