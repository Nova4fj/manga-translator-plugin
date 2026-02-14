"""Tests for quality control system."""

import pytest

from manga_translator.quality_control import (
    QCIssue,
    QCReport,
    QualityChecker,
)


class TestQCReport:
    def test_empty(self):
        r = QCReport()
        assert r.error_count == 0
        assert r.warning_count == 0
        assert r.score == 1.0
        assert r.passed is True

    def test_with_errors(self):
        r = QCReport(issues=[
            QCIssue(severity="error", category="untranslated", message="x"),
            QCIssue(severity="warning", category="length", message="y"),
        ], checked_count=5)
        assert r.error_count == 1
        assert r.warning_count == 1
        assert r.passed is False
        assert r.score < 1.0

    def test_score_floor(self):
        issues = [QCIssue(severity="error", category="x", message="m") for _ in range(20)]
        r = QCReport(issues=issues, checked_count=1)
        assert r.score == 0.0

    def test_summary(self):
        r = QCReport(
            issues=[QCIssue(severity="error", category="untranslated", message="test", page=1, bubble=2)],
            page_count=1, bubble_count=3, checked_count=3,
        )
        s = r.summary()
        assert "p1b2" in s
        assert "untranslated" in s


class TestUntranslatedDetection:
    def test_clean_english(self):
        qc = QualityChecker()
        issues = qc.check_translation("テスト", "Test", target_lang="en")
        untranslated = [i for i in issues if i.category == "untranslated"]
        assert len(untranslated) == 0

    def test_fully_untranslated(self):
        qc = QualityChecker()
        issues = qc.check_translation("テスト", "テスト", target_lang="en")
        errors = [i for i in issues if i.category == "untranslated" and i.severity == "error"]
        assert len(errors) == 1

    def test_partial_translation(self):
        qc = QualityChecker()
        # ~50% CJK chars (4 CJK / 8 total non-space) → warning range (0.3-0.8)
        issues = qc.check_translation("テスト", "Testテストab", target_lang="en")
        untranslated = [i for i in issues if i.category == "untranslated"]
        assert len(untranslated) >= 1

    def test_cjk_target_lang_ok(self):
        qc = QualityChecker()
        issues = qc.check_translation("Hello", "こんにちは", target_lang="ja")
        untranslated = [i for i in issues if i.category == "untranslated"]
        assert len(untranslated) == 0

    def test_empty_text(self):
        qc = QualityChecker()
        issues = qc.check_translation("", "", target_lang="en")
        assert len(issues) == 0


class TestLengthRatio:
    def test_normal_ratio(self):
        qc = QualityChecker()
        issues = qc.check_translation("テスト", "Test", target_lang="en")
        length_issues = [i for i in issues if i.category == "length"]
        assert len(length_issues) == 0

    def test_too_long(self):
        qc = QualityChecker(max_length_ratio=3.0)
        issues = qc.check_translation("短い", "This is a very very very long translation text", target_lang="en")
        length_issues = [i for i in issues if i.category == "length"]
        assert len(length_issues) == 1

    def test_too_short(self):
        qc = QualityChecker(min_length_ratio=0.5)
        issues = qc.check_translation("This is a long source text in English", "X", target_lang="en")
        length_issues = [i for i in issues if i.category == "length"]
        assert len(length_issues) == 1


class TestTerminology:
    def test_correct_term(self):
        qc = QualityChecker()
        terms = {"naruto": "Naruto"}
        issues = qc.check_translation(
            "ナルト", "Naruto is here", target_lang="en", terms=terms,
        )
        term_issues = [i for i in issues if i.category == "terminology"]
        assert len(term_issues) == 0

    def test_missing_term(self):
        qc = QualityChecker()
        # Source term "naruto" found in target, but expected "Naruto-kun" is missing
        terms = {"naruto": "Naruto-kun"}
        issues = qc.check_translation(
            "ナルト", "The naruto character is here", target_lang="en", terms=terms,
        )
        term_issues = [i for i in issues if i.category == "terminology"]
        assert len(term_issues) == 1


class TestCheckPage:
    def test_check_page(self):
        qc = QualityChecker()
        pairs = [
            {"source": "テスト", "target": "Test"},
            {"source": "漫画", "target": "漫画"},  # untranslated
        ]
        report = qc.check_page(pairs, target_lang="en", page_num=1)
        assert report.bubble_count == 2
        assert report.checked_count == 2
        assert report.error_count >= 1

    def test_check_batch(self):
        qc = QualityChecker()
        pages = [
            [{"source": "a", "target": "A"}],
            [{"source": "b", "target": "B"}],
        ]
        report = qc.check_batch(pages, target_lang="en")
        assert report.page_count == 2
        assert report.bubble_count == 2
        assert report.passed is True


class TestCustomThresholds:
    def test_strict_cjk_threshold(self):
        qc = QualityChecker(cjk_threshold=0.1)
        # Even a small amount of CJK triggers warning
        issues = qc.check_translation("テスト", "Test テ", target_lang="en")
        untranslated = [i for i in issues if i.category == "untranslated"]
        assert len(untranslated) >= 1
