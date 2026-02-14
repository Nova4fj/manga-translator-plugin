"""Automated quality control for manga translations.

Checks translation output for common issues: untranslated text,
length mismatches, terminology consistency, and overall quality scoring.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class QCIssue:
    """A single quality control issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "untranslated", "length", "terminology", "consistency"
    message: str
    page: int = 0
    bubble: int = 0


@dataclass
class QCReport:
    """Aggregated quality control report."""
    issues: List[QCIssue] = field(default_factory=list)
    page_count: int = 0
    bubble_count: int = 0
    checked_count: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def score(self) -> float:
        """Quality score from 0.0 to 1.0."""
        if self.checked_count == 0:
            return 1.0
        penalty = self.error_count * 0.1 + self.warning_count * 0.03
        return max(0.0, 1.0 - penalty)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def summary(self) -> str:
        lines = [
            f"QC Report: {self.score:.0%} quality score",
            f"  Pages: {self.page_count}, Bubbles: {self.bubble_count}",
            f"  Issues: {self.error_count} errors, {self.warning_count} warnings",
        ]
        if self.issues:
            lines.append("  Details:")
            for issue in self.issues[:20]:
                prefix = {"error": "E", "warning": "W", "info": "I"}.get(issue.severity, "?")
                loc = f"p{issue.page}b{issue.bubble}" if issue.page else ""
                lines.append(f"    [{prefix}] {loc} {issue.category}: {issue.message}")
            if len(self.issues) > 20:
                lines.append(f"    ... and {len(self.issues) - 20} more")
        return "\n".join(lines)


# CJK Unicode ranges for detecting untranslated text
_CJK_PATTERN = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"
)


class QualityChecker:
    """Runs quality checks on translation results.

    Args:
        max_length_ratio: Maximum allowed target/source length ratio.
        min_length_ratio: Minimum allowed target/source length ratio.
        cjk_threshold: Maximum fraction of CJK chars in target (when target should be non-CJK).
    """

    def __init__(
        self,
        max_length_ratio: float = 4.0,
        min_length_ratio: float = 0.2,
        cjk_threshold: float = 0.3,
    ):
        self.max_length_ratio = max_length_ratio
        self.min_length_ratio = min_length_ratio
        self.cjk_threshold = cjk_threshold

    def check_translation(
        self,
        source: str,
        target: str,
        target_lang: str = "en",
        page: int = 0,
        bubble: int = 0,
        terms: Optional[Dict[str, str]] = None,
    ) -> List[QCIssue]:
        """Run all checks on a single source/target pair."""
        issues: List[QCIssue] = []

        if not source.strip() or not target.strip():
            return issues

        issues.extend(self._check_untranslated(target, target_lang, page, bubble))
        issues.extend(self._check_length_ratio(source, target, page, bubble))
        if terms:
            issues.extend(self._check_terminology(target, terms, page, bubble))

        return issues

    def check_page(
        self,
        pairs: List[Dict[str, str]],
        target_lang: str = "en",
        page_num: int = 0,
        terms: Optional[Dict[str, str]] = None,
    ) -> QCReport:
        """Check all translations on a page.

        Args:
            pairs: List of dicts with "source" and "target" keys.
            target_lang: Expected target language code.
            page_num: Page number for issue tracking.
            terms: Terminology dict (source term -> expected translation).

        Returns:
            QCReport with all issues found.
        """
        report = QCReport(page_count=1, bubble_count=len(pairs))

        for i, pair in enumerate(pairs):
            source = pair.get("source", "")
            target = pair.get("target", "")
            report.checked_count += 1
            issues = self.check_translation(
                source, target, target_lang,
                page=page_num, bubble=i + 1, terms=terms,
            )
            report.issues.extend(issues)

        return report

    def check_batch(
        self,
        pages: List[List[Dict[str, str]]],
        target_lang: str = "en",
        terms: Optional[Dict[str, str]] = None,
    ) -> QCReport:
        """Check translations across multiple pages."""
        combined = QCReport()

        for page_num, pairs in enumerate(pages):
            page_report = self.check_page(
                pairs, target_lang, page_num=page_num + 1, terms=terms,
            )
            combined.issues.extend(page_report.issues)
            combined.page_count += 1
            combined.bubble_count += page_report.bubble_count
            combined.checked_count += page_report.checked_count

        return combined

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_untranslated(
        self, target: str, target_lang: str, page: int, bubble: int,
    ) -> List[QCIssue]:
        """Detect untranslated CJK text in target when target should be non-CJK."""
        issues = []
        if target_lang in ("ja", "zh", "ko"):
            return issues  # CJK target is expected

        cjk_chars = len(_CJK_PATTERN.findall(target))
        total_chars = len(target.replace(" ", ""))
        if total_chars == 0:
            return issues

        ratio = cjk_chars / total_chars
        if ratio > self.cjk_threshold:
            if ratio > 0.8:
                issues.append(QCIssue(
                    severity="error", category="untranslated",
                    message=f"Text appears untranslated ({ratio:.0%} CJK)",
                    page=page, bubble=bubble,
                ))
            else:
                issues.append(QCIssue(
                    severity="warning", category="untranslated",
                    message=f"Possible partial translation ({ratio:.0%} CJK)",
                    page=page, bubble=bubble,
                ))
        return issues

    def _check_length_ratio(
        self, source: str, target: str, page: int, bubble: int,
    ) -> List[QCIssue]:
        """Check if target length is reasonable relative to source."""
        issues = []
        src_len = len(source.strip())
        tgt_len = len(target.strip())
        if src_len == 0:
            return issues

        ratio = tgt_len / src_len
        if ratio > self.max_length_ratio:
            issues.append(QCIssue(
                severity="warning", category="length",
                message=f"Translation much longer than source (ratio {ratio:.1f}x)",
                page=page, bubble=bubble,
            ))
        elif ratio < self.min_length_ratio:
            issues.append(QCIssue(
                severity="warning", category="length",
                message=f"Translation much shorter than source (ratio {ratio:.1f}x)",
                page=page, bubble=bubble,
            ))
        return issues

    def _check_terminology(
        self, target: str, terms: Dict[str, str], page: int, bubble: int,
    ) -> List[QCIssue]:
        """Check if expected terminology translations appear in target."""
        issues = []
        target_lower = target.lower()
        for source_term, expected in terms.items():
            if source_term.lower() in target_lower and expected.lower() not in target_lower:
                issues.append(QCIssue(
                    severity="warning", category="terminology",
                    message=f"Term '{source_term}' found but expected translation '{expected}' missing",
                    page=page, bubble=bubble,
                ))
        return issues
