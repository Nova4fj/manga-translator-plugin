"""Tests for SFX / onomatopoeia detector."""

import numpy as np
import pytest
import cv2

from manga_translator.components.sfx_detector import SFXDetector, SFXRegion, SFXType


def _make_blank_page(h=600, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_sfx_stroke(image, x, y, w, h, thickness=5):
    """Draw a bold stroke pattern resembling SFX text."""
    for i in range(0, w, 10):
        cv2.line(image, (x + i, y), (x + i + 5, y + h), (0, 0, 0), thickness)
    for j in range(0, h, 12):
        cv2.line(image, (x, y + j), (x + w, y + j + 3), (0, 0, 0), thickness)
    return image


class _FakeBubble:
    def __init__(self, bbox):
        self.bbox = bbox


class TestSFXDetector:
    def test_empty_image(self):
        detector = SFXDetector()
        assert detector.detect_sfx(np.array([]), []) == []

    def test_none_image(self):
        detector = SFXDetector()
        assert detector.detect_sfx(None, []) == []

    def test_blank_page_no_sfx(self):
        detector = SFXDetector()
        assert detector.detect_sfx(_make_blank_page()) == []

    def test_detect_sfx_stroke(self):
        detector = SFXDetector(min_area=200)
        page = _make_blank_page()
        _draw_sfx_stroke(page, 50, 50, 80, 100, thickness=6)
        result = detector.detect_sfx(page)
        assert len(result) >= 1
        assert all(isinstance(r, SFXRegion) for r in result)
        found = any(30 <= r.bbox[0] <= 150 and 30 <= r.bbox[1] <= 200 for r in result)
        assert found, f"Expected SFX near (50,50), got {[r.bbox for r in result]}"

    def test_sfx_excluded_by_bubble(self):
        detector = SFXDetector(min_area=200)
        page = _make_blank_page()
        _draw_sfx_stroke(page, 50, 50, 80, 100, thickness=6)
        bubble = _FakeBubble(bbox=(0, 0, 300, 300))
        result = detector.detect_sfx(page, [bubble])
        in_bubble = [r for r in result if r.bbox[0] <= 300 and r.bbox[1] <= 300]
        assert len(in_bubble) == 0

    def test_sfx_type_classification(self):
        detector = SFXDetector(min_area=100)
        page = _make_blank_page(800, 800)
        _draw_sfx_stroke(page, 50, 50, 200, 30, thickness=4)
        result = detector.detect_sfx(page)
        assert len(result) >= 1

    def test_grayscale_input(self):
        detector = SFXDetector(min_area=200)
        page = np.full((600, 400), 255, dtype=np.uint8)
        for i in range(0, 80, 10):
            cv2.line(page, (50 + i, 50), (55 + i, 150), 0, 6)
        for j in range(0, 100, 12):
            cv2.line(page, (50, 50 + j), (130, 53 + j), 0, 6)
        result = detector.detect_sfx(page)
        assert len(result) >= 1

    def test_sfx_region_fields(self):
        detector = SFXDetector(min_area=200)
        page = _make_blank_page()
        _draw_sfx_stroke(page, 100, 100, 80, 100, thickness=6)
        result = detector.detect_sfx(page)
        if result:
            r = result[0]
            assert r.id >= 0
            assert len(r.bbox) == 4
            assert r.area > 0
            assert 0.0 <= r.confidence <= 1.0
            assert isinstance(r.sfx_type, SFXType)
            assert r.center == (r.bbox[0] + r.bbox[2] // 2, r.bbox[1] + r.bbox[3] // 2)

    def test_multiple_sfx_regions(self):
        detector = SFXDetector(min_area=200)
        page = _make_blank_page(800, 800)
        _draw_sfx_stroke(page, 50, 50, 80, 100, thickness=6)
        _draw_sfx_stroke(page, 500, 500, 80, 100, thickness=6)
        result = detector.detect_sfx(page)
        assert len(result) >= 2
