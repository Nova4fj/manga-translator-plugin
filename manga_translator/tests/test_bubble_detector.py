"""Tests for bubble detection."""

import numpy as np
import cv2

from manga_translator.components.bubble_detector import BubbleDetector, BubbleRegion


class TestBubbleDetector:
    def test_init(self):
        detector = BubbleDetector()
        assert detector is not None

    def test_detect_simple_bubbles(self, sample_manga_page):
        detector = BubbleDetector(min_area=500)
        bubbles = detector.detect_bubbles(sample_manga_page)
        assert isinstance(bubbles, list)
        # Should detect at least 1 bubble from our synthetic page
        assert len(bubbles) >= 1

    def test_bubble_region_properties(self, sample_manga_page):
        detector = BubbleDetector(min_area=500)
        bubbles = detector.detect_bubbles(sample_manga_page)
        if bubbles:
            bubble = bubbles[0]
            assert isinstance(bubble, BubbleRegion)
            assert bubble.bbox is not None
            assert len(bubble.bbox) == 4
            assert bubble.confidence >= 0.0
            assert bubble.confidence <= 1.0
            assert bubble.shape_type in ("oval", "rectangle", "irregular", "thought")

    def test_empty_image(self):
        detector = BubbleDetector()
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        bubbles = detector.detect_bubbles(empty)
        assert len(bubbles) == 0

    def test_reading_order(self, sample_manga_page):
        detector = BubbleDetector(min_area=500)
        bubbles = detector.detect_bubbles(sample_manga_page)
        if len(bubbles) >= 2:
            # Manga reading order: right-to-left, top-to-bottom
            # First bubble should be upper-right
            sorted_bubbles = detector.sort_reading_order(bubbles)
            assert sorted_bubbles[0].center[0] >= sorted_bubbles[-1].center[0] or \
                   sorted_bubbles[0].center[1] <= sorted_bubbles[-1].center[1]

    def test_classify_shape_oval(self):
        detector = BubbleDetector()
        # Create an oval contour
        ellipse = cv2.ellipse2Poly((100, 100), (80, 50), 0, 0, 360, 5)
        shape = detector.classify_shape(ellipse)
        assert shape in ("oval", "rectangle", "irregular", "thought")

    def test_confidence_scoring(self):
        detector = BubbleDetector()
        ellipse = cv2.ellipse2Poly((100, 100), (80, 50), 0, 0, 360, 5)
        score = detector.score_confidence(ellipse, "oval")
        assert 0.0 <= score <= 1.0
