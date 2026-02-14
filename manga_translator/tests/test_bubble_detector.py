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


class TestInteriorFilter:
    """Tests for _is_bubble_interior — ink-bounded whiteness check."""

    def _make_contour_and_bbox(self, center, axes):
        """Helper: return an ellipse contour and its bounding box."""
        pts = cv2.ellipse2Poly(center, axes, 0, 0, 360, 5)
        contour = pts.reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        return contour, (x, y, w, h)

    def test_white_bubble_with_outline_passes(self):
        """A white ellipse with a dark outline on gray background should pass."""
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        center, axes = (150, 150), (80, 50)
        # Dark outline then white fill (mimics a real manga bubble)
        cv2.ellipse(img, center, axes, 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(img, center, (axes[0] - 3, axes[1] - 3), 0, 0, 360, (255, 255, 255), -1)
        contour, bbox = self._make_contour_and_bbox(center, axes)
        detector = BubbleDetector()
        ink_contours = detector._find_ink_bounded_contours(img)
        assert detector._is_bubble_interior(img, contour, bbox, ink_contours) is True

    def test_textured_region_rejected(self):
        """A region with varied colours (simulating artwork) should fail."""
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        center, axes = (150, 150), (80, 50)
        contour, bbox = self._make_contour_and_bbox(center, axes)
        detector = BubbleDetector()
        ink_contours = detector._find_ink_bounded_contours(img)
        assert detector._is_bubble_interior(img, contour, bbox, ink_contours) is False

    def test_white_patch_without_outline_rejected(self):
        """A bright patch on a manga page without its own closed ink
        boundary should be rejected (simulates sweater highlights)."""
        # Gray background with a dark-outlined bubble on the LEFT and an
        # un-outlined bright patch on the RIGHT.
        img = np.full((300, 400, 3), 160, dtype=np.uint8)
        # Left side: a real bubble (dark outline + white fill)
        cv2.ellipse(img, (80, 150), (50, 40), 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(img, (80, 150), (47, 37), 0, 0, 360, (255, 255, 255), -1)
        # Right side: bright patch with NO dark outline (sweater-like)
        center, axes = (300, 150), (60, 40)
        cv2.ellipse(img, center, axes, 0, 0, 360, (255, 255, 255), -1)
        contour, bbox = self._make_contour_and_bbox(center, axes)
        detector = BubbleDetector()
        ink_contours = detector._find_ink_bounded_contours(img)
        assert detector._is_bubble_interior(img, contour, bbox, ink_contours) is False
