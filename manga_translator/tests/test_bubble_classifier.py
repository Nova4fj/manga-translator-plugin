"""Tests for bubble shape classifier."""

import numpy as np

from manga_translator.components.bubble_classifier import (
    BubbleClassifier,
    BubbleType,
    ClassificationResult,
)


def _make_circle_contour(cx=200, cy=200, radius=80, n_points=64):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.array([
        [int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))]
        for a in angles
    ], dtype=np.int32).reshape(-1, 1, 2)
    return pts


def _make_rectangle_contour(x=100, y=100, w=200, h=80):
    pts = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
    ], dtype=np.int32).reshape(-1, 1, 2)
    return pts


def _make_cloud_contour(cx=200, cy=200, radius=80, bumps=12, bump_size=15):
    angles = np.linspace(0, 2 * np.pi, bumps * 4, endpoint=False)
    pts = []
    for i, a in enumerate(angles):
        r = radius + bump_size * np.sin(i * np.pi / 2)
        pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _make_spiky_contour(cx=200, cy=200, radius=80, spikes=8, spike_len=40):
    pts = []
    for i in range(spikes * 2):
        angle = 2 * np.pi * i / (spikes * 2)
        r = radius + spike_len if i % 2 == 0 else radius
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


class TestBubbleClassifier:
    def test_classify_circle_as_speech(self):
        clf = BubbleClassifier()
        contour = _make_circle_contour()
        result = clf.classify(contour)
        assert isinstance(result, ClassificationResult)
        assert result.bubble_type == BubbleType.SPEECH
        assert result.confidence > 0.3

    def test_classify_rectangle_as_narration(self):
        clf = BubbleClassifier()
        contour = _make_rectangle_contour()
        result = clf.classify(contour)
        assert result.bubble_type == BubbleType.NARRATION
        assert result.confidence > 0.3

    def test_classify_cloud_as_thought(self):
        clf = BubbleClassifier()
        contour = _make_cloud_contour(bumps=16, bump_size=20)
        result = clf.classify(contour)
        assert result.bubble_type in (BubbleType.THOUGHT, BubbleType.SHOUT)

    def test_classify_spiky_as_shout(self):
        clf = BubbleClassifier()
        contour = _make_spiky_contour(spikes=10, spike_len=50)
        result = clf.classify(contour)
        assert result.bubble_type == BubbleType.SHOUT
        assert result.confidence > 0.3

    def test_features_populated(self):
        clf = BubbleClassifier()
        contour = _make_circle_contour()
        result = clf.classify(contour)
        assert "circularity" in result.features
        assert "solidity" in result.features
        assert "defect_count" in result.features
        assert "aspect_ratio" in result.features

    def test_batch_classify(self):
        clf = BubbleClassifier()
        contours = [
            _make_circle_contour(),
            _make_rectangle_contour(),
            _make_spiky_contour(),
        ]
        results = clf.classify_batch(contours)
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_caption_near_edge(self):
        clf = BubbleClassifier()
        contour = _make_rectangle_contour(x=100, y=5, w=150, h=30)
        result = clf.classify(contour, image_shape=(800, 600), bbox=(100, 5, 150, 30))
        assert result.bubble_type in (BubbleType.CAPTION, BubbleType.NARRATION)

    def test_confidence_threshold(self):
        clf = BubbleClassifier(min_confidence=0.99)
        contour = _make_circle_contour()
        result = clf.classify(contour)
        assert isinstance(result.bubble_type, BubbleType)

    def test_empty_contour(self):
        clf = BubbleClassifier()
        contour = np.array([], dtype=np.int32).reshape(0, 1, 2)
        result = clf.classify(contour)
        assert result.bubble_type == BubbleType.UNKNOWN

    def test_classify_with_image_shape(self):
        clf = BubbleClassifier()
        contour = _make_circle_contour()
        result = clf.classify(contour, image_shape=(800, 600))
        assert "area_ratio" in result.features
        assert result.features["area_ratio"] > 0
