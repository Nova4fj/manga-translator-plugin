"""Bubble shape classifier for manga speech bubbles.

Classifies detected bubbles into semantic types (speech, thought, shout,
narration, caption) based on contour shape analysis.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BubbleType(Enum):
    """Semantic type of a speech bubble."""
    SPEECH = "speech"
    THOUGHT = "thought"
    SHOUT = "shout"
    NARRATION = "narration"
    CAPTION = "caption"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of bubble type classification."""
    bubble_type: BubbleType
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)


class BubbleClassifier:
    """Classifies manga bubbles into semantic types using contour analysis."""

    def __init__(
        self,
        min_confidence: float = 0.3,
        caption_max_area_ratio: float = 0.02,
    ):
        self.min_confidence = min_confidence
        self.caption_max_area_ratio = caption_max_area_ratio

    def classify(
        self,
        contour: np.ndarray,
        image_shape: Optional[tuple] = None,
        bbox: Optional[tuple] = None,
    ) -> ClassificationResult:
        """Classify a single bubble contour."""
        features = self._extract_features(contour, image_shape, bbox)

        scores = {
            BubbleType.SPEECH: self._score_speech(features),
            BubbleType.THOUGHT: self._score_thought(features),
            BubbleType.SHOUT: self._score_shout(features),
            BubbleType.NARRATION: self._score_narration(features),
            BubbleType.CAPTION: self._score_caption(features),
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < self.min_confidence:
            best_type = BubbleType.UNKNOWN
            best_score = 1.0 - max(scores.values())

        return ClassificationResult(
            bubble_type=best_type,
            confidence=round(best_score, 3),
            features=features,
        )

    def classify_batch(
        self,
        contours: List[np.ndarray],
        image_shape: Optional[tuple] = None,
        bboxes: Optional[List[tuple]] = None,
    ) -> List[ClassificationResult]:
        """Classify a batch of bubble contours."""
        bboxes = bboxes or [None] * len(contours)
        return [
            self.classify(c, image_shape, b)
            for c, b in zip(contours, bboxes)
        ]

    def _extract_features(
        self,
        contour: np.ndarray,
        image_shape: Optional[tuple] = None,
        bbox: Optional[tuple] = None,
    ) -> Dict[str, float]:
        """Extract shape features from a contour."""
        if len(contour) == 0:
            return {
                "circularity": 0.0, "solidity": 0.0,
                "defect_count": 0.0, "avg_defect_depth": 0.0,
                "aspect_ratio": 0.0, "area_ratio": 0.0,
                "edge_proximity": 0.0, "vertex_count": 0.0, "area": 0.0,
            }

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defect_count = 0
        avg_defect_depth = 0.0
        if len(contour) >= 4 and len(hull_indices) >= 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    defect_count = len(defects)
                    depths = defects[:, 0, 3] / 256.0
                    avg_defect_depth = float(np.mean(depths))
            except cv2.error:
                pass

        if bbox is not None:
            _, _, bw, bh = bbox
        else:
            _, _, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / max(bh, 1)

        area_ratio = 0.0
        if image_shape is not None:
            page_area = image_shape[0] * image_shape[1]
            area_ratio = area / page_area if page_area > 0 else 0.0

        edge_proximity = 0.0
        if image_shape is not None and bbox is not None:
            x, y, bw, bh = bbox
            ih, iw = image_shape
            top_dist = y / ih
            bottom_dist = (ih - y - bh) / ih
            left_dist = x / iw
            right_dist = (iw - x - bw) / iw
            edge_proximity = 1.0 - min(top_dist, bottom_dist, left_dist, right_dist)

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(approx)

        return {
            "circularity": round(circularity, 4),
            "solidity": round(solidity, 4),
            "defect_count": float(defect_count),
            "avg_defect_depth": round(avg_defect_depth, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "area_ratio": round(area_ratio, 6),
            "edge_proximity": round(edge_proximity, 4),
            "vertex_count": float(vertex_count),
            "area": float(area),
        }

    def _score_speech(self, f: Dict[str, float]) -> float:
        # Require minimum circularity — rectangles should not score as speech
        if f["circularity"] < 0.7 or f["area"] == 0:
            return 0.0
        score = 0.0
        score += 0.3 * min(f["circularity"] / 0.8, 1.0)
        score += 0.3 * min(f["solidity"] / 0.9, 1.0)
        score += 0.2 * max(0, 1.0 - f["avg_defect_depth"] / 10.0)
        ar = f["aspect_ratio"]
        # Penalize extreme aspect ratios (speech bubbles are roughly round)
        score += 0.2 * max(0, 1.0 - abs(ar - 1.2) / 1.5)
        return round(min(score, 1.0), 3)

    def _score_thought(self, f: Dict[str, float]) -> float:
        if f["area"] == 0:
            return 0.0
        score = 0.0
        score += 0.3 * max(0, 1.0 - f["solidity"])
        score += 0.3 * min(f["defect_count"] / 15.0, 1.0)
        score += 0.2 * min(f["avg_defect_depth"] / 8.0, 1.0)
        score += 0.2 * min(f["circularity"] / 0.5, 1.0)
        return round(min(score, 1.0), 3)

    def _score_shout(self, f: Dict[str, float]) -> float:
        if f["area"] == 0:
            return 0.0
        score = 0.0
        score += 0.3 * min(f["avg_defect_depth"] / 15.0, 1.0)
        score += 0.3 * min(f["vertex_count"] / 12.0, 1.0)
        sol = f["solidity"]
        score += 0.2 * max(0, 1.0 - abs(sol - 0.75) / 0.3)
        score += 0.2 * max(0, 1.0 - f["circularity"])
        return round(min(score, 1.0), 3)

    def _score_narration(self, f: Dict[str, float]) -> float:
        if f["area"] == 0:
            return 0.0
        score = 0.0
        score += 0.25 * max(0, 1.0 - f["circularity"] / 0.6)
        score += 0.25 * min(f["solidity"] / 0.95, 1.0)
        vc = f["vertex_count"]
        score += 0.25 * max(0, 1.0 - abs(vc - 4) / 8.0)
        score += 0.25 * max(0, 1.0 - f["defect_count"] / 5.0)
        return round(min(score, 1.0), 3)

    def _score_caption(self, f: Dict[str, float]) -> float:
        narr_score = self._score_narration(f)
        size_score = max(0, 1.0 - f["area_ratio"] / self.caption_max_area_ratio) if f["area_ratio"] > 0 else 0.5
        edge_score = f["edge_proximity"]
        score = 0.4 * narr_score + 0.3 * size_score + 0.3 * edge_score
        return round(min(score, 1.0), 3)
