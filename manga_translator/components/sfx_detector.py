"""Sound effect (SFX) / onomatopoeia detection for manga pages.

Detects stylized text drawn directly on artwork outside speech bubbles,
such as impact sounds, ambient noises, and motion effects.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SFXType(Enum):
    """Type of sound effect."""
    IMPACT = "impact"
    AMBIENT = "ambient"
    MOTION = "motion"
    EMOTION = "emotion"
    UNKNOWN = "unknown"


@dataclass
class SFXRegion:
    """Detected sound effect region."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    confidence: float
    sfx_type: SFXType = SFXType.UNKNOWN
    mask: Optional[np.ndarray] = None


class SFXDetector:
    """Detects sound effects / onomatopoeia in manga pages.

    SFX are typically large, stylized text drawn directly on artwork,
    outside of speech bubbles.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 200000,
        min_stroke_width: float = 3.0,
        edge_threshold: int = 80,
        overlap_threshold: float = 0.3,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_stroke_width = min_stroke_width
        self.edge_threshold = edge_threshold
        self.overlap_threshold = overlap_threshold

    def detect_sfx(
        self,
        image: np.ndarray,
        bubble_regions: Optional[List] = None,
    ) -> List[SFXRegion]:
        """Detect SFX regions in a manga page."""
        if image is None or image.size == 0:
            return []

        bubble_regions = bubble_regions or []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        bubble_mask = np.zeros((h, w), dtype=np.uint8)
        for b in bubble_regions:
            bx, by, bw, bh = b.bbox
            bubble_mask[by:by + bh, bx:bx + bw] = 255

        candidates = self._find_stroke_regions(gray)

        sfx_regions: List[SFXRegion] = []
        sfx_id = 0
        for contour in candidates:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            region_mask = bubble_mask[y:y + ch, x:x + cw]
            if region_mask.size > 0:
                overlap_ratio = np.count_nonzero(region_mask) / region_mask.size
                if overlap_ratio > self.overlap_threshold:
                    continue

            confidence = self._compute_confidence(gray, contour, x, y, cw, ch)
            if confidence < 0.3:
                continue

            sfx_type = self._classify_sfx(contour, area, cw, ch)
            center = (x + cw // 2, y + ch // 2)
            sfx_regions.append(SFXRegion(
                id=sfx_id,
                bbox=(x, y, cw, ch),
                contour=contour,
                center=center,
                area=area,
                confidence=confidence,
                sfx_type=sfx_type,
            ))
            sfx_id += 1

        logger.debug("Detected %d SFX regions", len(sfx_regions))
        return sfx_regions

    def _find_stroke_regions(self, gray: np.ndarray) -> List[np.ndarray]:
        """Find regions with strong stroke-like edges."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        return list(contours)

    def _compute_confidence(
        self, gray: np.ndarray, contour: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> float:
        """Compute SFX detection confidence."""
        if w == 0 or h == 0:
            return 0.0
        region = gray[y:y + h, x:x + w]
        if region.size == 0:
            return 0.0

        edges = cv2.Canny(region, self.edge_threshold, self.edge_threshold * 2)
        edge_density = np.count_nonzero(edges) / edges.size
        contrast = float(np.std(region)) / 128.0

        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        mean_stroke = float(np.mean(dist[dist > 0])) if np.any(dist > 0) else 0.0
        stroke_score = min(mean_stroke / self.min_stroke_width, 1.0)

        confidence = 0.4 * min(edge_density * 5, 1.0) + 0.3 * min(contrast, 1.0) + 0.3 * stroke_score
        return round(min(confidence, 1.0), 3)

    def _classify_sfx(self, contour: np.ndarray, area: float, w: int, h: int) -> SFXType:
        """Classify SFX type based on shape features."""
        aspect_ratio = w / max(h, 1)
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            return SFXType.MOTION
        if area > 20000:
            return SFXType.IMPACT
        if area < 3000:
            return SFXType.AMBIENT
        if 0.7 < aspect_ratio < 1.4:
            return SFXType.EMOTION
        return SFXType.UNKNOWN
