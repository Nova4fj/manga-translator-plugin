"""Pre-OCR text region filtering to reduce false positives.

Analyzes detected bubble regions to estimate whether they actually
contain text, avoiding expensive OCR calls on empty or non-text regions.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TextRegionScore:
    """Analysis result for a candidate text region."""
    has_text: bool
    confidence: float  # 0.0-1.0
    edge_density: float
    variance: float
    text_line_score: float
    region_type: str  # "text", "empty", "solid", "gradient", "noise"


class TextRegionFilter:
    """Filters bubble regions to identify those likely containing text.

    Uses lightweight image analysis (no OCR) to estimate text presence:
    - Edge density: text regions have moderate edge density
    - Variance: text has moderate variance (not solid, not pure noise)
    - Horizontal/vertical line detection: text forms line patterns
    - Mean intensity: text on white bubbles has specific brightness range
    """

    def __init__(
        self,
        min_edge_density: float = 0.02,
        max_edge_density: float = 0.35,
        min_variance: float = 100.0,
        min_confidence: float = 0.3,
    ):
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density
        self.min_variance = min_variance
        self.min_confidence = min_confidence

    def analyze_region(self, region: np.ndarray) -> TextRegionScore:
        """Analyze a single cropped region for text presence."""
        if region is None or region.size == 0:
            return TextRegionScore(False, 0.0, 0.0, 0.0, 0.0, "empty")

        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        h, w = gray.shape[:2]
        total_pixels = h * w
        if total_pixels < 100:  # Too small for meaningful analysis
            return TextRegionScore(False, 0.0, 0.0, 0.0, 0.0, "empty")

        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / total_pixels

        # 2. Pixel variance
        variance = float(np.var(gray))

        # 3. Mean intensity
        mean_intensity = float(np.mean(gray))

        # 4. Text line score — horizontal projection analysis
        # Text creates periodic horizontal bands of high/low intensity
        text_line_score = self._compute_text_line_score(gray)

        # Classify region type
        region_type, confidence = self._classify(
            edge_density, variance, mean_intensity, text_line_score
        )

        has_text = bool(region_type == "text" and confidence >= self.min_confidence)

        return TextRegionScore(
            has_text=has_text,
            confidence=confidence,
            edge_density=edge_density,
            variance=variance,
            text_line_score=text_line_score,
            region_type=region_type,
        )

    def filter_regions(
        self, regions: List[np.ndarray]
    ) -> List[Tuple[int, TextRegionScore]]:
        """Analyze multiple regions. Returns list of (index, score) for text regions."""
        results = []
        for i, region in enumerate(regions):
            score = self.analyze_region(region)
            if score.has_text:
                results.append((i, score))
        return results

    def _compute_text_line_score(self, gray: np.ndarray) -> float:
        """Estimate text line presence via horizontal projection profile.

        Text creates alternating bands of dark (text) and light (background)
        in the horizontal projection. The variance of the projection
        correlates with text presence.
        """
        h, w = gray.shape
        if h < 5 or w < 5:
            return 0.0

        # Horizontal projection: mean intensity per row
        projection = np.mean(gray, axis=1)

        # Normalize projection
        proj_range = projection.max() - projection.min()
        if proj_range < 10:  # Very uniform = no text
            return 0.0

        # Count zero-crossings of the derivative (transitions = text lines)
        derivative = np.diff(projection)
        # Smooth derivative
        if len(derivative) > 5:
            kernel = np.ones(3) / 3
            derivative = np.convolve(derivative, kernel, mode='valid')

        zero_crossings = np.sum(np.abs(np.diff(np.sign(derivative))) > 0)

        # Normalize by height — expect ~2-10 crossings per text line
        crossings_per_height = zero_crossings / h

        # Score: peaks around 0.05-0.3 crossings per pixel height
        if crossings_per_height < 0.02:
            return 0.1
        elif crossings_per_height > 0.5:
            return 0.2  # Probably noise, not text
        else:
            return min(crossings_per_height * 3, 1.0)

    def _classify(
        self,
        edge_density: float,
        variance: float,
        mean_intensity: float,
        text_line_score: float,
    ) -> Tuple[str, float]:
        """Classify region type and confidence."""
        # Solid region (no edges, low variance)
        if edge_density < self.min_edge_density and variance < self.min_variance:
            return "solid", 0.1

        # Very noisy (too many edges)
        if edge_density > self.max_edge_density:
            return "noise", 0.1

        # Gradient (low edges but moderate variance)
        if edge_density < self.min_edge_density and variance >= self.min_variance:
            return "gradient", 0.15

        # Text candidate
        confidence = 0.0

        # Edge density contribution (moderate = good)
        if self.min_edge_density <= edge_density <= 0.2:
            confidence += 0.3
        elif edge_density <= self.max_edge_density:
            confidence += 0.15

        # Variance contribution
        if variance > 500:
            confidence += 0.25
        elif variance > self.min_variance:
            confidence += 0.15

        # Text line score contribution
        confidence += text_line_score * 0.3

        # Mean intensity: text on white background is typically bright overall
        if mean_intensity > 180:
            confidence += 0.15
        elif mean_intensity > 100:
            confidence += 0.05

        return "text", min(confidence, 1.0)
