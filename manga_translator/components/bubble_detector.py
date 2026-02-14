"""Speech bubble detection for manga pages using OpenCV."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BubbleRegion:
    """Detected speech bubble region."""
    id: int
    contour: np.ndarray  # OpenCV contour
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: float
    confidence: float  # 0.0 to 1.0
    shape_type: str  # "oval", "rectangle", "irregular", "thought"
    mask: Optional[np.ndarray] = None  # binary mask of the bubble interior


class BubbleDetector:
    """Detects speech bubbles in manga pages using computer vision."""

    def __init__(
        self,
        min_area: int = 1000,
        max_area: int = 500000,
        edge_sensitivity: int = 100,
        contour_approx_epsilon: float = 0.02,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.edge_sensitivity = edge_sensitivity
        self.contour_approx_epsilon = contour_approx_epsilon
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def detect_bubbles(self, image: np.ndarray) -> List[BubbleRegion]:
        """Main detection pipeline.

        Args:
            image: BGR or grayscale input image (numpy array from cv2.imread).

        Returns:
            List of ``BubbleRegion`` objects sorted in manga reading order.
        """
        if image is None or image.size == 0:
            return []

        preprocessed = self.preprocess_image(image)
        contours = self.find_bubble_contours(preprocessed)

        bubbles: List[BubbleRegion] = []
        bubble_id = 0

        for contour in contours:
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Reject contours that span nearly the entire image (borders)
            img_h, img_w = image.shape[:2]
            if w > img_w * 0.95 and h > img_h * 0.95:
                continue

            # Aspect-ratio filter
            aspect_ratio = w / h if h > 0 else 0.0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            area = cv2.contourArea(contour)

            # Area filter
            if area < self.min_area or area > self.max_area:
                continue

            shape_type = self.classify_shape(contour)
            confidence = self.score_confidence(contour, shape_type)

            # Only keep detections with meaningful confidence
            if confidence < 0.15:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            mask = self.create_bubble_mask(contour, image.shape[:2])

            bubble = BubbleRegion(
                id=bubble_id,
                contour=contour,
                bbox=(x, y, w, h),
                center=(cx, cy),
                area=area,
                confidence=confidence,
                shape_type=shape_type,
                mask=mask,
            )
            bubbles.append(bubble)
            bubble_id += 1

        bubbles = self.sort_reading_order(bubbles)

        # Re-assign sequential IDs after sorting
        for idx, bubble in enumerate(bubbles):
            bubble.id = idx

        return bubbles

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale, blur, and apply adaptive thresholding.

        The pipeline is tuned for white / light-coloured bubbles sitting on
        varied manga artwork backgrounds.  Adaptive thresholding handles the
        local contrast differences between panels far better than a single
        global threshold.

        Returns:
            Binary image (uint8, values 0 or 255) where bubble interiors are
            white and the background is black.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gentle Gaussian blur to remove scan noise / screentone while
        # keeping bubble edges intact.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold — THRESH_BINARY gives us white foreground on
        # black background.  A fairly large block size (51) ensures we adapt
        # over panel-sized regions.
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=10,
        )

        # Morphological close to bridge small gaps in bubble outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Edge detection with Canny to find strong bubble boundaries
        edges = cv2.Canny(blurred, self.edge_sensitivity // 2, self.edge_sensitivity)

        # Dilate edges slightly so that nearby contour segments merge
        edge_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Combine the two sources: the adaptive-threshold binary *minus* the
        # detected edges.  This tends to keep the bright bubble interiors
        # while cutting along the dark outlines.
        combined = cv2.subtract(closed, edge_dilated)

        # A second morphological close on the combined result to smooth any
        # remaining gaps.
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        return combined

    # ------------------------------------------------------------------
    # Contour extraction
    # ------------------------------------------------------------------

    def find_bubble_contours(self, preprocessed: np.ndarray) -> List[np.ndarray]:
        """Find external contours in the preprocessed binary image and return
        those whose area falls within ``[min_area, max_area]``.

        The preprocessing produces white (255) for bubble interiors. We find
        contours of these white regions directly. If the white regions cover
        most of the image (>80%), the polarity is inverted first so that
        contour detection targets the bubble interiors correctly.

        An approximation step (``approxPolyDP``) simplifies the contours to
        reduce noise while keeping the overall shape.
        """
        # If white pixels dominate, bubbles are white regions — invert so
        # findContours picks up the bubble boundaries as external contours.
        white_ratio = np.mean(preprocessed > 0)
        if white_ratio > 0.80:
            binary = cv2.bitwise_not(preprocessed)
        else:
            binary = preprocessed

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Approximate the contour to smooth out jagged screentone edges
            perimeter = cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(
                cnt, self.contour_approx_epsilon * perimeter, closed=True
            )
            filtered.append(approx)

        return filtered

    # ------------------------------------------------------------------
    # Shape classification
    # ------------------------------------------------------------------

    def classify_shape(self, contour: np.ndarray) -> str:
        """Classify a contour as ``oval``, ``rectangle``, ``thought``, or
        ``irregular`` based on geometric descriptors.

        Metrics used
        ------------
        * **circularity** = 4 * pi * area / perimeter^2
        * **solidity** = area / convex_hull_area
        * **vertex_count** after ``approxPolyDP``
        * **aspect_ratio** of the bounding rectangle
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter == 0:
            return "irregular"

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Approximate polygon for vertex counting
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        vertex_count = len(approx)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0.0

        # ---- Decision rules ----

        # Thought bubbles tend to have bumpy / scalloped edges which lower
        # solidity while the overall shape is still roundish.
        if circularity > 0.35 and solidity < 0.85 and vertex_count > 8:
            return "thought"

        # Ovals / circles: high circularity AND high solidity
        if circularity > 0.55 and solidity > 0.88:
            return "oval"

        # Rectangles: few vertices, moderate circularity, high solidity
        if vertex_count <= 6 and solidity > 0.90 and 0.5 < aspect_ratio < 2.0:
            return "rectangle"

        # More relaxed oval check — elliptical bubbles that are elongated
        if circularity > 0.40 and solidity > 0.85:
            return "oval"

        return "irregular"

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def score_confidence(self, contour: np.ndarray, shape_type: str) -> float:
        """Score how likely a contour is a genuine speech bubble (0.0 – 1.0).

        The score is a weighted combination of:

        * **circularity** — bubbles are smooth and round-ish
        * **solidity** — bubbles have high solidity (few concavities)
        * **fill_ratio** — ratio of contour area to bounding-rect area
        * **smoothness** — inverse of the normalised vertex count after
          polygon approximation (smooth curves have fewer vertices)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter == 0 or area == 0:
            return 0.0

        # --- Circularity (0-1, 1 = perfect circle) ---
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        circularity = min(circularity, 1.0)

        # --- Solidity ---
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # --- Fill ratio (area / bounding rect area) ---
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h if w * h > 0 else 1
        fill_ratio = area / bbox_area

        # --- Smoothness (fewer vertices after approx = smoother) ---
        epsilon = 0.015 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        # Normalise: a perfect circle approximation has ~12-20 vertices;
        # very jagged contours can have hundreds.
        vertex_count = len(approx)
        smoothness = 1.0 - min(vertex_count / 80.0, 1.0)

        # --- Weighted combination ---
        weights = {
            "circularity": 0.30,
            "solidity": 0.30,
            "fill_ratio": 0.20,
            "smoothness": 0.20,
        }
        raw_score = (
            weights["circularity"] * circularity
            + weights["solidity"] * solidity
            + weights["fill_ratio"] * fill_ratio
            + weights["smoothness"] * smoothness
        )

        # Shape-type bonus: well-classified shapes get a small boost
        shape_bonus = {
            "oval": 0.10,
            "rectangle": 0.05,
            "thought": 0.05,
            "irregular": -0.05,
        }
        raw_score += shape_bonus.get(shape_type, 0.0)

        return float(np.clip(raw_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Mask creation
    # ------------------------------------------------------------------

    def create_bubble_mask(
        self, contour: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create a binary mask (uint8, 0/255) for the bubble interior.

        Args:
            contour: OpenCV contour array.
            image_shape: ``(height, width)`` of the source image.

        Returns:
            Single-channel mask with 255 inside the contour and 0 outside.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], contourIdx=0, color=255, thickness=cv2.FILLED)
        return mask

    # ------------------------------------------------------------------
    # Reading-order sort
    # ------------------------------------------------------------------

    def sort_reading_order(self, bubbles: List[BubbleRegion]) -> List[BubbleRegion]:
        """Sort bubbles in manga reading order: **right-to-left** columns,
        **top-to-bottom** within each column.

        The page is divided into vertical strips (columns).  Bubbles that
        share an overlapping horizontal span are considered to be in the same
        column and are sorted top-to-bottom.  Columns themselves are sorted
        right-to-left.
        """
        if not bubbles:
            return bubbles

        # Determine a reasonable column width — use the median bubble width
        widths = [b.bbox[2] for b in bubbles]
        median_width = float(np.median(widths))
        # Column tolerance: if two bubble centres are within 1.5x the median
        # width, consider them in the same column.
        col_tolerance = median_width * 1.5

        # Sort primarily by descending X (right-to-left), secondarily by
        # ascending Y (top-to-bottom).
        # Group into columns first.
        sorted_by_x = sorted(bubbles, key=lambda b: -b.center[0])

        columns: List[List[BubbleRegion]] = []
        for bubble in sorted_by_x:
            placed = False
            for col in columns:
                # Check if this bubble's centre X is close to the column's
                # average centre X.
                col_avg_x = np.mean([b.center[0] for b in col])
                if abs(bubble.center[0] - col_avg_x) <= col_tolerance:
                    col.append(bubble)
                    placed = True
                    break
            if not placed:
                columns.append([bubble])

        # Sort each column's bubbles top-to-bottom
        for col in columns:
            col.sort(key=lambda b: b.center[1])

        # Sort columns right-to-left by average X
        columns.sort(key=lambda col: -np.mean([b.center[0] for b in col]))

        # Flatten
        ordered: List[BubbleRegion] = []
        for col in columns:
            ordered.extend(col)

        return ordered

    # ------------------------------------------------------------------
    # Crop helper
    # ------------------------------------------------------------------

    def crop_bubble_region(self, image: np.ndarray, bubble: BubbleRegion) -> np.ndarray:
        """Extract the image region inside a bubble, masked to the bubble
        interior.

        Pixels outside the bubble contour are set to white (255) — the
        typical background colour expected by downstream OCR.

        Args:
            image: Source BGR or grayscale image.
            bubble: A ``BubbleRegion`` previously returned by
                ``detect_bubbles``.

        Returns:
            Cropped image with non-bubble pixels set to white.
        """
        x, y, w, h = bubble.bbox

        # Clamp to image bounds
        img_h, img_w = image.shape[:2]
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)

        cropped = image[y1:y2, x1:x2].copy()

        # Build a local mask for just the bounding-box region
        if bubble.mask is not None:
            local_mask = bubble.mask[y1:y2, x1:x2]
        else:
            local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            shifted_contour = bubble.contour.copy()
            shifted_contour[:, :, 0] -= x1
            shifted_contour[:, :, 1] -= y1
            cv2.drawContours(
                local_mask, [shifted_contour], 0, color=255, thickness=cv2.FILLED
            )

        # Set outside pixels to white
        if len(cropped.shape) == 3:
            cropped[local_mask == 0] = (255, 255, 255)
        else:
            cropped[local_mask == 0] = 255

        return cropped
