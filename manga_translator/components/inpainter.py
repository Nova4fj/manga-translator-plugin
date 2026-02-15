"""Text removal via inpainting for manga pages."""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class InpaintResult:
    """Result from inpainting operation."""
    image: np.ndarray  # cleaned image region
    method_used: str
    quality_score: float  # 0.0 to 1.0 estimated quality


class Inpainter:
    """Removes text from manga bubbles using inpainting techniques."""

    METHODS = ("opencv_telea", "opencv_ns", "blur", "lama", "auto")

    def __init__(
        self,
        method: str = "opencv_telea",
        inpaint_radius: int = 5,
        blur_kernel_size: int = 15,
        mask_dilation: int = 5,
        lama_model_dir: Optional[str] = None,
        lama_device: str = "auto",
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown inpainting method '{method}'. "
                f"Choose from {self.METHODS}"
            )
        self.method = method
        self.inpaint_radius = inpaint_radius
        self.blur_kernel_size = blur_kernel_size
        self.mask_dilation = mask_dilation
        self._lama_model_dir = lama_model_dir
        self._lama_device = lama_device
        self._neural_inpainter = None

    # ------------------------------------------------------------------
    # Neural backend
    # ------------------------------------------------------------------

    def _get_neural_inpainter(self):
        """Lazy-load the neural inpainter."""
        if self._neural_inpainter is None:
            from manga_translator.components.neural_inpainter import NeuralInpainter
            self._neural_inpainter = NeuralInpainter(
                model_dir=self._lama_model_dir,
                device=self._lama_device,
            )
        return self._neural_inpainter

    def is_neural_available(self) -> bool:
        """Check if neural inpainting (LaMa) is available."""
        try:
            return self._get_neural_inpainter().is_available()
        except Exception:
            return False

    def inpaint_lama(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Neural inpainting using LaMa model.

        Produces significantly higher quality results than OpenCV methods,
        especially for complex backgrounds (screentone, gradients, patterns).
        Falls back to OpenCV Telea if the model is unavailable.
        """
        neural = self._get_neural_inpainter()
        if not neural.is_available():
            logger.warning("LaMa model not available, falling back to opencv_telea")
            return self.inpaint_telea(image, mask)

        try:
            return neural.inpaint(image, mask)
        except Exception as e:
            logger.error("LaMa inpainting failed: %s. Falling back to opencv_telea.", e)
            return self.inpaint_telea(image, mask)

    def inpaint_lama_regions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        padding: int = 32,
    ) -> np.ndarray:
        """Batch neural inpainting — process each region separately for efficiency.

        Args:
            image: Full page BGR image.
            mask: Full page mask (255 = inpaint).
            bboxes: List of (x, y, w, h) bounding boxes for each text region.
            padding: Extra context pixels around each bbox.

        Returns:
            Inpainted full-page image.
        """
        neural = self._get_neural_inpainter()
        if not neural.is_available():
            logger.warning("LaMa not available, falling back to opencv_telea for batch")
            return self.inpaint_telea(image, mask)

        result = image.copy()
        for bbox in bboxes:
            try:
                result = neural.inpaint_region(result, mask, bbox, padding=padding)
            except Exception as e:
                logger.warning("LaMa region inpaint failed for bbox %s: %s", bbox, e)
                # Fall back to OpenCV for this region
                x, y, w, h = bbox
                x1, y1 = max(x - padding, 0), max(y - padding, 0)
                x2 = min(x + w + padding, image.shape[1])
                y2 = min(y + h + padding, image.shape[0])
                crop_mask = mask[y1:y2, x1:x2]
                crop_img = result[y1:y2, x1:x2]
                inpainted = cv2.inpaint(
                    self._ensure_bgr(crop_img), crop_mask,
                    self.inpaint_radius, cv2.INPAINT_TELEA
                )
                if crop_img.ndim == 2:
                    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
                result[y1:y2, x1:x2] = inpainted
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove_text_with_fallback(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        quality_threshold: float = 0.5,
        constraint_mask: Optional[np.ndarray] = None,
    ) -> InpaintResult:
        """Remove text with automatic quality-based fallback.

        Tries the configured method first.  If the quality score is below
        *quality_threshold* and the method is ``"lama"``, falls back to
        OpenCV Telea.  If the initial method was an OpenCV variant and
        LaMa is available, tries LaMa as an alternative.

        Returns the best result by quality score.
        """
        primary = self.remove_text(image, text_mask, constraint_mask=constraint_mask)

        if primary.quality_score >= quality_threshold:
            return primary

        logger.info(
            "Primary inpainting quality %.3f < threshold %.3f, trying fallback",
            primary.quality_score, quality_threshold,
        )

        # Determine fallback method
        if primary.method_used == "lama":
            fallback_method = "opencv_telea"
        elif self.is_neural_available():
            fallback_method = "lama"
        elif primary.method_used == "opencv_telea":
            fallback_method = "opencv_ns"
        else:
            fallback_method = "opencv_telea"

        # Don't retry the same method
        if fallback_method == primary.method_used:
            return primary

        # Run fallback
        mask = self._ensure_mask(text_mask)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.mask_dilation * 2 + 1, self.mask_dilation * 2 + 1),
        )
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Clip dilated mask to bubble contour to prevent bleed
        if constraint_mask is not None:
            dilated_mask = cv2.bitwise_and(
                dilated_mask, self._ensure_mask(constraint_mask)
            )

        if fallback_method == "lama":
            fallback_img = self.inpaint_lama(image, dilated_mask)
        elif fallback_method == "opencv_ns":
            fallback_img = self.inpaint_navier_stokes(image, dilated_mask)
        else:
            fallback_img = self.inpaint_telea(image, dilated_mask)

        fallback_quality = self.assess_quality(image, fallback_img, dilated_mask)

        if fallback_quality > primary.quality_score:
            logger.info(
                "Fallback %s (quality %.3f) beat primary %s (%.3f)",
                fallback_method, fallback_quality,
                primary.method_used, primary.quality_score,
            )
            return InpaintResult(
                image=fallback_img,
                method_used=fallback_method,
                quality_score=fallback_quality,
            )

        return primary

    def remove_text(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        constraint_mask: Optional[np.ndarray] = None,
    ) -> InpaintResult:
        """Remove text from *image* using the configured method.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale source image.
        text_mask : np.ndarray
            Single-channel uint8 mask where 255 marks text pixels to remove.
        constraint_mask : np.ndarray, optional
            Binary mask constraining the inpainting region.  After dilation,
            the mask is clipped to this boundary so that inpainting does not
            bleed outside the bubble contour.

        Returns
        -------
        InpaintResult
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")
        if text_mask is None or text_mask.size == 0:
            raise ValueError("Text mask is empty.")

        # Ensure mask is single-channel uint8
        mask = self._ensure_mask(text_mask)

        # Dilate the mask so inpainting covers anti-aliased text edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.mask_dilation * 2 + 1, self.mask_dilation * 2 + 1),
        )
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Clip dilated mask to bubble contour to prevent bleed
        if constraint_mask is not None:
            dilated_mask = cv2.bitwise_and(
                dilated_mask, self._ensure_mask(constraint_mask)
            )

        # Select method --------------------------------------------------
        chosen_method = self.method
        if chosen_method == "auto":
            chosen_method = self.auto_select_method(image, dilated_mask)
            logger.info("Auto-selected inpainting method: %s", chosen_method)

        # Dispatch --------------------------------------------------------
        if chosen_method == "lama":
            inpainted = self.inpaint_lama(image, dilated_mask)
        elif chosen_method == "opencv_telea":
            inpainted = self.inpaint_telea(image, dilated_mask)
        elif chosen_method == "opencv_ns":
            inpainted = self.inpaint_navier_stokes(image, dilated_mask)
        elif chosen_method == "blur":
            inpainted = self.inpaint_blur(image, dilated_mask)
        else:
            # Fallback
            inpainted = self.inpaint_telea(image, dilated_mask)

        quality = self.assess_quality(image, inpainted, dilated_mask)
        logger.debug(
            "Inpainting complete (method=%s, quality=%.3f)",
            chosen_method,
            quality,
        )

        return InpaintResult(
            image=inpainted,
            method_used=chosen_method,
            quality_score=quality,
        )

    def create_text_mask(
        self, image: np.ndarray, bubble_mask: np.ndarray
    ) -> np.ndarray:
        """Create a mask of text pixels within a speech-bubble region.

        Uses adaptive thresholding to find dark text on the light bubble
        background that is typical of manga.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale source image (full page or region).
        bubble_mask : np.ndarray
            Single-channel uint8 mask where 255 marks the interior of
            speech bubbles.

        Returns
        -------
        np.ndarray
            Binary uint8 mask (0 / 255) marking text pixels.
        """
        bubble_mask = self._ensure_mask(bubble_mask)

        # 1. Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy() if image.ndim == 2 else image[:, :, 0].copy()

        # 2. Isolate bubble interior — set everything outside to white so
        #    thresholding ignores it.
        bubble_region = gray.copy()
        bubble_region[bubble_mask == 0] = 255

        # 3. Determine bubble background brightness.  For light bubbles
        #    (typical manga), use a direct intensity threshold that is far
        #    more reliable than adaptive/Otsu which can miss thin or
        #    anti-aliased strokes.
        interior_pixels = gray[bubble_mask > 0]
        bg_bright = len(interior_pixels) > 0 and float(np.percentile(interior_pixels, 90)) > 180

        if bg_bright:
            # Light bubble: mark every pixel significantly darker than the
            # background as text.  The 90th-percentile represents the clean
            # background; anything >50 levels below it is clearly text.
            bg_level = float(np.percentile(interior_pixels, 90))
            dark_cutoff = max(int(bg_level - 50), 80)
            thresh = np.zeros_like(gray, dtype=np.uint8)
            thresh[gray < dark_cutoff] = 255
            thresh[bubble_mask == 0] = 0
        else:
            # Darker / patterned bubble: fall back to adaptive + Otsu
            block_size = self._adaptive_block_size(bubble_region)
            thresh_adaptive = cv2.adaptiveThreshold(
                bubble_region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=block_size,
                C=10,
            )
            _, thresh_otsu = cv2.threshold(
                bubble_region, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            thresh = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
            thresh = cv2.bitwise_and(thresh, thresh, mask=bubble_mask)

        # 4. Morphological cleanup — close small gaps inside characters,
        #    then open to remove noise specks.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # 5. Filter out tiny connected components (noise) — keep only
        #    components whose area is >= a minimum fraction of the bubble.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        bubble_area = max(int(np.count_nonzero(bubble_mask)), 1)
        # Minimum area: 0.05 % of bubble area, but at least 8 pixels
        min_area = max(int(bubble_area * 0.0005), 8)

        text_mask = np.zeros_like(cleaned)
        for i in range(1, num_labels):  # skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                text_mask[labels == i] = 255

        return text_mask

    # ------------------------------------------------------------------
    # Inpainting back-ends
    # ------------------------------------------------------------------

    def inpaint_telea(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fast marching method (Telea, 2004).

        Good general-purpose inpainting; fast and handles most bubble
        backgrounds well.
        """
        mask = self._ensure_mask(mask)
        src = self._ensure_bgr(image)
        result = cv2.inpaint(src, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        # Return in the same colour space as input
        if image.ndim == 2:
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        return result

    def inpaint_navier_stokes(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Navier-Stokes fluid dynamics method (Bertalmio et al., 2001).

        Preserves edges better at the cost of more computation.
        """
        mask = self._ensure_mask(mask)
        src = self._ensure_bgr(image)
        result = cv2.inpaint(src, mask, self.inpaint_radius, cv2.INPAINT_NS)
        if image.ndim == 2:
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        return result

    def inpaint_blur(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Simple Gaussian-blur fill for near-uniform backgrounds.

        For plain white (or solid-colour) manga bubbles this is the
        fastest approach and produces very clean results.
        """
        mask = self._ensure_mask(mask)
        src = image.copy()

        # Ensure kernel size is odd
        ksize = self.blur_kernel_size
        if ksize % 2 == 0:
            ksize += 1

        # Heavy blur of the full image — text strokes disappear
        blurred = cv2.GaussianBlur(src, (ksize, ksize), 0)

        # Replace only the masked pixels with the blurred version
        mask_bool = mask > 0
        if src.ndim == 3:
            src[mask_bool] = blurred[mask_bool]
        else:
            src[mask_bool] = blurred[mask_bool]

        # Second lighter smoothing pass across the boundary to reduce
        # sharp transitions between original and filled pixels.
        boundary_ksize = max(ksize // 3, 3)
        if boundary_ksize % 2 == 0:
            boundary_ksize += 1
        # Dilate mask slightly for the blend zone
        blend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blend_mask = cv2.dilate(mask, blend_kernel, iterations=1)
        blend_zone = cv2.GaussianBlur(src, (boundary_ksize, boundary_ksize), 0)
        blend_bool = blend_mask > 0
        # Alpha-blend the boundary region
        alpha = 0.5
        if src.ndim == 3:
            src[blend_bool] = cv2.addWeighted(
                src[blend_bool], 1 - alpha, blend_zone[blend_bool], alpha, 0
            )
        else:
            src[blend_bool] = (
                (1 - alpha) * src[blend_bool] + alpha * blend_zone[blend_bool]
            ).astype(src.dtype)

        return src

    # ------------------------------------------------------------------
    # Quality / analysis helpers
    # ------------------------------------------------------------------

    def assess_quality(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Estimate inpainting quality (0.0 = poor, 1.0 = perfect).

        Checks two aspects around the inpainted region:
        1. **Colour consistency** — the mean colour of inpainted pixels
           should be close to the mean colour of the surrounding area.
        2. **Edge artefacts** — there should be few strong edges along
           the boundary of the inpainted zone; a seamless fill has a
           smooth transition.
        """
        mask = self._ensure_mask(mask)
        if np.count_nonzero(mask) == 0:
            return 1.0  # nothing was inpainted

        # Work in grayscale for analysis
        if inpainted.ndim == 3:
            inp_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
            cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            inp_gray = inpainted

        # --- Colour consistency score ---
        # Compare mean intensity inside the inpainted region with the
        # surrounding (dilated mask minus original mask).
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        surround_mask = cv2.dilate(mask, dilate_k, iterations=1)
        surround_mask = cv2.subtract(surround_mask, mask)

        if np.count_nonzero(surround_mask) == 0:
            colour_score = 0.5  # cannot assess
        else:
            mean_inner = float(cv2.mean(inp_gray, mask=mask)[0])
            mean_surround = float(cv2.mean(inp_gray, mask=surround_mask)[0])
            # Maximum expected difference = 80 grey levels
            diff = abs(mean_inner - mean_surround)
            colour_score = max(0.0, 1.0 - diff / 80.0)

        # --- Edge-artefact score ---
        # Detect edges in the inpainted image; strong edges right at the
        # boundary of the fill are a sign of visible seams.
        edges = cv2.Canny(inp_gray, 50, 150)
        # Build a thin ring at the boundary of the mask
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner = cv2.erode(mask, erode_k, iterations=1)
        boundary = cv2.subtract(mask, inner)
        if np.count_nonzero(boundary) == 0:
            edge_score = 1.0
        else:
            edge_pixels = cv2.bitwise_and(edges, edges, mask=boundary)
            edge_ratio = float(np.count_nonzero(edge_pixels)) / float(
                np.count_nonzero(boundary)
            )
            # Fewer boundary edges = better; ratio above 0.4 is poor
            edge_score = max(0.0, 1.0 - edge_ratio / 0.4)

        # --- Variance consistency score ---
        # The local variance inside the inpainted region should be
        # similar to the local variance outside.
        var_inner = float(np.var(inp_gray[mask > 0])) if np.count_nonzero(mask) > 0 else 0.0
        var_surround = (
            float(np.var(inp_gray[surround_mask > 0]))
            if np.count_nonzero(surround_mask) > 0
            else var_inner
        )
        max_var = max(var_inner, var_surround, 1.0)
        var_diff = abs(var_inner - var_surround) / max_var
        var_score = max(0.0, 1.0 - var_diff)

        # --- Texture consistency score ---
        # Compare Laplacian energy (a texture measure) between inpainted
        # and surrounding regions.  Similar texture energy = good.
        lap = cv2.Laplacian(inp_gray, cv2.CV_64F)
        if np.count_nonzero(mask) > 0 and np.count_nonzero(surround_mask) > 0:
            lap_inner = float(np.mean(np.abs(lap[mask > 0])))
            lap_surround = float(np.mean(np.abs(lap[surround_mask > 0])))
            max_lap = max(lap_inner, lap_surround, 1.0)
            texture_diff = abs(lap_inner - lap_surround) / max_lap
            texture_score = max(0.0, 1.0 - texture_diff)
        else:
            texture_score = 0.5

        # Weighted combination
        quality = (
            0.30 * colour_score
            + 0.30 * edge_score
            + 0.20 * var_score
            + 0.20 * texture_score
        )
        return round(min(max(quality, 0.0), 1.0), 4)

    def analyze_background_complexity(
        self, image: np.ndarray, mask: np.ndarray
    ) -> float:
        """Analyse how complex the background is around the text.

        Returns a value in [0, 1] where 0 is a perfectly uniform
        background (e.g. plain white bubble) and 1 is highly textured
        (e.g. screentone or patterned fill).

        The analysis considers:
        * Intensity variance in the surrounding region.
        * Edge density (Canny) in the surrounding region.
        * Texture energy via the Laplacian.
        """
        mask = self._ensure_mask(mask)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Build a surrounding-region mask (exclude text itself)
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        surround = cv2.dilate(mask, dilate_k, iterations=1)
        surround = cv2.subtract(surround, mask)
        if np.count_nonzero(surround) == 0:
            return 0.5  # indeterminate

        region_pixels = gray[surround > 0]

        # 1. Normalised variance (cap at 2000)
        variance = float(np.var(region_pixels))
        var_score = min(variance / 2000.0, 1.0)

        # 2. Edge density via Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = edges[surround > 0]
        edge_density = float(np.count_nonzero(edge_pixels)) / max(
            float(edge_pixels.size), 1.0
        )
        edge_score = min(edge_density / 0.3, 1.0)

        # 3. Laplacian energy (texture measure)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_region = np.abs(lap[surround > 0])
        lap_energy = float(np.mean(lap_region))
        texture_score = min(lap_energy / 50.0, 1.0)

        complexity = 0.35 * var_score + 0.35 * edge_score + 0.30 * texture_score
        return round(min(max(complexity, 0.0), 1.0), 4)

    def auto_select_method(
        self, image: np.ndarray, mask: np.ndarray
    ) -> str:
        """Automatically choose the best inpainting method.

        * Simple / uniform background  -> ``"blur"`` (fastest, cleanest)
        * Moderate complexity           -> ``"opencv_telea"``
        * High complexity (screentone)  -> ``"lama"`` if available, else ``"opencv_ns"``
        """
        complexity = self.analyze_background_complexity(image, mask)
        logger.debug("Background complexity: %.3f", complexity)

        if complexity < 0.15:
            return "blur"
        elif complexity < 0.55:
            return "opencv_telea"
        else:
            if self.is_neural_available():
                return "lama"
            return "opencv_ns"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_mask(mask: np.ndarray) -> np.ndarray:
        """Guarantee a single-channel uint8 binary mask."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Ensure strictly binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Convert grayscale to BGR if needed (OpenCV inpaint needs BGR)."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def _adaptive_block_size(image: np.ndarray) -> int:
        """Choose an adaptive-threshold block size based on image dimensions.

        The block size must be odd and >= 3.  We pick roughly 1/20th of
        the smaller dimension, clamped to a reasonable range.
        """
        h, w = image.shape[:2]
        size = max(h, w) // 20
        size = max(size, 11)
        size = min(size, 101)
        if size % 2 == 0:
            size += 1
        return size
