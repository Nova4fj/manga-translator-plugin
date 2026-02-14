"""Image processing utilities for the manga translator plugin."""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from file path."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(image: np.ndarray, path: str, quality: int = 95) -> None:
    """Save image to file."""
    ext = path.rsplit(".", 1)[-1].lower()
    params = []
    if ext in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    cv2.imwrite(path, image, params)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_for_processing(
    image: np.ndarray, max_dimension: int = 4096
) -> Tuple[np.ndarray, float]:
    """Resize image if too large for processing. Returns (resized, scale_factor)."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_dimension:
        return image, 1.0

    scale = max_dimension / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_bbox(
    bbox: Tuple[int, int, int, int], scale: float
) -> Tuple[int, int, int, int]:
    """Scale a bounding box by a factor."""
    x, y, w, h = bbox
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))


def crop_region(
    image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 0
) -> np.ndarray:
    """Crop a region from an image with optional padding."""
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)
    return image[y1:y2, x1:x2].copy()


def numpy_to_pil(image: np.ndarray):
    """Convert OpenCV BGR image to PIL Image."""
    from PIL import Image

    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    arr = np.array(pil_image)
    if len(arr.shape) == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def estimate_memory_usage(image: np.ndarray) -> int:
    """Estimate memory usage in bytes for processing this image."""
    # Rough estimate: ~10x image size for full pipeline
    return image.nbytes * 10
