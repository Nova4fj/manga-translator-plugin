"""Input validation for the translation pipeline."""

import os
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class InputValidator:
    """Validates inputs before they enter the pipeline."""

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    # Limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_DIMENSION = 10000  # pixels
    MIN_DIMENSION = 10  # pixels

    VALID_LANGUAGES = {'ja', 'zh', 'ko', 'en', 'fr', 'de', 'es', 'pt', 'ru', 'ar', 'auto'}

    @classmethod
    def validate_image_path(cls, path: str) -> str:
        """Validate an image file path. Returns the path or raises ValidationError."""
        if not path:
            raise ValidationError("Image path is empty")
        if not os.path.exists(path):
            raise ValidationError(f"File not found: {path}")
        if not os.path.isfile(path):
            raise ValidationError(f"Not a file: {path}")

        _, ext = os.path.splitext(path)
        if ext.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported format: {ext}. Supported: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
            )

        size = os.path.getsize(path)
        if size == 0:
            raise ValidationError(f"File is empty: {path}")
        if size > cls.MAX_FILE_SIZE:
            raise ValidationError(
                f"File too large: {size / 1024 / 1024:.1f}MB (max {cls.MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
            )

        return path

    @classmethod
    def validate_image_array(cls, image: np.ndarray) -> np.ndarray:
        """Validate a numpy image array. Returns the array or raises ValidationError."""
        if image is None:
            raise ValidationError("Image is None")
        if not isinstance(image, np.ndarray):
            raise ValidationError(f"Expected numpy array, got {type(image).__name__}")
        if image.ndim not in (2, 3):
            raise ValidationError(f"Expected 2D or 3D array, got {image.ndim}D")
        if image.size == 0:
            raise ValidationError("Image is empty (0 pixels)")

        h, w = image.shape[:2]
        if h < cls.MIN_DIMENSION or w < cls.MIN_DIMENSION:
            raise ValidationError(
                f"Image too small: {w}x{h} (minimum {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})"
            )
        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            raise ValidationError(
                f"Image too large: {w}x{h} (maximum {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})"
            )

        return image

    @classmethod
    def validate_language(cls, lang: str, param_name: str = "language") -> str:
        """Validate a language code."""
        if not lang:
            raise ValidationError(f"{param_name} is empty")
        if lang.lower() not in cls.VALID_LANGUAGES:
            raise ValidationError(
                f"Unknown {param_name}: '{lang}'. Valid: {', '.join(sorted(cls.VALID_LANGUAGES))}"
            )
        return lang.lower()

    @classmethod
    def validate_output_path(cls, path: str) -> str:
        """Validate an output file path (parent dir must exist or be creatable)."""
        if not path:
            raise ValidationError("Output path is empty")
        parent = os.path.dirname(path) or "."
        if not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                raise ValidationError(f"Cannot create output directory: {e}")
        return path

    @classmethod
    def validate_batch_paths(cls, paths: List[str]) -> Tuple[List[str], List[str]]:
        """Validate a list of paths. Returns (valid_paths, error_messages)."""
        valid = []
        errors = []
        for path in paths:
            try:
                cls.validate_image_path(path)
                valid.append(path)
            except ValidationError as e:
                errors.append(str(e))
        return valid, errors
