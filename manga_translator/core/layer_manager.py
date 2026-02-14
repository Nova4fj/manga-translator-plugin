"""GIMP layer management utilities.

This module provides layer operations that work both within GIMP (via gimpfu)
and in standalone mode (using numpy arrays as layer substitutes).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Layer:
    """Represents an image layer (GIMP-independent)."""

    name: str
    image: np.ndarray
    visible: bool = True
    opacity: float = 1.0
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class LayerStack:
    """Manages a stack of layers for the translation output."""

    layers: List[Layer] = field(default_factory=list)
    width: int = 0
    height: int = 0

    def add_layer(self, name: str, image: np.ndarray, **kwargs) -> Layer:
        """Add a new layer to the stack."""
        layer = Layer(name=name, image=image, **kwargs)
        self.layers.append(layer)
        logger.debug("Added layer '%s' (%dx%d)", name, image.shape[1], image.shape[0])
        return layer

    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def remove_layer(self, name: str) -> bool:
        """Remove a layer by name."""
        for i, layer in enumerate(self.layers):
            if layer.name == name:
                self.layers.pop(i)
                return True
        return False

    def flatten(self) -> np.ndarray:
        """Flatten all visible layers into a single image."""
        if not self.layers:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        result = self.layers[0].image.copy()
        for layer in self.layers[1:]:
            if not layer.visible:
                continue
            result = self._composite_layer(result, layer)
        return result

    def _composite_layer(self, base: np.ndarray, layer: Layer) -> np.ndarray:
        """Composite a layer onto the base image."""
        result = base.copy()
        img = layer.image
        opacity = layer.opacity

        # Handle offset
        y1 = max(0, layer.offset_y)
        x1 = max(0, layer.offset_x)
        y2 = min(base.shape[0], layer.offset_y + img.shape[0])
        x2 = min(base.shape[1], layer.offset_x + img.shape[1])

        sy1 = y1 - layer.offset_y
        sx1 = x1 - layer.offset_x
        sy2 = sy1 + (y2 - y1)
        sx2 = sx1 + (x2 - x1)

        if y2 <= y1 or x2 <= x1:
            return result

        # Check for alpha channel
        if img.shape[2] == 4:
            alpha = img[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0 * opacity
            fg = img[sy1:sy2, sx1:sx2, :3].astype(np.float32)
            bg = result[y1:y2, x1:x2, :3].astype(np.float32)
            result[y1:y2, x1:x2, :3] = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
        else:
            if opacity < 1.0:
                fg = img[sy1:sy2, sx1:sx2].astype(np.float32)
                bg = result[y1:y2, x1:x2].astype(np.float32)
                result[y1:y2, x1:x2] = (fg * opacity + bg * (1 - opacity)).astype(np.uint8)
            else:
                result[y1:y2, x1:x2] = img[sy1:sy2, sx1:sx2]

        return result


class GimpLayerAdapter:
    """Adapter for working with GIMP layers when running inside GIMP."""

    def __init__(self, gimp_image=None):
        self._gimp_image = gimp_image
        self._gimp_available = self._check_gimp()

    def _check_gimp(self) -> bool:
        try:
            from gimpfu import pdb  # noqa: F401
            return True
        except ImportError:
            return False

    def create_layer(
        self, name: str, image_data: np.ndarray
    ) -> Optional[object]:
        """Create a GIMP layer from numpy array."""
        if not self._gimp_available or self._gimp_image is None:
            return None

        from gimpfu import NORMAL_MODE
        import gimp

        h, w = image_data.shape[:2]
        layer = gimp.Layer(
            self._gimp_image, name, w, h, gimp.RGBA_IMAGE, 100, NORMAL_MODE
        )
        self._gimp_image.add_layer(layer, -1)

        # Set pixel data
        pixel_region = layer.get_pixel_rgn(0, 0, w, h, True, True)
        pixel_region[:, :] = image_data.tobytes()
        layer.flush()
        layer.merge_shadow(True)
        layer.update(0, 0, w, h)

        return layer

    @property
    def is_gimp_available(self) -> bool:
        return self._gimp_available
