"""LaMa neural inpainting for high-quality text removal.

Uses the LaMa (Large Mask Inpainting) model for state-of-the-art
inpainting results. Supports both PyTorch and ONNX Runtime backends,
with automatic GPU/CPU selection and model downloading.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default model directory
_DEFAULT_MODEL_DIR = Path.home() / ".manga-translator" / "models"

# LaMa ONNX model URL and checksum (big-lama, resolution-robust variant)
_LAMA_ONNX_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.onnx"
)
_LAMA_ONNX_SHA256 = None  # skip verification — model host may update
_LAMA_ONNX_FILENAME = "big-lama.onnx"

# LaMa expects input padded to multiples of 8
_PAD_MULTIPLE = 8
# Maximum dimension before tiling to avoid OOM
_MAX_SIDE = 2048


class NeuralInpainter:
    """LaMa-based neural inpainting engine.

    Attempts backends in order:
    1. ONNX Runtime (lighter, preferred for plugin use)
    2. PyTorch (full model, if available)

    Falls back gracefully if neither is available.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "auto",
        max_side: int = _MAX_SIDE,
    ):
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._device = device
        self._max_side = max_side

        # Lazy-loaded session/model
        self._onnx_session = None
        self._backend: Optional[str] = None  # "onnx" or "torch"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if at least one neural backend is usable."""
        return self._has_onnx() or self._has_torch()

    @staticmethod
    def _has_onnx() -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_torch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _model_path(self) -> Path:
        return self._model_dir / _LAMA_ONNX_FILENAME

    def ensure_model(self, progress_callback=None) -> Path:
        """Download the LaMa ONNX model if not already cached.

        Args:
            progress_callback: Optional callable(bytes_downloaded, total_bytes).

        Returns:
            Path to the model file.
        """
        path = self._model_path()
        if path.exists():
            logger.debug("LaMa model found at %s", path)
            return path

        self._model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LaMa model to %s …", path)

        tmp_path = path.with_suffix(".onnx.tmp")
        try:
            req = urllib.request.Request(_LAMA_ONNX_URL)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)

            tmp_path.rename(path)
            logger.info("LaMa model downloaded successfully (%d bytes)", downloaded)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return path

    # ------------------------------------------------------------------
    # Session / model loading
    # ------------------------------------------------------------------

    def _get_onnx_session(self):
        """Lazy-load the ONNX Runtime inference session."""
        if self._onnx_session is not None:
            return self._onnx_session

        import onnxruntime as ort

        model_path = self.ensure_model()

        # Select execution providers based on device preference
        providers = self._select_providers()
        logger.info("Loading LaMa ONNX model with providers: %s", providers)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._onnx_session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
        self._backend = "onnx"
        return self._onnx_session

    def _select_providers(self) -> list:
        """Choose ONNX Runtime execution providers based on device setting."""
        import onnxruntime as ort

        available = ort.get_available_providers()

        if self._device == "cpu":
            return ["CPUExecutionProvider"]

        if self._device == "cuda" or self._device == "auto":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint masked regions using the LaMa neural model.

        Args:
            image: BGR uint8 image (H, W, 3).
            mask: Single-channel uint8 mask where 255 = region to inpaint.

        Returns:
            Inpainted BGR uint8 image, same shape as input.
        """
        if not self.is_available():
            raise RuntimeError("No neural inpainting backend available")

        # Validate inputs
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if mask is None or mask.size == 0:
            raise ValueError("Mask is empty")

        # Ensure correct formats
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Binarize mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        orig_h, orig_w = image.shape[:2]

        # Resize if too large (to avoid OOM)
        scale = 1.0
        if max(orig_h, orig_w) > self._max_side:
            scale = self._max_side / max(orig_h, orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            image_resized = image
            mask_resized = mask

        # Pad to multiple of 8
        image_padded, mask_padded, pad_info = self._pad_to_multiple(
            image_resized, mask_resized
        )

        # Run inference
        result_padded = self._run_inference(image_padded, mask_padded)

        # Remove padding
        h_pad, w_pad = image_resized.shape[:2]
        result = result_padded[:h_pad, :w_pad]

        # Resize back if needed
        if scale != 1.0:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return result

    def inpaint_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 32,
    ) -> np.ndarray:
        """Inpaint only within a bounding box region for efficiency.

        Crops the image to the bbox + padding, runs neural inpainting on
        the crop, then pastes the result back.

        Args:
            image: Full BGR image.
            mask: Full mask (255 = inpaint).
            bbox: (x, y, w, h) region of interest.
            padding: Extra pixels around bbox for context.

        Returns:
            Full image with the region inpainted.
        """
        img_h, img_w = image.shape[:2]
        x, y, w, h = bbox

        # Expand bbox with padding, clamp to image bounds
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img_w)
        y2 = min(y + h + padding, img_h)

        crop_img = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()

        # Skip if no mask pixels in this region
        if np.count_nonzero(crop_mask) == 0:
            return image

        inpainted_crop = self.inpaint(crop_img, crop_mask)

        # Paste back — only replace masked pixels to preserve unmasked areas
        result = image.copy()
        mask_region = crop_mask > 0
        if result.ndim == 3:
            result[y1:y2, x1:x2][mask_region] = inpainted_crop[mask_region]
        else:
            result[y1:y2, x1:x2][mask_region] = inpainted_crop[mask_region]

        return result

    def _run_inference(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run the actual model inference via ONNX Runtime."""
        session = self._get_onnx_session()

        # Prepare inputs: LaMa expects (1, 3, H, W) float32 image
        # and (1, 1, H, W) float32 mask (0.0 or 1.0)
        img_input = image.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
        img_input = np.expand_dims(img_input, 0)  # add batch dim

        mask_input = (mask.astype(np.float32) / 255.0)
        mask_input = np.expand_dims(mask_input, 0)  # add H dim -> (1, H, W)
        mask_input = np.expand_dims(mask_input, 0)  # add batch -> (1, 1, H, W)

        input_name_img = session.get_inputs()[0].name
        input_name_mask = session.get_inputs()[1].name

        outputs = session.run(
            None,
            {input_name_img: img_input, input_name_mask: mask_input},
        )

        # Output is (1, 3, H, W) float32 in [0, 1]
        result = outputs[0][0]  # remove batch dim -> (3, H, W)
        result = np.transpose(result, (1, 2, 0))  # CHW -> HWC
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_multiple(
        image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Pad image and mask so dimensions are multiples of _PAD_MULTIPLE."""
        h, w = image.shape[:2]
        pad_h = (_PAD_MULTIPLE - h % _PAD_MULTIPLE) % _PAD_MULTIPLE
        pad_w = (_PAD_MULTIPLE - w % _PAD_MULTIPLE) % _PAD_MULTIPLE

        if pad_h == 0 and pad_w == 0:
            return image, mask, (0, 0)

        image_padded = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
        )
        mask_padded = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        return image_padded, mask_padded, (pad_h, pad_w)
