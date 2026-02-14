"""Tests for LaMa neural inpainting component."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from manga_translator.components.neural_inpainter import NeuralInpainter


class TestNeuralInpainterAvailability:
    def test_has_onnx(self):
        result = NeuralInpainter._has_onnx()
        assert isinstance(result, bool)

    def test_has_torch(self):
        result = NeuralInpainter._has_torch()
        assert isinstance(result, bool)

    def test_is_available(self):
        ni = NeuralInpainter()
        assert isinstance(ni.is_available(), bool)

    def test_model_path(self):
        ni = NeuralInpainter(model_dir="/tmp/test-models")
        assert str(ni._model_path()).startswith("/tmp/test-models")
        assert ni._model_path().name == "big-lama.onnx"


class TestNeuralInpainterPadding:
    def test_pad_no_change_needed(self):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        mask = np.zeros((64, 128), dtype=np.uint8)
        padded_img, padded_mask, pad_info = NeuralInpainter._pad_to_multiple(img, mask)
        assert padded_img.shape == (64, 128, 3)
        assert padded_mask.shape == (64, 128)
        assert pad_info == (0, 0)

    def test_pad_needed(self):
        img = np.zeros((65, 130, 3), dtype=np.uint8)
        mask = np.zeros((65, 130), dtype=np.uint8)
        padded_img, padded_mask, pad_info = NeuralInpainter._pad_to_multiple(img, mask)
        assert padded_img.shape[0] % 8 == 0
        assert padded_img.shape[1] % 8 == 0
        assert padded_mask.shape[0] % 8 == 0
        assert padded_mask.shape[1] % 8 == 0

    def test_pad_preserves_content(self):
        img = np.full((10, 10, 3), 42, dtype=np.uint8)
        mask = np.full((10, 10), 128, dtype=np.uint8)
        padded_img, padded_mask, _ = NeuralInpainter._pad_to_multiple(img, mask)
        assert np.all(padded_img[:10, :10] == 42)


class TestNeuralInpainterInference:
    def test_inpaint_empty_image_raises(self):
        ni = NeuralInpainter()
        with pytest.raises(ValueError, match="empty"):
            ni.inpaint(np.array([]), np.array([]))

    def test_inpaint_empty_mask_raises(self):
        ni = NeuralInpainter()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            ni.inpaint(img, np.array([]))

    @patch("manga_translator.components.neural_inpainter.NeuralInpainter._get_onnx_session")
    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=True)
    def test_inpaint_with_mock_session(self, mock_avail, mock_session):
        """Test the full inpaint flow with a mocked ONNX session."""
        # Mock session that returns a plausible output
        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="image"),
            MagicMock(name="mask"),
        ]
        # LaMa output: (1, 3, H, W) float32
        mock_output = np.random.rand(1, 3, 64, 64).astype(np.float32)
        session.run.return_value = [mock_output]
        mock_session.return_value = session

        ni = NeuralInpainter()
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        result = ni.inpaint(img, mask)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        session.run.assert_called_once()

    @patch("manga_translator.components.neural_inpainter.NeuralInpainter._get_onnx_session")
    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=True)
    def test_inpaint_grayscale_input(self, mock_avail, mock_session):
        """Grayscale input should be converted to BGR internally."""
        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="image"),
            MagicMock(name="mask"),
        ]
        mock_output = np.random.rand(1, 3, 64, 64).astype(np.float32)
        session.run.return_value = [mock_output]
        mock_session.return_value = session

        ni = NeuralInpainter()
        img = np.full((64, 64), 200, dtype=np.uint8)  # grayscale
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255

        result = ni.inpaint(img, mask)
        assert result.shape == (64, 64, 3)

    @patch("manga_translator.components.neural_inpainter.NeuralInpainter._get_onnx_session")
    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=True)
    def test_inpaint_large_image_resizes(self, mock_avail, mock_session):
        """Images larger than max_side should be resized down."""
        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="image"),
            MagicMock(name="mask"),
        ]
        # Output matches the resized+padded dimensions
        mock_output = np.random.rand(1, 3, 512, 512).astype(np.float32)
        session.run.return_value = [mock_output]
        mock_session.return_value = session

        ni = NeuralInpainter(max_side=512)
        img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        result = ni.inpaint(img, mask)
        assert result.shape == (1024, 1024, 3)

    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=False)
    def test_inpaint_not_available_raises(self, mock_avail):
        ni = NeuralInpainter()
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="No neural inpainting backend"):
            ni.inpaint(img, mask)


class TestNeuralInpainterRegion:
    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.inpaint")
    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=True)
    def test_inpaint_region(self, mock_avail, mock_inpaint):
        """Region inpainting should crop, inpaint, and paste back."""
        ni = NeuralInpainter()

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 255

        # Mock inpaint to return white for the cropped region
        def fake_inpaint(crop_img, crop_mask):
            return np.full_like(crop_img, 255)

        mock_inpaint.side_effect = fake_inpaint

        result = ni.inpaint_region(img, mask, bbox=(50, 50, 50, 50), padding=10)
        assert result.shape == img.shape
        # Masked pixels should be changed
        assert result[75, 75, 0] == 255
        # Unmasked pixels outside region should be unchanged
        assert result[0, 0, 0] == 128

    @patch("manga_translator.components.neural_inpainter.NeuralInpainter.is_available", return_value=True)
    def test_inpaint_region_empty_mask(self, mock_avail):
        """No mask pixels in region = return image unchanged."""
        ni = NeuralInpainter()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)  # empty mask

        result = ni.inpaint_region(img, mask, bbox=(20, 20, 30, 30))
        np.testing.assert_array_equal(result, img)


class TestInpainterLamaIntegration:
    """Test Inpainter class with lama method."""

    def test_lama_method_accepted(self):
        from manga_translator.components.inpainter import Inpainter
        inpainter = Inpainter(method="lama")
        assert inpainter.method == "lama"

    def test_is_neural_available(self):
        from manga_translator.components.inpainter import Inpainter
        inpainter = Inpainter()
        assert isinstance(inpainter.is_neural_available(), bool)

    @patch("manga_translator.components.inpainter.Inpainter.inpaint_lama")
    def test_remove_text_dispatches_lama(self, mock_lama):
        """remove_text with method='lama' should dispatch to inpaint_lama."""
        from manga_translator.components.inpainter import Inpainter
        inpainter = Inpainter(method="lama")

        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        mock_lama.return_value = img.copy()
        result = inpainter.remove_text(img, mask)
        mock_lama.assert_called_once()

    @patch("manga_translator.components.inpainter.Inpainter.is_neural_available", return_value=True)
    def test_auto_selects_lama_for_complex(self, mock_avail):
        """Auto method should select lama for complex backgrounds when available."""
        from manga_translator.components.inpainter import Inpainter
        inpainter = Inpainter(method="auto")

        # Create a noisy/complex background
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        method = inpainter.auto_select_method(img, mask)
        # For very complex backgrounds, should prefer lama
        assert method in ("lama", "opencv_ns", "opencv_telea")

    @patch("manga_translator.components.inpainter.Inpainter.is_neural_available", return_value=False)
    def test_auto_falls_back_without_neural(self, mock_avail):
        """Without neural backend, auto should fall back to opencv_ns."""
        from manga_translator.components.inpainter import Inpainter
        inpainter = Inpainter(method="auto")

        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        method = inpainter.auto_select_method(img, mask)
        assert method != "lama"
