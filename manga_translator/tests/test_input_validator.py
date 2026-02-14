"""Tests for input validation module."""

from unittest import mock

import numpy as np
import pytest

from manga_translator.input_validator import InputValidator, ValidationError


# ---------------------------------------------------------------------------
# TestValidateImagePath
# ---------------------------------------------------------------------------

class TestValidateImagePath:
    """Tests for InputValidator.validate_image_path."""

    def test_valid_path(self, tmp_path):
        """A valid .png file with content should pass validation."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = InputValidator.validate_image_path(str(img))
        assert result == str(img)

    def test_empty_path(self):
        """An empty string should raise ValidationError."""
        with pytest.raises(ValidationError, match="Image path is empty"):
            InputValidator.validate_image_path("")

    def test_nonexistent(self):
        """A path that does not exist should raise ValidationError."""
        with pytest.raises(ValidationError, match="File not found"):
            InputValidator.validate_image_path("/nonexistent/image.png")

    def test_directory(self, tmp_path):
        """A directory path should raise ValidationError."""
        with pytest.raises(ValidationError, match="Not a file"):
            InputValidator.validate_image_path(str(tmp_path))

    def test_unsupported_extension(self, tmp_path):
        """A file with an unsupported extension should raise ValidationError."""
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        with pytest.raises(ValidationError, match="Unsupported format"):
            InputValidator.validate_image_path(str(txt))

    def test_empty_file(self, tmp_path):
        """A zero-byte file should raise ValidationError."""
        img = tmp_path / "empty.png"
        img.write_bytes(b"")
        with pytest.raises(ValidationError, match="File is empty"):
            InputValidator.validate_image_path(str(img))

    def test_large_file(self, tmp_path):
        """A file exceeding MAX_FILE_SIZE should raise ValidationError."""
        img = tmp_path / "huge.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 10)
        fake_size = InputValidator.MAX_FILE_SIZE + 1
        with mock.patch("os.path.getsize", return_value=fake_size):
            with pytest.raises(ValidationError, match="File too large"):
                InputValidator.validate_image_path(str(img))


# ---------------------------------------------------------------------------
# TestValidateImageArray
# ---------------------------------------------------------------------------

class TestValidateImageArray:
    """Tests for InputValidator.validate_image_array."""

    def test_valid_3d_array(self):
        """A 100x100x3 uint8 array should pass validation."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = InputValidator.validate_image_array(img)
        assert result is img

    def test_valid_2d_array(self):
        """A 100x100 grayscale array should pass validation."""
        img = np.zeros((100, 100), dtype=np.uint8)
        result = InputValidator.validate_image_array(img)
        assert result is img

    def test_none(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="Image is None"):
            InputValidator.validate_image_array(None)

    def test_not_ndarray(self):
        """A non-ndarray value should raise ValidationError."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            InputValidator.validate_image_array("not an image")

    def test_empty_array(self):
        """A zero-size array should raise ValidationError."""
        img = np.array([]).reshape(0, 0)
        with pytest.raises(ValidationError, match="Image is empty"):
            InputValidator.validate_image_array(img)

    def test_too_small(self):
        """An image below MIN_DIMENSION should raise ValidationError."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValidationError, match="Image too small"):
            InputValidator.validate_image_array(img)

    def test_too_large(self):
        """An image exceeding MAX_DIMENSION should raise ValidationError (mocked shape)."""
        # Use a normal-sized array but lower MAX_DIMENSION to trigger the check
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch.object(InputValidator, "MAX_DIMENSION", 50):
            with pytest.raises(ValidationError, match="Image too large"):
                InputValidator.validate_image_array(img)


# ---------------------------------------------------------------------------
# TestValidateLanguage
# ---------------------------------------------------------------------------

class TestValidateLanguage:
    """Tests for InputValidator.validate_language."""

    def test_valid_language(self):
        """A valid language code should be returned."""
        assert InputValidator.validate_language("ja") == "ja"

    def test_invalid_language(self):
        """An unknown language code should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unknown language"):
            InputValidator.validate_language("xx")

    def test_case_insensitive(self):
        """Language codes should be normalized to lowercase."""
        assert InputValidator.validate_language("JA") == "ja"


# ---------------------------------------------------------------------------
# TestValidateBatchPaths
# ---------------------------------------------------------------------------

class TestValidateBatchPaths:
    """Tests for InputValidator.validate_batch_paths."""

    def test_mixed_paths(self, tmp_path):
        """A mix of valid and invalid paths should be split correctly."""
        good = tmp_path / "good.png"
        good.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        bad_ext = tmp_path / "bad.txt"
        bad_ext.write_text("hello")

        paths = [str(good), str(bad_ext), "/nonexistent/missing.png"]
        valid, errors = InputValidator.validate_batch_paths(paths)

        assert valid == [str(good)]
        assert len(errors) == 2
        assert any("Unsupported format" in e for e in errors)
        assert any("File not found" in e for e in errors)
