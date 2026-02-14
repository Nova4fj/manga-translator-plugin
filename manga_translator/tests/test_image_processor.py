"""Tests for image processing utilities."""

import numpy as np
import pytest
import tempfile
import os

from manga_translator.core.image_processor import (
    load_image,
    save_image,
    to_grayscale,
    resize_for_processing,
    scale_bbox,
    crop_region,
    numpy_to_pil,
    pil_to_numpy,
    estimate_memory_usage,
)


class TestLoadSaveImage:
    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path.png")

    def test_save_and_load_png(self, tmp_path):
        img = np.full((100, 80, 3), 128, dtype=np.uint8)
        path = str(tmp_path / "test.png")
        save_image(img, path)
        loaded = load_image(path)
        assert loaded.shape == (100, 80, 3)

    def test_save_jpg(self, tmp_path):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        path = str(tmp_path / "test.jpg")
        save_image(img, path, quality=80)
        assert os.path.exists(path)


class TestToGrayscale:
    def test_bgr_to_gray(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = to_grayscale(bgr)
        assert len(gray.shape) == 2

    def test_already_gray(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = to_grayscale(gray)
        assert result.shape == (10, 10)


class TestResizeForProcessing:
    def test_small_image_no_resize(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        resized, scale = resize_for_processing(img, max_dimension=4096)
        assert scale == 1.0
        assert resized.shape == img.shape

    def test_large_image_resized(self):
        img = np.zeros((8000, 6000, 3), dtype=np.uint8)
        resized, scale = resize_for_processing(img, max_dimension=4096)
        assert scale < 1.0
        assert max(resized.shape[:2]) <= 4096

    def test_scale_factor_correct(self):
        img = np.zeros((8000, 4000, 3), dtype=np.uint8)
        resized, scale = resize_for_processing(img, max_dimension=4000)
        assert scale == pytest.approx(0.5, abs=0.01)


class TestScaleBbox:
    def test_scale_up(self):
        bbox = (10, 20, 30, 40)
        scaled = scale_bbox(bbox, 0.5)
        assert scaled == (20, 40, 60, 80)

    def test_scale_identity(self):
        bbox = (10, 20, 30, 40)
        scaled = scale_bbox(bbox, 1.0)
        assert scaled == (10, 20, 30, 40)


class TestCropRegion:
    def test_basic_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:40, 30:60] = 255
        cropped = crop_region(img, (30, 20, 30, 20))
        assert cropped.shape == (20, 30, 3)

    def test_crop_with_padding(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_region(img, (30, 20, 30, 20), padding=5)
        assert cropped.shape == (30, 40, 3)

    def test_crop_clamped_to_bounds(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped = crop_region(img, (40, 40, 20, 20), padding=5)
        assert cropped.shape[0] <= 50
        assert cropped.shape[1] <= 50


class TestNumpyPilConversion:
    def test_bgr_roundtrip(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        pil = numpy_to_pil(img)
        back = pil_to_numpy(pil)
        assert back.shape == img.shape

    def test_grayscale_roundtrip(self):
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        pil = numpy_to_pil(img)
        back = pil_to_numpy(pil)
        assert len(back.shape) == 2


class TestEstimateMemory:
    def test_basic(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mem = estimate_memory_usage(img)
        assert mem == img.nbytes * 10
        assert mem > 0
