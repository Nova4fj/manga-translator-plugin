"""Tests for multi-format export manager."""

import os
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

from manga_translator.export_manager import ExportManager, ExportOptions


@pytest.fixture
def sample_image():
    """Create a simple BGR test image."""
    img = np.full((100, 80, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 90), (0, 0, 255), 2)
    return img


@pytest.fixture
def sample_images(sample_image):
    """Create multiple test images."""
    img2 = np.full((100, 80, 3), 150, dtype=np.uint8)
    return [sample_image, img2]


@pytest.fixture
def manager():
    return ExportManager()


class TestExportOptions:
    def test_defaults(self):
        opts = ExportOptions()
        assert opts.quality == 95
        assert opts.png_compression == 6
        assert opts.pdf_title == ""
        assert opts.cbz_metadata is None

    def test_custom(self):
        opts = ExportOptions(quality=80, pdf_title="My Manga")
        assert opts.quality == 80
        assert opts.pdf_title == "My Manga"


class TestFormatDetection:
    def test_png(self):
        assert ExportManager.detect_format("file.png") == "png"

    def test_jpg(self):
        assert ExportManager.detect_format("file.jpg") == "jpg"

    def test_jpeg(self):
        assert ExportManager.detect_format("file.jpeg") == "jpg"

    def test_pdf(self):
        assert ExportManager.detect_format("file.pdf") == "pdf"

    def test_cbz(self):
        assert ExportManager.detect_format("file.cbz") == "cbz"

    def test_case_insensitive(self):
        assert ExportManager.detect_format("file.PNG") == "png"
        assert ExportManager.detect_format("file.JPG") == "jpg"


class TestExportPNG:
    def test_export_png(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.png")
        result = manager.export([sample_image], path, format="png")
        assert os.path.exists(result)
        loaded = cv2.imread(result)
        assert loaded.shape == sample_image.shape

    def test_png_compression(self, sample_image, tmp_path):
        mgr = ExportManager(ExportOptions(png_compression=9))
        path = str(tmp_path / "compressed.png")
        mgr.export([sample_image], path, format="png")
        assert os.path.exists(path)


class TestExportJPEG:
    def test_export_jpg(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.jpg")
        result = manager.export([sample_image], path, format="jpg")
        assert os.path.exists(result)

    def test_jpeg_alias(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.jpeg")
        result = manager.export([sample_image], path, format="jpeg")
        assert os.path.exists(result)

    def test_quality(self, sample_image, tmp_path):
        low = ExportManager(ExportOptions(quality=10))
        high = ExportManager(ExportOptions(quality=95))
        low_path = str(tmp_path / "low.jpg")
        high_path = str(tmp_path / "high.jpg")
        low.export([sample_image], low_path, format="jpg")
        high.export([sample_image], high_path, format="jpg")
        assert os.path.getsize(low_path) < os.path.getsize(high_path)


class TestExportPDF:
    def test_single_page(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.pdf")
        result = manager.export([sample_image], path, format="pdf")
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_multi_page(self, manager, sample_images, tmp_path):
        path = str(tmp_path / "multi.pdf")
        result = manager.export(sample_images, path, format="pdf")
        assert os.path.exists(result)

    def test_pdf_metadata(self, sample_image, tmp_path):
        opts = ExportOptions(pdf_title="Test Manga", pdf_author="Tester")
        mgr = ExportManager(opts)
        path = str(tmp_path / "meta.pdf")
        mgr.export([sample_image], path, format="pdf")
        assert os.path.exists(path)


class TestExportCBZ:
    def test_single_page(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.cbz")
        result = manager.export([sample_image], path, format="cbz")
        assert os.path.exists(result)
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "page_0001.png" in names

    def test_multi_page(self, manager, sample_images, tmp_path):
        path = str(tmp_path / "multi.cbz")
        manager.export(sample_images, path, format="cbz")
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            assert "page_0001.png" in names
            assert "page_0002.png" in names

    def test_cbz_metadata(self, sample_image, tmp_path):
        opts = ExportOptions(cbz_metadata={
            "title": "Test Manga",
            "author": "Author",
            "series": "Series",
            "language": "en",
        })
        mgr = ExportManager(opts)
        path = str(tmp_path / "meta.cbz")
        mgr.export([sample_image], path, format="cbz")
        with zipfile.ZipFile(path, "r") as zf:
            assert "ComicInfo.xml" in zf.namelist()
            xml = zf.read("ComicInfo.xml").decode("utf-8")
            assert "<Title>Test Manga</Title>" in xml
            assert "<Writer>Author</Writer>" in xml

    def test_cbz_no_metadata(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "no_meta.cbz")
        manager.export([sample_image], path, format="cbz")
        with zipfile.ZipFile(path, "r") as zf:
            assert "ComicInfo.xml" not in zf.namelist()


class TestComicInfo:
    def test_xml_escaping(self):
        metadata = {"title": "A & B <C>"}
        xml = ExportManager._build_comic_info(metadata)
        assert "&amp;" in xml
        assert "&lt;" in xml
        assert "&gt;" in xml

    def test_all_fields(self):
        metadata = {
            "title": "T", "series": "S", "author": "A",
            "artist": "R", "language": "en", "page_count": 10,
            "summary": "Sum",
        }
        xml = ExportManager._build_comic_info(metadata)
        assert "<Title>T</Title>" in xml
        assert "<Series>S</Series>" in xml
        assert "<Writer>A</Writer>" in xml
        assert "<Penciller>R</Penciller>" in xml
        assert "<LanguageISO>en</LanguageISO>" in xml
        assert "<PageCount>10</PageCount>" in xml
        assert "<Summary>Sum</Summary>" in xml


class TestExportGeneral:
    def test_auto_detect_format(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "auto.png")
        result = manager.export([sample_image], path)
        assert os.path.exists(result)

    def test_unsupported_format(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "out.bmp")
        with pytest.raises(ValueError, match="Unsupported format"):
            manager.export([sample_image], path, format="bmp")

    def test_empty_images(self, manager, tmp_path):
        path = str(tmp_path / "out.png")
        with pytest.raises(ValueError, match="No images"):
            manager.export([], path)

    def test_creates_directories(self, manager, sample_image, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "out.png")
        result = manager.export([sample_image], path)
        assert os.path.exists(result)


class TestExportBatch:
    def test_batch(self, manager, sample_image, tmp_path):
        # Write source images
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"page_{i}.png"), sample_image)

        input_paths = [str(tmp_path / f"page_{i}.png") for i in range(3)]
        out_dir = str(tmp_path / "output")
        results = manager.export_batch(input_paths, out_dir, format="png")
        assert len(results) == 3
        for r in results:
            assert os.path.exists(r)

    def test_batch_skips_missing(self, manager, tmp_path):
        results = manager.export_batch(
            [str(tmp_path / "missing.png")],
            str(tmp_path / "output"),
        )
        assert len(results) == 0
