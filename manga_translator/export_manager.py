"""Multi-format export for translated manga pages.

Supports PNG, JPEG, PDF, and CBZ (comic book archive) output formats.
"""

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Options for export operations."""
    quality: int = 95  # JPEG quality (1-100)
    png_compression: int = 6  # PNG compression (0-9)
    pdf_title: str = ""
    pdf_author: str = ""
    cbz_metadata: Optional[Dict[str, Any]] = None


class ExportManager:
    """Exports translated manga pages in multiple formats.

    Supported formats:
    - ``png``: Lossless image format
    - ``jpg`` / ``jpeg``: Lossy image format with configurable quality
    - ``pdf``: Multi-page PDF document
    - ``cbz``: Comic Book ZIP archive (widely used for digital manga)
    """

    SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "pdf", "cbz"}

    def __init__(self, options: Optional[ExportOptions] = None):
        self.options = options or ExportOptions()

    def export(
        self,
        images: List[np.ndarray],
        output_path: str,
        format: Optional[str] = None,
    ) -> str:
        """Export one or more images to the specified format.

        Args:
            images: List of BGR numpy arrays.
            output_path: Destination file path.
            format: Output format. If None, detected from file extension.

        Returns:
            Absolute path to the output file.

        Raises:
            ValueError: If format is unsupported.
        """
        if not images:
            raise ValueError("No images to export")

        if format is None:
            format = self.detect_format(output_path)

        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        output_path = str(Path(output_path).resolve())
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if format == "png":
            return self._export_png(images[0], output_path)
        elif format in ("jpg", "jpeg"):
            return self._export_jpeg(images[0], output_path)
        elif format == "pdf":
            return self._export_pdf(images, output_path)
        elif format == "cbz":
            return self._export_cbz(images, output_path)

        raise ValueError(f"Unhandled format: {format}")

    def export_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        format: str = "png",
    ) -> List[str]:
        """Export multiple image files to a directory.

        Args:
            image_paths: List of source image file paths.
            output_dir: Output directory.
            format: Target format for all images.

        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for path in image_paths:
            name = Path(path).stem
            output_path = os.path.join(output_dir, f"{name}.{format}")

            img = cv2.imread(path)
            if img is None:
                logger.warning("Could not read %s, skipping", path)
                continue

            result = self.export([img], output_path, format=format)
            output_paths.append(result)

        return output_paths

    @staticmethod
    def detect_format(path: str) -> str:
        """Detect format from file extension."""
        ext = Path(path).suffix.lower().lstrip(".")
        if ext in ("jpg", "jpeg"):
            return "jpg"
        return ext

    # ------------------------------------------------------------------
    # Format implementations
    # ------------------------------------------------------------------

    def _export_png(self, image: np.ndarray, path: str) -> str:
        """Export as PNG with configurable compression."""
        params = [cv2.IMWRITE_PNG_COMPRESSION, self.options.png_compression]
        cv2.imwrite(path, image, params)
        logger.debug("Exported PNG: %s", path)
        return path

    def _export_jpeg(self, image: np.ndarray, path: str) -> str:
        """Export as JPEG with configurable quality."""
        params = [cv2.IMWRITE_JPEG_QUALITY, self.options.quality]
        cv2.imwrite(path, image, params)
        logger.debug("Exported JPEG: %s", path)
        return path

    def _export_pdf(self, images: List[np.ndarray], path: str) -> str:
        """Export as multi-page PDF using Pillow."""
        pil_images = []
        for img in images:
            # Convert BGR to RGB for Pillow
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # PDF requires RGB mode
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        if not pil_images:
            raise ValueError("No valid images for PDF export")

        first = pil_images[0]
        rest = pil_images[1:] if len(pil_images) > 1 else []

        first.save(
            path,
            "PDF",
            save_all=True,
            append_images=rest,
            title=self.options.pdf_title or "Manga Translation",
            author=self.options.pdf_author or "Manga Translator",
        )
        logger.debug("Exported PDF (%d pages): %s", len(pil_images), path)
        return path

    def _export_cbz(self, images: List[np.ndarray], path: str) -> str:
        """Export as CBZ (Comic Book ZIP) archive.

        Each page is stored as a PNG inside the ZIP with sequential naming.
        A ComicInfo.xml metadata file is included if metadata is provided.
        """
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(images):
                # Encode image as PNG bytes
                success, buf = cv2.imencode(".png", img)
                if not success:
                    logger.warning("Failed to encode page %d, skipping", i)
                    continue
                page_name = f"page_{i+1:04d}.png"
                zf.writestr(page_name, buf.tobytes())

            # Add metadata if provided
            metadata = self.options.cbz_metadata or {}
            if metadata:
                comic_info = self._build_comic_info(metadata)
                zf.writestr("ComicInfo.xml", comic_info)

        logger.debug("Exported CBZ (%d pages): %s", len(images), path)
        return path

    @staticmethod
    def _build_comic_info(metadata: Dict[str, Any]) -> str:
        """Build a ComicInfo.xml string from metadata dict."""
        lines = ['<?xml version="1.0" encoding="utf-8"?>', "<ComicInfo>"]
        tag_map = {
            "title": "Title",
            "series": "Series",
            "author": "Writer",
            "artist": "Penciller",
            "language": "LanguageISO",
            "page_count": "PageCount",
            "summary": "Summary",
        }
        for key, tag in tag_map.items():
            if key in metadata:
                value = str(metadata[key])
                # Escape XML special characters
                value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  <{tag}>{value}</{tag}>")
        lines.append("</ComicInfo>")
        return "\n".join(lines)
