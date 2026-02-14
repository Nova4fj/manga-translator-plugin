"""Text typesetting for translated manga text.

Renders translated text into manga speech bubbles using PIL/Pillow for font
loading, text measurement, word wrapping, and drawing.  Supports outline
(stroke) text, automatic font-size selection via binary search, and
cross-platform font discovery.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextLayout:
    """Computed text layout for a bubble."""
    lines: List[str]
    font_name: str
    font_size: int
    line_height: int
    total_height: int
    total_width: int
    x_offset: int  # horizontal offset within bubble bbox
    y_offset: int  # vertical offset within bubble bbox
    text_color: Tuple[int, int, int]


@dataclass
class TypesetResult:
    """Result from typesetting operation."""
    image: np.ndarray       # image with text rendered onto it
    text_mask: np.ndarray   # binary mask of rendered text pixels (255 = text)
    layout: TextLayout


# ---------------------------------------------------------------------------
# Typesetter
# ---------------------------------------------------------------------------

class Typesetter:
    """Renders translated text into manga speech bubbles.

    Parameters
    ----------
    default_font : str
        Preferred font family name (or path).  The typesetter will attempt to
        locate it in the system font directories.
    font_size_ratio : float
        Maximum font size expressed as a fraction of the bubble height.
    min_font_size, max_font_size : int
        Hard limits on the font size (in pixels).
    text_color : tuple
        RGB colour for the rendered text.
    outline_color : tuple
        RGB colour for the optional text outline/stroke.
    outline_width : int
        Stroke width in pixels.  ``0`` disables the outline.
    line_spacing : float
        Multiplier applied to the font size to obtain line height.
    padding_ratio : float
        Fraction of the bubble dimensions reserved as inner padding.
    """

    # Common font file extensions recognised during font search.
    _FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}

    def __init__(
        self,
        default_font: str = "Comic Sans MS",
        font_size_ratio: float = 0.7,
        min_font_size: int = 10,
        max_font_size: int = 72,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        outline_color: Tuple[int, int, int] = (255, 255, 255),
        outline_width: int = 0,
        line_spacing: float = 1.2,
        padding_ratio: float = 0.1,
    ) -> None:
        self.default_font = default_font
        self.font_size_ratio = font_size_ratio
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.text_color = text_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.line_spacing = line_spacing
        self.padding_ratio = padding_ratio

        # Pre-build the list of directories to search for fonts.
        self._font_dirs: List[Path] = self.get_system_font_dirs()

        # Cache resolved font paths so we don't re-scan directories every call.
        self._font_cache: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def typeset_text(
        self,
        image: np.ndarray,
        text: str,
        bbox: Tuple[int, int, int, int],
        bubble_mask: Optional[np.ndarray] = None,
    ) -> TypesetResult:
        """Render translated *text* into a bubble region on *image*.

        Parameters
        ----------
        image : np.ndarray
            Source image as an H x W x 3 (or H x W x 4) uint8 array.
        text : str
            The translated string to render.
        bbox : tuple of int
            ``(x, y, width, height)`` bounding box of the target bubble.
        bubble_mask : np.ndarray, optional
            Binary mask (same size as *image*) where non-zero pixels belong to
            the bubble interior.  Currently reserved for future use (e.g.
            constraining text to the actual bubble shape rather than its
            bounding rectangle).

        Returns
        -------
        TypesetResult
        """
        text = text.strip()
        if not text:
            # Nothing to render -- return the image untouched.
            empty_layout = TextLayout(
                lines=[], font_name=self.default_font, font_size=0,
                line_height=0, total_height=0, total_width=0,
                x_offset=0, y_offset=0, text_color=self.text_color,
            )
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return TypesetResult(image=image.copy(), text_mask=mask, layout=empty_layout)

        bx, by, bw, bh = bbox

        # 1. Compute available space inside the bubble (with padding).
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        available_width = max(bw - 2 * pad_x, 1)
        available_height = max(bh - 2 * pad_y, 1)

        # 2. Resolve the font file.
        font_path = self.find_font(self.default_font)
        if font_path is None:
            logger.warning(
                "Could not locate font '%s'; falling back to Pillow default.",
                self.default_font,
            )

        # 3. Find the optimal font size via binary search.
        optimal_size = self.calculate_optimal_font_size(
            text, available_width, available_height, font_path,
        )

        # 4. Load the font at the chosen size.
        font = self._load_font(font_path, optimal_size)

        # 5. Wrap text and compute layout metrics.
        lines = self.wrap_text(text, font, available_width)
        line_height = int(optimal_size * self.line_spacing)
        total_height = line_height * len(lines)

        # Measure the widest line to determine total_width.
        total_width = 0
        for line in lines:
            line_bbox = font.getbbox(line)
            lw = line_bbox[2] - line_bbox[0]
            if lw > total_width:
                total_width = lw

        # 6. Centre the text block within the padded bubble area.
        x_offset = pad_x + max((available_width - total_width) // 2, 0)
        y_offset = pad_y + max((available_height - total_height) // 2, 0)

        layout = TextLayout(
            lines=lines,
            font_name=self.default_font,
            font_size=optimal_size,
            line_height=line_height,
            total_height=total_height,
            total_width=total_width,
            x_offset=x_offset,
            y_offset=y_offset,
            text_color=self.text_color,
        )

        # 7. Render the text onto the image.
        result_image, text_mask = self.render_text_to_image(
            image, layout, bbox, font,
        )

        return TypesetResult(image=result_image, text_mask=text_mask, layout=layout)

    # ------------------------------------------------------------------
    # Font discovery
    # ------------------------------------------------------------------

    def find_font(self, font_name: str) -> Optional[str]:
        """Locate a font file on disk by family name or filename.

        The method first checks whether *font_name* is already an existing file
        path.  If not, it scans the platform's system font directories for a
        matching filename (case-insensitive, with or without extension).

        Returns the absolute path as a string, or ``None`` if not found.
        """
        if font_name in self._font_cache:
            return self._font_cache[font_name]

        # Direct path?
        candidate = Path(font_name)
        if candidate.is_file():
            resolved = str(candidate.resolve())
            self._font_cache[font_name] = resolved
            return resolved

        # Normalise the search term: lowercase, strip common extensions.
        search = font_name.lower()
        search_no_ext = search
        for ext in self._FONT_EXTENSIONS:
            if search.endswith(ext):
                search_no_ext = search[: -len(ext)]
                break

        # Also accept names with spaces replaced by hyphens/underscores.
        search_variants = {
            search_no_ext,
            search_no_ext.replace(" ", ""),
            search_no_ext.replace(" ", "-"),
            search_no_ext.replace(" ", "_"),
        }

        for font_dir in self._font_dirs:
            if not font_dir.is_dir():
                continue
            for path in font_dir.rglob("*"):
                if path.suffix.lower() not in self._FONT_EXTENSIONS:
                    continue
                stem = path.stem.lower()
                if stem in search_variants:
                    resolved = str(path.resolve())
                    self._font_cache[font_name] = resolved
                    logger.debug("Resolved font '%s' -> %s", font_name, resolved)
                    return resolved

        # Not found.
        self._font_cache[font_name] = None
        return None

    def get_system_font_dirs(self) -> List[Path]:
        """Return platform-specific system font directories."""
        dirs: List[Path] = []
        system = platform.system()

        if system == "Linux":
            dirs.extend([
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local" / "share" / "fonts",
            ])
        elif system == "Darwin":  # macOS
            dirs.extend([
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts",
            ])
        elif system == "Windows":
            windir = Path(
                platform.environ.get("WINDIR", "C:/Windows")  # type: ignore[attr-defined]
                if hasattr(platform, "environ")
                else "C:/Windows"
            )
            # os.environ is the reliable source on Windows.
            import os
            windir = Path(os.environ.get("WINDIR", "C:\\Windows"))
            dirs.append(windir / "Fonts")
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if localappdata:
                dirs.append(Path(localappdata) / "Microsoft" / "Windows" / "Fonts")

        return dirs

    # ------------------------------------------------------------------
    # Font size calculation
    # ------------------------------------------------------------------

    def calculate_optimal_font_size(
        self,
        text: str,
        available_width: int,
        available_height: int,
        font_path: Optional[str],
    ) -> int:
        """Use binary search to find the largest font size that fits.

        The text is word-wrapped at each candidate size; the result must fit
        within both *available_width* and *available_height*.

        Returns the chosen font size clamped to
        ``[self.min_font_size, self.max_font_size]``.
        """
        # Upper bound: the smaller of max_font_size and a size derived from
        # the bubble height via font_size_ratio.
        upper = min(
            self.max_font_size,
            int(available_height * self.font_size_ratio),
        )
        upper = max(upper, self.min_font_size)

        lo = self.min_font_size
        hi = upper
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._load_font(font_path, mid)
            lines = self.wrap_text(text, font, available_width)
            line_height = int(mid * self.line_spacing)
            total_height = line_height * len(lines)

            # Check widest line fits (it should given wrap_text, but be safe).
            widest = 0
            for line in lines:
                lbbox = font.getbbox(line)
                lw = lbbox[2] - lbbox[0]
                if lw > widest:
                    widest = lw

            if total_height <= available_height and widest <= available_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    # ------------------------------------------------------------------
    # Text wrapping
    # ------------------------------------------------------------------

    def wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> List[str]:
        """Word-wrap *text* so each line fits within *max_width* pixels.

        Explicit newlines (``\\n``) in the input are honoured.  Words are
        never split unless a single word is wider than *max_width*, in which
        case it is broken character-by-character.
        """
        if max_width <= 0:
            return [text]

        paragraphs = text.split("\n")
        result_lines: List[str] = []

        for paragraph in paragraphs:
            words = paragraph.split()
            if not words:
                result_lines.append("")
                continue

            current_line = ""

            for word in words:
                # Test whether appending this word exceeds the limit.
                test_line = f"{current_line} {word}".strip() if current_line else word
                test_bbox = font.getbbox(test_line)
                test_width = test_bbox[2] - test_bbox[0]

                if test_width <= max_width:
                    current_line = test_line
                else:
                    # Flush current line (if any).
                    if current_line:
                        result_lines.append(current_line)

                    # Check if the single word itself exceeds max_width.
                    word_bbox = font.getbbox(word)
                    word_width = word_bbox[2] - word_bbox[0]

                    if word_width > max_width:
                        # Character-level break for very long words.
                        current_line = self._break_word(word, font, max_width, result_lines)
                    else:
                        current_line = word

            if current_line:
                result_lines.append(current_line)

        return result_lines if result_lines else [""]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_text_to_image(
        self,
        image: np.ndarray,
        layout: TextLayout,
        bbox: Tuple[int, int, int, int],
        font: ImageFont.FreeTypeFont,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render *layout* onto *image* and return ``(result_image, text_mask)``.

        Text is drawn centred horizontally within the bubble bounding box.
        Each line is centred independently so that short and long lines look
        balanced.

        Parameters
        ----------
        image : np.ndarray
            H x W x C uint8 source image.
        layout : TextLayout
            Pre-computed layout (see :meth:`typeset_text`).
        bbox : tuple
            ``(x, y, width, height)`` of the target bubble.
        font : ImageFont.FreeTypeFont
            The loaded Pillow font at the correct size.

        Returns
        -------
        result_image : np.ndarray
            Copy of the source image with text rendered on it.
        text_mask : np.ndarray
            Single-channel uint8 mask (same H x W as image) with 255 where
            text pixels were drawn.
        """
        bx, by, bw, bh = bbox
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1

        # Convert numpy image to PIL.
        if channels == 4:
            pil_mode = "RGBA"
        elif channels == 3:
            pil_mode = "RGB"
        else:
            pil_mode = "L"

        pil_image = Image.fromarray(image, mode=pil_mode)
        draw = ImageDraw.Draw(pil_image)

        # Separate mask image for tracking text pixels.
        mask_image = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_image)

        text_color = layout.text_color
        y_cursor = by + layout.y_offset

        for line in layout.lines:
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]

            # Centre this line horizontally within the bubble.
            line_x = bx + layout.x_offset + max((layout.total_width - line_width) // 2, 0)
            line_y = y_cursor

            # Optional outline / stroke.
            if self.outline_width > 0:
                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=self.outline_color,
                    stroke_width=self.outline_width,
                    stroke_fill=self.outline_color,
                )
                mask_draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=255,
                    stroke_width=self.outline_width,
                    stroke_fill=255,
                )

            # Main text.
            draw.text((line_x, line_y), line, font=font, fill=text_color)
            mask_draw.text((line_x, line_y), line, font=font, fill=255)

            y_cursor += layout.line_height

        result_image = np.array(pil_image)
        text_mask = np.array(mask_image)

        return result_image, text_mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_font(
        self, font_path: Optional[str], size: int
    ) -> ImageFont.FreeTypeFont:
        """Load a PIL font at the given *size*.

        Falls back to the Pillow built-in default if *font_path* is ``None``
        or cannot be loaded.
        """
        if font_path is not None:
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError) as exc:
                logger.debug(
                    "Failed to load font '%s' at size %d: %s", font_path, size, exc,
                )

        # Attempt Pillow's built-in default TrueType font.
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            pass

        # Last resort: Pillow's bitmap default (very limited but always works).
        return ImageFont.load_default()

    @staticmethod
    def _break_word(
        word: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        output_lines: List[str],
    ) -> str:
        """Break a single *word* that exceeds *max_width* character-by-character.

        Completed sub-lines are appended to *output_lines*.  Returns the
        remaining partial line that has not yet been flushed.
        """
        current = ""
        for ch in word:
            test = current + ch
            bbox = font.getbbox(test)
            tw = bbox[2] - bbox[0]
            if tw > max_width and current:
                output_lines.append(current)
                current = ch
            else:
                current = test
        return current
