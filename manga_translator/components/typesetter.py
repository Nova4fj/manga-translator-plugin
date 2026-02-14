"""Text typesetting for translated manga text.

Renders translated text into manga speech bubbles using PIL/Pillow for font
loading, text measurement, word wrapping, and drawing.  Supports outline
(stroke) text, automatic font-size selection via binary search, and
cross-platform font discovery.
"""

import logging
import platform
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK ideograph or kana character."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)    # CJK Extension A
        or (0x3040 <= cp <= 0x309F)    # Hiragana
        or (0x30A0 <= cp <= 0x30FF)    # Katakana
        or (0xAC00 <= cp <= 0xD7AF)    # Hangul Syllables
        or (0x3000 <= cp <= 0x303F)    # CJK Symbols and Punctuation
        or (0xFF00 <= cp <= 0xFFEF)    # Halfwidth/Fullwidth Forms
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
    )


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
    orientation: str = "horizontal"  # "horizontal" or "vertical"


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

    # Absolute minimum font size — used when min_font_size is too large for
    # the bubble.  Going below 6 px produces illegible text.
    _ABSOLUTE_MIN_FONT_SIZE = 6

    # Common font file extensions recognised during font search.
    _FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}

    # Font categories for context-aware font selection.
    FONT_CATEGORIES = {
        "dialogue": [
            "Comic Sans MS", "CC Wild Words", "Manga Temple",
            "Anime Ace", "Komika Text", "BadaBoom BB",
        ],
        "narration": [
            "Times New Roman", "Georgia", "Minion Pro",
            "Noto Serif", "DejaVu Serif",
        ],
        "sfx": [
            "Impact", "Bebas Neue", "Oswald", "Anton",
            "Bangers", "Permanent Marker",
        ],
        "emphasis": [
            "Arial Black", "Helvetica Bold", "Futura Bold",
            "Noto Sans Bold", "DejaVu Sans Bold",
        ],
    }

    # Short words that shouldn't start a new line (avoid orphaned prepositions).
    _NO_BREAK_AFTER = {"a", "an", "the", "I", "in", "on", "at", "to", "of", "is", "no"}

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
        alignment: str = "center",
        font_category: str = "dialogue",
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
        self.alignment = alignment  # "center", "left", "right"
        self.font_category = font_category

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
        orientation: str = "auto",
        source_lang: str = "",
        target_lang: str = "",
        font_override: Optional[str] = None,
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
            the bubble interior.
        orientation : str
            ``"auto"``, ``"horizontal"``, or ``"vertical"``.  When ``"auto"``,
            the typesetter detects orientation from bubble shape and text content.
        source_lang : str
            Source language code for orientation detection (e.g. ``"ja"``).
        target_lang : str
            Target language code (e.g. ``"en"``).  When set to a non-CJK
            language, orientation is forced to horizontal.

        Returns
        -------
        TypesetResult
        """
        # Detect or force orientation
        if orientation == "auto":
            detected = self.detect_orientation(text.strip(), bbox, source_lang, target_lang)
        else:
            detected = orientation

        if detected == "vertical":
            return self.typeset_vertical(image, text, bbox, bubble_mask)
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

        # 0. Shrink bbox to fit inside the actual bubble shape.
        bx, by, bw, bh = self._compute_effective_bbox(bbox, bubble_mask)

        # 1. Compute available space inside the bubble (with padding).
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        available_width = max(bw - 2 * pad_x, 1)
        available_height = max(bh - 2 * pad_y, 1)

        # 2. Resolve the font file.
        if font_override:
            font_path = font_override
        else:
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

    def find_font_by_category(self, category: str) -> Optional[str]:
        """Try to find a font from the given category.

        Iterates through the category's font list and returns the first
        one found on the system.  Returns ``None`` if none are available.
        """
        fonts = self.FONT_CATEGORIES.get(category, [])
        for name in fonts:
            path = self.find_font(name)
            if path is not None:
                return path
        return None

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

        # Verify best actually fits.  If min_font_size was too large for the
        # bubble, try progressively smaller sizes down to the absolute minimum.
        font = self._load_font(font_path, best)
        lines = self.wrap_text(text, font, available_width)
        line_height = int(best * self.line_spacing)
        total_height = line_height * len(lines)
        if total_height > available_height and best > self._ABSOLUTE_MIN_FONT_SIZE:
            for size in range(best - 1, self._ABSOLUTE_MIN_FONT_SIZE - 1, -1):
                font = self._load_font(font_path, size)
                lines = self.wrap_text(text, font, available_width)
                lh = int(size * self.line_spacing)
                th = lh * len(lines)
                if th <= available_height:
                    return size
            return self._ABSOLUTE_MIN_FONT_SIZE

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

        Supports both word-level (Latin) and character-level (CJK) breaking.
        Explicit newlines (``\\n``) in the input are honoured.  Avoids
        breaking after short prepositions/articles for better readability.
        """
        if max_width <= 0:
            return [text]

        paragraphs = text.split("\n")
        result_lines: List[str] = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                result_lines.append("")
                continue

            # Tokenize: split into words/CJK-characters
            tokens = self._tokenize_for_wrap(paragraph)
            if not tokens:
                result_lines.append("")
                continue

            current_line = ""

            for i, token in enumerate(tokens):
                test_line = (current_line + token) if current_line else token.lstrip()
                test_bbox = font.getbbox(test_line)
                test_width = test_bbox[2] - test_bbox[0]

                if test_width <= max_width:
                    current_line = test_line
                else:
                    # Semantic check: avoid orphaning short prepositions
                    if current_line:
                        last_word = current_line.rsplit(None, 1)[-1] if current_line.strip() else ""
                        if last_word.strip() in self._NO_BREAK_AFTER and len(current_line.split()) > 1:
                            # Move the short word to the next line
                            parts = current_line.rsplit(None, 1)
                            if len(parts) == 2:
                                result_lines.append(parts[0])
                                current_line = parts[1] + token
                                continue

                        result_lines.append(current_line)

                    # Check if the single token exceeds max_width
                    token_stripped = token.lstrip()
                    token_bbox = font.getbbox(token_stripped)
                    token_width = token_bbox[2] - token_bbox[0]

                    if token_width > max_width:
                        current_line = self._break_word(
                            token_stripped, font, max_width, result_lines,
                        )
                    else:
                        current_line = token_stripped

            if current_line:
                result_lines.append(current_line)

        return result_lines if result_lines else [""]

    @staticmethod
    def _tokenize_for_wrap(text: str) -> List[str]:
        """Split text into tokens for wrapping: words for Latin, individual
        characters for CJK, preserving whitespace attachment.

        Returns tokens where each Latin word carries its preceding space,
        and CJK characters are individual tokens.
        """
        tokens: List[str] = []
        current_word = ""

        for ch in text:
            if _is_cjk_char(ch):
                # Flush any accumulated Latin word
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                tokens.append(ch)
            elif ch == " ":
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                current_word = " "  # space attaches to next word
            else:
                current_word += ch

        if current_word:
            tokens.append(current_word)

        return tokens

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

            # Align this line within the bubble.
            if self.alignment == "left":
                line_x = bx + layout.x_offset
            elif self.alignment == "right":
                line_x = bx + layout.x_offset + max(layout.total_width - line_width, 0)
            else:  # center (default)
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
    # Vertical text support
    # ------------------------------------------------------------------

    # Characters that should be rotated 90° clockwise in vertical text.
    # Includes Latin letters, digits, and some punctuation.
    _ROTATE_IN_VERTICAL = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789"
    )

    # Small kana and punctuation that have special vertical forms.
    # These are rendered normally (not rotated) in vertical text.
    _VERTICAL_PUNCTUATION_MAP = {
        "—": "︱", "–": "︲",
        "（": "︵", "）": "︶",
        "「": "﹁", "」": "﹂",
        "『": "﹃", "』": "﹄",
        "【": "︻", "】": "︼",
        "〈": "︿", "〉": "﹀",
        "《": "︽", "》": "︾",
        "…": "⋮",
        "〜": "∣",
    }

    @staticmethod
    def detect_orientation(
        text: str,
        bbox: Tuple[int, int, int, int],
        source_lang: str = "",
        target_lang: str = "",
    ) -> str:
        """Detect whether text should be rendered vertically or horizontally.

        Uses bubble aspect ratio and language/content heuristics.

        Args:
            text: The text to render.
            bbox: (x, y, w, h) bounding box of the bubble.
            source_lang: Source language code (e.g. "ja", "zh", "ko").
            target_lang: Target language code (e.g. "en").  Non-CJK target
                languages always get horizontal orientation.

        Returns:
            "vertical" or "horizontal".
        """
        _CJK_LANGS = {"ja", "zh", "ko", "zh-cn", "zh-tw"}

        # If the target language is known and non-CJK, the rendered text is
        # a non-CJK script → always horizontal.
        if target_lang and target_lang not in _CJK_LANGS:
            return "horizontal"

        _, _, w, h = bbox

        # CJK languages in tall/narrow bubbles → vertical
        is_cjk_lang = source_lang in _CJK_LANGS

        # Count CJK characters in text
        cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
        cjk_ratio = cjk_count / max(len(text), 1)

        aspect_ratio = h / max(w, 1)

        # Strong signal: tall bubble + CJK content
        if aspect_ratio > 1.5 and (is_cjk_lang or cjk_ratio > 0.5):
            return "vertical"

        # Moderate signal: very tall bubble + some CJK
        if aspect_ratio > 2.0 and cjk_ratio > 0.2:
            return "vertical"

        return "horizontal"

    def typeset_vertical(
        self,
        image: np.ndarray,
        text: str,
        bbox: Tuple[int, int, int, int],
        bubble_mask: Optional[np.ndarray] = None,
    ) -> TypesetResult:
        """Render text vertically (top-to-bottom, right-to-left columns).

        This is the standard layout for Japanese manga speech bubbles.
        """
        text = text.strip()
        if not text:
            empty_layout = TextLayout(
                lines=[], font_name=self.default_font, font_size=0,
                line_height=0, total_height=0, total_width=0,
                x_offset=0, y_offset=0, text_color=self.text_color,
                orientation="vertical",
            )
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return TypesetResult(image=image.copy(), text_mask=mask, layout=empty_layout)

        # Shrink bbox to fit inside the actual bubble shape.
        bx, by, bw, bh = self._compute_effective_bbox(bbox, bubble_mask)

        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        available_width = max(bw - 2 * pad_x, 1)
        available_height = max(bh - 2 * pad_y, 1)

        font_path = self.find_font(self.default_font)

        # Find optimal font size for vertical layout
        optimal_size = self._calc_vertical_font_size(
            text, available_width, available_height, font_path,
        )

        font = self._load_font(font_path, optimal_size)

        # Break text into vertical columns
        columns = self._wrap_vertical(text, font, available_height, optimal_size)
        col_width = int(optimal_size * self.line_spacing)
        total_width = col_width * len(columns)
        total_height = available_height

        # Right-to-left: first column starts at the right edge
        x_start = pad_x + max((available_width - total_width) // 2, 0) + total_width - col_width
        y_start = pad_y

        layout = TextLayout(
            lines=columns,
            font_name=self.default_font,
            font_size=optimal_size,
            line_height=col_width,
            total_height=total_height,
            total_width=total_width,
            x_offset=x_start,
            y_offset=y_start,
            text_color=self.text_color,
            orientation="vertical",
        )

        result_image, text_mask = self._render_vertical(
            image, layout, bbox, font, columns,
        )

        return TypesetResult(image=result_image, text_mask=text_mask, layout=layout)

    def _calc_vertical_font_size(
        self,
        text: str,
        available_width: int,
        available_height: int,
        font_path: Optional[str],
    ) -> int:
        """Binary search for the largest font size for vertical layout."""
        upper = min(
            self.max_font_size,
            int(available_width * self.font_size_ratio),
        )
        upper = max(upper, self.min_font_size)

        lo = self.min_font_size
        hi = upper
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._load_font(font_path, mid)
            columns = self._wrap_vertical(text, font, available_height, mid)
            col_width = int(mid * self.line_spacing)
            total_width = col_width * len(columns)

            if total_width <= available_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    def _wrap_vertical(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_height: int,
        font_size: int,
    ) -> List[str]:
        """Break text into vertical columns that fit within max_height.

        Each column is a string of characters rendered top-to-bottom.
        Characters are placed one per cell with spacing = font_size * line_spacing.
        """
        char_height = int(font_size * self.line_spacing)
        max_chars_per_col = max(max_height // char_height, 1)

        # Split on newlines first, then break into columns
        paragraphs = text.split("\n")
        columns: List[str] = []
        current_col = ""

        for para in paragraphs:
            for ch in para:
                if len(current_col) >= max_chars_per_col:
                    columns.append(current_col)
                    current_col = ""
                current_col += ch

            # Paragraph break → start new column
            if current_col:
                columns.append(current_col)
                current_col = ""

        if current_col:
            columns.append(current_col)

        return columns if columns else [""]

    def _render_vertical(
        self,
        image: np.ndarray,
        layout: TextLayout,
        bbox: Tuple[int, int, int, int],
        font: ImageFont.FreeTypeFont,
        columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render vertical columns onto the image.

        Columns are drawn right-to-left. Each character is drawn
        individually, top-to-bottom within its column.
        """
        bx, by, bw, bh = bbox
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1

        if channels == 4:
            pil_mode = "RGBA"
        elif channels == 3:
            pil_mode = "RGB"
        else:
            pil_mode = "L"

        pil_image = Image.fromarray(image, mode=pil_mode)
        draw = ImageDraw.Draw(pil_image)
        mask_image = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_image)

        col_width = layout.line_height  # reused as column width
        char_height = int(layout.font_size * self.line_spacing)

        for col_idx, col_text in enumerate(columns):
            # Right-to-left: column 0 is rightmost
            col_x = bx + layout.x_offset - col_idx * col_width

            for char_idx, ch in enumerate(col_text):
                char_y = by + layout.y_offset + char_idx * char_height

                # Map punctuation to vertical forms
                display_ch = self._VERTICAL_PUNCTUATION_MAP.get(ch, ch)

                # For Latin characters, render normally (small in vertical context)
                if ch in self._ROTATE_IN_VERTICAL:
                    # Centre the character in the column
                    ch_bbox = font.getbbox(display_ch)
                    ch_w = ch_bbox[2] - ch_bbox[0]
                    ch_x = col_x + max((col_width - ch_w) // 2, 0)
                else:
                    ch_bbox = font.getbbox(display_ch)
                    ch_w = ch_bbox[2] - ch_bbox[0]
                    ch_x = col_x + max((col_width - ch_w) // 2, 0)

                if self.outline_width > 0:
                    draw.text(
                        (ch_x, char_y), display_ch, font=font,
                        fill=self.outline_color,
                        stroke_width=self.outline_width,
                        stroke_fill=self.outline_color,
                    )
                    mask_draw.text(
                        (ch_x, char_y), display_ch, font=font,
                        fill=255, stroke_width=self.outline_width,
                        stroke_fill=255,
                    )

                draw.text(
                    (ch_x, char_y), display_ch, font=font,
                    fill=layout.text_color,
                )
                mask_draw.text(
                    (ch_x, char_y), display_ch, font=font, fill=255,
                )

        result_image = np.array(pil_image)
        text_mask = np.array(mask_image)
        return result_image, text_mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_inscribed_rect(
        mask_crop: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Find the largest axis-aligned rectangle inside a binary mask.

        Uses the histogram-based maximal-rectangle algorithm (O(rows*cols)).

        Parameters
        ----------
        mask_crop : np.ndarray
            Binary uint8 mask where 255 means "inside bubble".

        Returns
        -------
        (x, y, w, h) in *local* (mask_crop) coordinates.
        """
        if mask_crop.size == 0:
            return (0, 0, mask_crop.shape[1] if mask_crop.ndim >= 2 else 0,
                    mask_crop.shape[0] if mask_crop.ndim >= 1 else 0)

        rows, cols = mask_crop.shape[:2]
        # Build a height histogram: height[r][c] = number of consecutive
        # "inside" rows ending at row r in column c.
        binary = (mask_crop > 0).astype(np.int32)
        height = np.zeros((rows, cols), dtype=np.int32)
        height[0] = binary[0]
        for r in range(1, rows):
            height[r] = np.where(binary[r] == 1, height[r - 1] + 1, 0)

        best_area = 0
        best_rect = (0, 0, cols, rows)  # fallback

        for r in range(rows):
            # Largest rectangle in histogram for this row.
            h_row = height[r]
            stack: list[int] = []  # stack of column indices
            left = [0] * cols

            # Left boundary
            for c in range(cols):
                while stack and h_row[stack[-1]] >= h_row[c]:
                    stack.pop()
                left[c] = stack[-1] + 1 if stack else 0
                stack.append(c)

            # Right boundary
            stack = []
            right = [0] * cols
            for c in range(cols - 1, -1, -1):
                while stack and h_row[stack[-1]] >= h_row[c]:
                    stack.pop()
                right[c] = stack[-1] - 1 if stack else cols - 1
                stack.append(c)

            for c in range(cols):
                w = right[c] - left[c] + 1
                h = int(h_row[c])
                area = w * h
                if area > best_area:
                    best_area = area
                    best_rect = (left[c], r - h + 1, w, h)

        return best_rect

    def _compute_effective_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        bubble_mask: Optional[np.ndarray],
    ) -> Tuple[int, int, int, int]:
        """Shrink *bbox* to the largest rectangle fitting inside *bubble_mask*.

        If *bubble_mask* is ``None``, returns the original *bbox* unchanged.
        """
        if bubble_mask is None:
            return bbox

        bx, by, bw, bh = bbox
        img_h, img_w = bubble_mask.shape[:2]

        # Clamp bbox to image bounds.
        x1 = max(bx, 0)
        y1 = max(by, 0)
        x2 = min(bx + bw, img_w)
        y2 = min(by + bh, img_h)
        if x2 <= x1 or y2 <= y1:
            return bbox

        crop = bubble_mask[y1:y2, x1:x2]
        if crop.size == 0 or not np.any(crop):
            return bbox

        lx, ly, lw, lh = self._largest_inscribed_rect(crop)
        return (x1 + lx, y1 + ly, lw, lh)

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
        """Break a single *word* that exceeds *max_width*.

        For Latin text, inserts hyphens at break points (e.g. "Speak-" / "ing")
        instead of splitting mid-character.  CJK text is still split per
        character.

        Completed sub-lines are appended to *output_lines*.  Returns the
        remaining partial line that has not yet been flushed.
        """
        # Determine whether this word is Latin (use hyphenation) or CJK.
        is_latin = not any(_is_cjk_char(ch) for ch in word)

        if is_latin:
            # Measure the width of a hyphen so we can reserve space.
            hyphen_w = font.getbbox("-")[2] - font.getbbox("-")[0]

            current = ""
            for ch in word:
                test = current + ch
                test_bbox = font.getbbox(test)
                tw = test_bbox[2] - test_bbox[0]
                # Check if adding this char (plus a hyphen) exceeds max_width.
                if tw + hyphen_w > max_width and len(current) > 1:
                    output_lines.append(current + "-")
                    current = ch
                elif tw > max_width and current:
                    # Fallback: even without hyphen space it doesn't fit.
                    output_lines.append(current)
                    current = ch
                else:
                    current = test
            return current
        else:
            # CJK: break per character (no hyphen needed).
            current = ""
            for ch in word:
                test = current + ch
                test_bbox = font.getbbox(test)
                tw = test_bbox[2] - test_bbox[0]
                if tw > max_width and current:
                    output_lines.append(current)
                    current = ch
                else:
                    current = test
            return current
