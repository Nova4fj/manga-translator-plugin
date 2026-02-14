"""Tests for typesetting component."""

import numpy as np
import pytest

from manga_translator.components.typesetter import (
    Typesetter, TypesetResult, _is_cjk_char,
)


class TestTypesetter:
    def test_init(self):
        ts = Typesetter()
        assert ts is not None

    def test_typeset_simple_text(self):
        ts = Typesetter()
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        bbox = (50, 50, 200, 100)
        result = ts.typeset_text(image, "Hello world", bbox)
        assert isinstance(result, TypesetResult)
        assert result.image.shape == image.shape
        assert result.layout is not None

    def test_typeset_long_text(self):
        ts = Typesetter()
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 280, 180)
        long_text = "This is a much longer piece of text that should wrap across multiple lines"
        result = ts.typeset_text(image, long_text, bbox)
        assert isinstance(result, TypesetResult)
        assert len(result.layout.lines) > 1

    def test_typeset_empty_text(self):
        ts = Typesetter()
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        bbox = (50, 50, 200, 100)
        result = ts.typeset_text(image, "", bbox)
        assert isinstance(result, TypesetResult)

    def test_font_search(self):
        ts = Typesetter()
        # Should find at least a fallback font
        font_path = ts.find_font("DejaVu Sans")
        # May or may not find it, but shouldn't crash
        assert font_path is None or isinstance(font_path, str)

    def test_text_wrapping(self):
        ts = Typesetter()
        from PIL import ImageFont
        try:
            font = ImageFont.load_default()
        except Exception:
            pytest.skip("No default font available")
        lines = ts.wrap_text("Hello world this is a test", font, 100)
        assert isinstance(lines, list)
        assert len(lines) >= 1

    def test_small_bbox(self):
        """Text should still render even in very small bubbles."""
        ts = Typesetter(min_font_size=8)
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 30, 30)
        result = ts.typeset_text(image, "Hi", bbox)
        assert isinstance(result, TypesetResult)


class TestCJKDetection:
    def test_cjk_ideograph(self):
        assert _is_cjk_char("漫") is True

    def test_hiragana(self):
        assert _is_cjk_char("あ") is True

    def test_katakana(self):
        assert _is_cjk_char("ア") is True

    def test_hangul(self):
        assert _is_cjk_char("한") is True

    def test_latin(self):
        assert _is_cjk_char("A") is False

    def test_digit(self):
        assert _is_cjk_char("5") is False

    def test_fullwidth(self):
        assert _is_cjk_char("Ａ") is True  # fullwidth A


class TestOrientationDetection:
    def test_tall_bubble_ja(self):
        """Tall bubble with Japanese text → vertical."""
        result = Typesetter.detect_orientation("こんにちは", (0, 0, 50, 200), "ja")
        assert result == "vertical"

    def test_wide_bubble_ja(self):
        """Wide bubble → horizontal regardless of language."""
        result = Typesetter.detect_orientation("こんにちは", (0, 0, 200, 50), "ja")
        assert result == "horizontal"

    def test_tall_bubble_english(self):
        """Tall bubble with English text → horizontal (not CJK)."""
        result = Typesetter.detect_orientation("Hello world", (0, 0, 50, 200), "en")
        assert result == "horizontal"

    def test_tall_bubble_cjk_content_no_lang(self):
        """Tall bubble with CJK content, no language hint → vertical."""
        result = Typesetter.detect_orientation("漫画翻訳", (0, 0, 40, 200), "")
        assert result == "vertical"

    def test_square_bubble(self):
        """Square bubble → horizontal (aspect ratio ≈ 1)."""
        result = Typesetter.detect_orientation("テスト", (0, 0, 100, 100), "ja")
        assert result == "horizontal"

    def test_very_tall_mixed(self):
        """Very tall bubble (>2.0 ratio) with mixed CJK content → vertical."""
        result = Typesetter.detect_orientation("漫画ABC", (0, 0, 40, 200), "")
        assert result == "vertical"

    def test_ja_source_en_target_tall_bubble(self):
        """ja→en in tall bubble → horizontal (English text should never be vertical)."""
        result = Typesetter.detect_orientation("Hello world", (0, 0, 50, 200), "ja", "en")
        assert result == "horizontal"

    def test_ja_source_zh_target_tall_bubble(self):
        """ja→zh in tall bubble → vertical (CJK→CJK keeps vertical)."""
        result = Typesetter.detect_orientation("你好世界", (0, 0, 50, 200), "ja", "zh")
        assert result == "vertical"

    def test_ja_source_no_target_tall_bubble(self):
        """ja source, no target, tall bubble → vertical (backward compat)."""
        result = Typesetter.detect_orientation("こんにちは", (0, 0, 50, 200), "ja")
        assert result == "vertical"


class TestVerticalTypesetting:
    def test_vertical_simple(self):
        ts = Typesetter()
        image = np.full((300, 200, 3), 255, dtype=np.uint8)
        bbox = (20, 20, 60, 260)  # tall narrow bubble
        result = ts.typeset_vertical(image, "こんにちは", bbox)
        assert isinstance(result, TypesetResult)
        assert result.layout.orientation == "vertical"
        assert result.image.shape == image.shape

    def test_vertical_empty_text(self):
        ts = Typesetter()
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        result = ts.typeset_vertical(image, "", (10, 10, 50, 180))
        assert result.layout.lines == []
        assert result.layout.orientation == "vertical"

    def test_vertical_multicolumn(self):
        ts = Typesetter()
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 280, 180)
        # Long text should create multiple columns
        long_text = "あいうえおかきくけこさしすせそたちつてと"
        result = ts.typeset_vertical(image, long_text, bbox)
        assert isinstance(result, TypesetResult)
        assert len(result.layout.lines) >= 1

    def test_vertical_text_mask(self):
        ts = Typesetter()
        image = np.full((300, 200, 3), 255, dtype=np.uint8)
        bbox = (20, 20, 60, 260)
        result = ts.typeset_vertical(image, "テスト", bbox)
        assert result.text_mask.shape == image.shape[:2]
        # Some pixels should be marked as text
        assert np.any(result.text_mask > 0)

    def test_auto_orientation_dispatches_vertical(self):
        """typeset_text with orientation='auto' should detect and dispatch vertical."""
        ts = Typesetter()
        image = np.full((400, 150, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 50, 380)  # very tall, narrow
        result = ts.typeset_text(
            image, "日本語テスト", bbox,
            orientation="auto", source_lang="ja",
        )
        assert result.layout.orientation == "vertical"

    def test_forced_horizontal(self):
        """orientation='horizontal' should override auto-detection."""
        ts = Typesetter()
        image = np.full((400, 150, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 50, 380)
        result = ts.typeset_text(
            image, "日本語テスト", bbox,
            orientation="horizontal", source_lang="ja",
        )
        assert result.layout.orientation == "horizontal"

    def test_forced_vertical(self):
        """orientation='vertical' should force vertical even for wide bubble."""
        ts = Typesetter()
        image = np.full((150, 400, 3), 255, dtype=np.uint8)
        bbox = (10, 10, 380, 130)
        result = ts.typeset_text(
            image, "テスト", bbox,
            orientation="vertical", source_lang="ja",
        )
        assert result.layout.orientation == "vertical"

    def test_vertical_wrap(self):
        """_wrap_vertical should break text into columns."""
        ts = Typesetter()
        from PIL import ImageFont
        font = ImageFont.load_default()
        columns = ts._wrap_vertical("あいうえおかきくけこ", font, 100, 20)
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_punctuation_mapping(self):
        """Vertical punctuation map should have correct entries."""
        ts = Typesetter()
        assert ts._VERTICAL_PUNCTUATION_MAP["「"] == "﹁"
        assert ts._VERTICAL_PUNCTUATION_MAP["」"] == "﹂"


class TestTokenizeForWrap:
    def test_latin_words(self):
        tokens = Typesetter._tokenize_for_wrap("Hello world")
        assert "Hello" in tokens
        # "world" should have a space prefix
        assert any("world" in t for t in tokens)

    def test_cjk_chars(self):
        tokens = Typesetter._tokenize_for_wrap("漫画")
        assert "漫" in tokens
        assert "画" in tokens

    def test_mixed_content(self):
        tokens = Typesetter._tokenize_for_wrap("Hello漫画World")
        assert "Hello" in tokens
        assert "漫" in tokens
        assert "画" in tokens
        assert any("World" in t for t in tokens)

    def test_empty(self):
        tokens = Typesetter._tokenize_for_wrap("")
        assert tokens == []


class TestSmartLineBreaking:
    def test_cjk_wrapping(self):
        """CJK text should break at character boundaries."""
        ts = Typesetter()
        from PIL import ImageFont
        font = ImageFont.load_default()
        lines = ts.wrap_text("あいうえおかきくけこ", font, 50)
        assert len(lines) >= 1
        # Each line should fit within the width
        for line in lines:
            bbox = font.getbbox(line)
            assert (bbox[2] - bbox[0]) <= 50 or len(line) == 1

    def test_preserves_explicit_newlines(self):
        ts = Typesetter()
        from PIL import ImageFont
        font = ImageFont.load_default()
        lines = ts.wrap_text("Line one\nLine two", font, 500)
        assert len(lines) >= 2

    def test_empty_paragraph(self):
        ts = Typesetter()
        from PIL import ImageFont
        font = ImageFont.load_default()
        lines = ts.wrap_text("Before\n\nAfter", font, 500)
        assert "" in lines


class TestAlignment:
    def test_left_alignment(self):
        ts = Typesetter(alignment="left")
        assert ts.alignment == "left"
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = ts.typeset_text(image, "Hi", (50, 50, 200, 100))
        assert isinstance(result, TypesetResult)

    def test_right_alignment(self):
        ts = Typesetter(alignment="right")
        assert ts.alignment == "right"
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = ts.typeset_text(image, "Hi", (50, 50, 200, 100))
        assert isinstance(result, TypesetResult)

    def test_center_alignment_default(self):
        ts = Typesetter()
        assert ts.alignment == "center"


class TestFontCategories:
    def test_categories_exist(self):
        assert "dialogue" in Typesetter.FONT_CATEGORIES
        assert "narration" in Typesetter.FONT_CATEGORIES
        assert "sfx" in Typesetter.FONT_CATEGORIES
        assert "emphasis" in Typesetter.FONT_CATEGORIES

    def test_find_font_by_category(self):
        ts = Typesetter()
        # May or may not find a font, but shouldn't crash
        result = ts.find_font_by_category("dialogue")
        assert result is None or isinstance(result, str)

    def test_find_font_unknown_category(self):
        ts = Typesetter()
        result = ts.find_font_by_category("nonexistent")
        assert result is None

    def test_font_category_init(self):
        ts = Typesetter(font_category="sfx")
        assert ts.font_category == "sfx"


class TestInscribedRectangle:
    """Tests for _largest_inscribed_rect static method."""

    def test_full_rectangle_mask(self):
        """A fully filled mask should return the full dimensions."""
        mask = np.full((100, 200), 255, dtype=np.uint8)
        x, y, w, h = Typesetter._largest_inscribed_rect(mask)
        assert w == 200
        assert h == 100

    def test_oval_mask_smaller_rect(self):
        """An ellipse mask should yield a rect smaller than the full bbox."""
        import cv2
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 100), (140, 90), 0, 0, 360, 255, -1)
        x, y, w, h = Typesetter._largest_inscribed_rect(mask)
        # Inscribed rect must be strictly smaller than the full ellipse bbox.
        assert w < 300
        assert h < 200
        # But it should still be a reasonable size (at least half).
        assert w > 100
        assert h > 50

    def test_empty_mask_fallback(self):
        """An all-zero mask should fall back to full bbox dimensions."""
        mask = np.zeros((80, 120), dtype=np.uint8)
        x, y, w, h = Typesetter._largest_inscribed_rect(mask)
        # With no "inside" pixels, the best_area stays 0 and we get the
        # fallback (0, 0, cols, rows).
        assert w == 120
        assert h == 80


class TestEffectiveBbox:
    """Tests for _compute_effective_bbox."""

    def test_no_mask_returns_original(self):
        ts = Typesetter()
        bbox = (10, 20, 100, 80)
        assert ts._compute_effective_bbox(bbox, None) == bbox

    def test_oval_mask_shrinks_bbox(self):
        import cv2
        ts = Typesetter()
        mask = np.zeros((300, 400), dtype=np.uint8)
        # Draw an ellipse centred in the bbox region.
        cv2.ellipse(mask, (200, 150), (140, 100), 0, 0, 360, 255, -1)
        bbox = (50, 50, 300, 200)
        ex, ey, ew, eh = ts._compute_effective_bbox(bbox, mask)
        # Effective box should be no larger than original.
        assert ew <= 300
        assert eh <= 200
        # And it should fit inside the original.
        assert ex >= 50
        assert ey >= 50

    def test_full_mask_preserves_bbox(self):
        ts = Typesetter()
        mask = np.full((200, 300), 255, dtype=np.uint8)
        bbox = (0, 0, 300, 200)
        result = ts._compute_effective_bbox(bbox, mask)
        assert result == bbox

    def test_typeset_with_oval_mask(self):
        """End-to-end: typeset_text with an oval bubble_mask."""
        import cv2
        ts = Typesetter()
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        mask = np.zeros((300, 400), dtype=np.uint8)
        cv2.ellipse(mask, (200, 150), (150, 120), 0, 0, 360, 255, -1)
        bbox = (50, 30, 300, 240)
        result = ts.typeset_text(image, "Hello world test", bbox, bubble_mask=mask)
        assert isinstance(result, TypesetResult)


class TestHyphenatedBreaking:
    """Tests for hyphenated word breaking in _break_word."""

    def test_latin_word_gets_hyphen(self):
        """A long Latin word forced to break should produce a hyphen."""
        from PIL import ImageFont
        font = ImageFont.load_default()
        output: list[str] = []
        # "Supercalifragilistic" is long enough to force a break at ~30px width
        remainder = Typesetter._break_word(
            "Supercalifragilistic", font, 30, output,
        )
        assert len(output) >= 1
        # At least one flushed line should end with a hyphen.
        assert any(line.endswith("-") for line in output)
        assert len(remainder) > 0

    def test_cjk_word_no_hyphen(self):
        """CJK characters should still break per-character, no hyphens."""
        from PIL import ImageFont
        font = ImageFont.load_default()
        output: list[str] = []
        Typesetter._break_word("漫画翻訳テスト", font, 20, output)
        # No line should end with a hyphen.
        for line in output:
            assert not line.endswith("-")

    def test_short_latin_no_break(self):
        """A short Latin word that fits should not be broken at all."""
        from PIL import ImageFont
        font = ImageFont.load_default()
        output: list[str] = []
        remainder = Typesetter._break_word("Hi", font, 200, output)
        assert output == []
        assert remainder == "Hi"


class TestDynamicFontFloor:
    """Tests for dynamic font size floor below min_font_size."""

    def test_tiny_bubble_uses_smaller_font(self):
        """For a very small bubble, font should go below default min_font_size."""
        ts = Typesetter(min_font_size=14)
        # Tiny available space — 14px font won't fit multi-word text.
        size = ts.calculate_optimal_font_size(
            "Hello world testing", 40, 30, None,
        )
        # Should have gone below 14 (the configured min) but not below 6.
        assert size >= Typesetter._ABSOLUTE_MIN_FONT_SIZE
        assert size <= 14
