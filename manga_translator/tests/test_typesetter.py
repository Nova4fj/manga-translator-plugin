"""Tests for typesetting component."""

import numpy as np
import pytest

from manga_translator.components.typesetter import Typesetter, TypesetResult, TextLayout


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
