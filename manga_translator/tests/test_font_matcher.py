"""Tests for font matcher."""

import tempfile
from pathlib import Path

from manga_translator.components.font_matcher import FontMatcher, FontProfile
from manga_translator.components.bubble_classifier import BubbleType


class TestFontMatcher:
    def test_default_speech_font(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.SPEECH)
        assert result.style == "comic"
        assert result.weight == "regular"

    def test_default_shout_font(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.SHOUT)
        assert result.style == "gothic"
        assert result.weight == "heavy"

    def test_default_thought_font(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.THOUGHT)
        assert result.style == "serif"
        assert result.weight == "light"

    def test_default_narration_font(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.NARRATION)
        assert result.style == "serif"
        assert result.weight == "regular"

    def test_default_caption_font(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.CAPTION)
        assert result.style == "sans"

    def test_unknown_fallback(self):
        fm = FontMatcher()
        result = fm.match_font(BubbleType.UNKNOWN)
        assert isinstance(result, FontProfile)

    def test_register_custom_font(self):
        fm = FontMatcher()
        custom = FontProfile(
            name="my-comic-font",
            path="/fonts/comic.ttf",
            style="comic",
            weight="bold",
            suitable_for=["speech"],
        )
        fm.register_font(custom)
        result = fm.match_font(BubbleType.SPEECH)
        assert result.name == "my-comic-font"
        assert result.path == "/fonts/comic.ttf"

    def test_custom_font_overrides_default(self):
        fm = FontMatcher()
        custom = FontProfile(
            name="custom-shout",
            style="display",
            weight="heavy",
            suitable_for=["shout"],
        )
        fm.register_font(custom)
        result = fm.match_font(BubbleType.SHOUT)
        assert result.name == "custom-shout"
        assert result.style == "display"

    def test_batch_match(self):
        fm = FontMatcher()
        types = [BubbleType.SPEECH, BubbleType.SHOUT, BubbleType.NARRATION]
        results = fm.match_font_batch(types)
        assert len(results) == 3
        assert results[0].style == "comic"
        assert results[1].style == "gothic"
        assert results[2].style == "serif"

    def test_list_fonts_empty(self):
        fm = FontMatcher()
        assert fm.list_fonts() == {}

    def test_list_fonts_after_register(self):
        fm = FontMatcher()
        fm.register_font(FontProfile(name="test-font"))
        assert "test-font" in fm.list_fonts()

    def test_scan_fonts_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy font files
            (Path(tmpdir) / "ComicBold.ttf").touch()
            (Path(tmpdir) / "GothicBlack.otf").touch()
            (Path(tmpdir) / "readme.txt").touch()  # Should be ignored

            fm = FontMatcher(fonts_dir=tmpdir)
            fonts = fm.list_fonts()
            assert "ComicBold" in fonts
            assert "GothicBlack" in fonts
            assert "readme" not in fonts
            assert fonts["ComicBold"].style == "comic"
            assert fonts["GothicBlack"].style == "gothic"

    def test_scan_nonexistent_dir(self):
        fm = FontMatcher(fonts_dir="/nonexistent/path")
        assert fm.list_fonts() == {}

    def test_guess_weight_from_name(self):
        assert FontMatcher._guess_weight("Arial-Bold") == "bold"
        assert FontMatcher._guess_weight("Thin-Sans") == "light"
        assert FontMatcher._guess_weight("ComicBlack") == "heavy"
        assert FontMatcher._guess_weight("Regular") == "regular"
