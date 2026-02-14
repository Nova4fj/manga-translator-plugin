"""Integration tests for the full translation pipeline."""

import numpy as np
import pytest

from manga_translator.manga_translator import MangaTranslationPipeline, PageTranslationResult
from manga_translator.config.settings import PluginSettings


class TestPipeline:
    def test_init_with_defaults(self):
        pipeline = MangaTranslationPipeline()
        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.ocr is not None
        assert pipeline.translator is not None
        assert pipeline.inpainter is not None
        assert pipeline.typesetter is not None

    def test_translate_empty_page(self):
        """Translating an empty (black) page should return gracefully."""
        pipeline = MangaTranslationPipeline()
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.translate_page(empty)
        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) == 0

    def test_translate_synthetic_page(self, sample_manga_page):
        """Translate the synthetic manga page fixture."""
        settings = PluginSettings()
        settings.ocr.primary_engine = "tesseract"
        settings.ocr.language_hint = "eng"

        pipeline = MangaTranslationPipeline(settings)
        result = pipeline.translate_page(sample_manga_page)

        assert isinstance(result, PageTranslationResult)
        assert result.original_image is not None
        assert result.final_image is not None
        assert result.final_image.shape == sample_manga_page.shape

    def test_progress_callback(self, sample_manga_page):
        """Progress callback should be called during translation."""
        settings = PluginSettings()
        settings.ocr.primary_engine = "tesseract"
        settings.ocr.language_hint = "eng"

        steps_seen = []

        def on_progress(step, total, message):
            steps_seen.append((step, total, message))

        pipeline = MangaTranslationPipeline(settings)
        pipeline.translate_page(sample_manga_page, progress_callback=on_progress)

        assert len(steps_seen) > 0

    def test_layer_stack(self, sample_manga_page):
        """Layer stack should contain original and translated layers."""
        settings = PluginSettings()
        settings.ocr.primary_engine = "tesseract"
        settings.ocr.language_hint = "eng"

        pipeline = MangaTranslationPipeline(settings)
        result = pipeline.translate_page(sample_manga_page)

        if result.layer_stack:
            assert result.layer_stack.get_layer("Original") is not None
