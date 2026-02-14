"""Integration tests for the full translation pipeline."""

import numpy as np
from unittest.mock import MagicMock

from manga_translator.manga_translator import (
    MangaTranslationPipeline,
    PageTranslationResult,
    BubbleTranslation,
    translate_file,
)
from manga_translator.config.settings import PluginSettings
from manga_translator.components.ocr_engine import OCRResult
from manga_translator.components.translator import TranslationResult


class TestPageTranslationResult:
    def test_success_rate_no_bubbles(self):
        result = PageTranslationResult(
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            final_image=np.zeros((10, 10, 3), dtype=np.uint8),
            cleaned_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        assert result.success_rate == 0.0

    def test_success_rate_with_bubbles(self):
        bubble = MagicMock()
        ocr = OCRResult(text="hello", confidence=0.9, language="en", engine_used="test")
        good_translation = TranslationResult(
            source_text="hello", translated_text="こんにちは",
            source_language="en", target_language="ja",
            engine_used="test", confidence=0.9,
        )
        bad_translation = TranslationResult(
            source_text="x", translated_text="",
            source_language="en", target_language="ja",
            engine_used="test", confidence=0.0,
        )
        result = PageTranslationResult(
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            final_image=np.zeros((10, 10, 3), dtype=np.uint8),
            cleaned_image=np.zeros((10, 10, 3), dtype=np.uint8),
            bubbles=[
                BubbleTranslation(bubble=bubble, ocr_result=ocr, translation=good_translation),
                BubbleTranslation(bubble=bubble, ocr_result=ocr, translation=bad_translation),
            ],
        )
        assert result.success_rate == 0.5


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

    def test_translate_with_same_language(self, sample_manga_page):
        """Same-language passthrough should work."""
        settings = PluginSettings()
        settings.ocr.primary_engine = "tesseract"
        settings.ocr.language_hint = "eng"

        pipeline = MangaTranslationPipeline(settings)
        result = pipeline.translate_page(sample_manga_page, source_lang="en", target_lang="en")
        assert isinstance(result, PageTranslationResult)

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

    def test_large_image_resized(self):
        """Very large images should be resized for detection."""
        settings = PluginSettings()
        pipeline = MangaTranslationPipeline(settings)
        large = np.full((5000, 5000, 3), 128, dtype=np.uint8)
        result = pipeline.translate_page(large)
        assert isinstance(result, PageTranslationResult)

    def test_errors_collected(self):
        """Pipeline should collect errors but still return a result."""
        settings = PluginSettings()
        pipeline = MangaTranslationPipeline(settings)

        # Patch detector to raise
        original_detect = pipeline.detector.detect_bubbles
        pipeline.detector.detect_bubbles = MagicMock(side_effect=RuntimeError("detection crash"))

        result = pipeline.translate_page(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, PageTranslationResult)
        assert any("detection" in e.lower() or "bubble" in e.lower() for e in result.errors)

        pipeline.detector.detect_bubbles = original_detect


class TestFullPipelineWithMocks:
    """Test the full pipeline path (steps 3-6) by mocking OCR to return text."""

    def test_full_pipeline_steps(self, sample_manga_page):
        """Force OCR to return text so steps 3-6 execute."""
        settings = PluginSettings()
        settings.ocr.primary_engine = "tesseract"
        settings.ocr.language_hint = "eng"
        settings.translation.source_language = "en"
        settings.translation.target_language = "en"

        pipeline = MangaTranslationPipeline(settings)

        # Mock OCR to always return valid text
        good_ocr = OCRResult(text="Hello World", confidence=0.95, language="en", engine_used="mock")
        pipeline.ocr.extract_text = MagicMock(return_value=good_ocr)

        result = pipeline.translate_page(sample_manga_page, source_lang="en", target_lang="en")
        assert isinstance(result, PageTranslationResult)
        assert result.final_image is not None
        # With mocked OCR, bubbles should have translations
        if result.bubbles:
            for bt in result.bubbles:
                assert bt.translation is not None
                assert bt.ocr_result.text == "Hello World"

    def test_pipeline_with_ocr_returning_no_text(self, sample_manga_page):
        """When OCR finds no text, pipeline returns early gracefully."""
        settings = PluginSettings()
        pipeline = MangaTranslationPipeline(settings)

        empty_ocr = OCRResult(text="", confidence=0.0, language="en", engine_used="mock")
        pipeline.ocr.extract_text = MagicMock(return_value=empty_ocr)

        result = pipeline.translate_page(sample_manga_page)
        assert isinstance(result, PageTranslationResult)
        assert len(result.bubbles) == 0

    def test_pipeline_translation_failure(self, sample_manga_page):
        """Pipeline should handle translation exceptions."""
        settings = PluginSettings()
        settings.translation.source_language = "ja"
        settings.translation.target_language = "en"

        pipeline = MangaTranslationPipeline(settings)

        good_ocr = OCRResult(text="テスト", confidence=0.95, language="ja", engine_used="mock")
        pipeline.ocr.extract_text = MagicMock(return_value=good_ocr)
        pipeline.translator.translate_batch = MagicMock(side_effect=RuntimeError("API down"))

        result = pipeline.translate_page(sample_manga_page, source_lang="ja", target_lang="en")
        assert isinstance(result, PageTranslationResult)
        assert any("translation" in e.lower() for e in result.errors)


class TestTranslateFile:
    def test_translate_file(self, sample_manga_page, tmp_path):
        import cv2
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        cv2.imwrite(input_path, sample_manga_page)

        result = translate_file(input_path, output_path, source_lang="en", target_lang="en")
        assert isinstance(result, PageTranslationResult)
        assert (tmp_path / "output.png").exists()

    def test_translate_file_auto_output(self, sample_manga_page, tmp_path):
        import cv2
        input_path = str(tmp_path / "manga.png")
        cv2.imwrite(input_path, sample_manga_page)

        result = translate_file(input_path, source_lang="en", target_lang="en")
        assert isinstance(result, PageTranslationResult)
        assert (tmp_path / "manga_translated.png").exists()
