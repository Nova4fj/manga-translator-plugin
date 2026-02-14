"""Main translation pipeline — wires all components together."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import numpy as np

from manga_translator.components.bubble_detector import BubbleDetector, BubbleRegion
from manga_translator.components.ocr_engine import OCREngine, OCRResult
from manga_translator.components.translator import TranslationManager, TranslationResult
from manga_translator.components.inpainter import Inpainter, InpaintResult
from manga_translator.components.typesetter import Typesetter, TypesetResult
from manga_translator.config.settings import PluginSettings, SettingsManager
from manga_translator.core.image_processor import resize_for_processing, scale_bbox
from manga_translator.core.layer_manager import LayerStack
from manga_translator.ui.progress import ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class BubbleTranslation:
    """Translation data for a single bubble."""

    bubble: BubbleRegion
    ocr_result: OCRResult
    translation: TranslationResult
    inpaint_result: Optional[InpaintResult] = None
    typeset_result: Optional[TypesetResult] = None


@dataclass
class PageTranslationResult:
    """Complete translation result for a manga page."""

    original_image: np.ndarray
    final_image: np.ndarray
    cleaned_image: np.ndarray  # after inpainting, before typesetting
    bubbles: List[BubbleTranslation] = field(default_factory=list)
    layer_stack: Optional[LayerStack] = None
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.bubbles:
            return 0.0
        successful = sum(
            1
            for b in self.bubbles
            if b.translation.translated_text and b.translation.confidence > 0.3
        )
        return successful / len(self.bubbles)


class MangaTranslationPipeline:
    """End-to-end manga page translation pipeline.

    Pipeline: Detect bubbles → OCR → Translate → Inpaint → Typeset → Assemble
    """

    def __init__(self, settings: Optional[PluginSettings] = None):
        if settings is None:
            settings = SettingsManager().get_settings()
        self.settings = settings

        # Initialize components
        self.detector = BubbleDetector(
            min_area=settings.detection.min_bubble_area,
            max_area=settings.detection.max_bubble_area,
            edge_sensitivity=settings.detection.edge_sensitivity,
            contour_approx_epsilon=settings.detection.contour_approx_epsilon,
            min_aspect_ratio=settings.detection.min_aspect_ratio,
            max_aspect_ratio=settings.detection.max_aspect_ratio,
        )
        self.ocr = OCREngine(
            primary_engine=settings.ocr.primary_engine,
            confidence_threshold=settings.ocr.confidence_threshold,
            tesseract_path=settings.ocr.tesseract_path,
            language_hint=settings.ocr.language_hint,
        )
        self.translator = TranslationManager(
            primary_engine=settings.translation.primary_engine,
            openai_api_key=settings.translation.openai_api_key,
            openai_model=settings.translation.openai_model,
            deepl_api_key=settings.translation.deepl_api_key,
            context_prompt=settings.translation.context_prompt,
        )
        self.inpainter = Inpainter(
            method=settings.inpainting.method,
            inpaint_radius=settings.inpainting.inpaint_radius,
            blur_kernel_size=settings.inpainting.blur_kernel_size,
            mask_dilation=settings.inpainting.mask_dilation,
        )
        self.typesetter = Typesetter(
            default_font=settings.typesetting.default_font,
            font_size_ratio=settings.typesetting.font_size_ratio,
            min_font_size=settings.typesetting.min_font_size,
            max_font_size=settings.typesetting.max_font_size,
            text_color=settings.typesetting.text_color,
            outline_color=settings.typesetting.outline_color,
            outline_width=settings.typesetting.outline_width,
            line_spacing=settings.typesetting.line_spacing,
            padding_ratio=settings.typesetting.padding_ratio,
        )

    def translate_page(
        self,
        image: np.ndarray,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> PageTranslationResult:
        """Translate a full manga page.

        Args:
            image: BGR numpy array of the manga page.
            source_lang: Override source language.
            target_lang: Override target language.
            progress_callback: Called with (step, total_steps, message).

        Returns:
            PageTranslationResult with all translation data.
        """
        src = source_lang or self.settings.translation.source_language
        tgt = target_lang or self.settings.translation.target_language

        tracker = ProgressTracker(callback=progress_callback)
        errors = []
        layer_stack = LayerStack(width=image.shape[1], height=image.shape[0])
        layer_stack.add_layer("Original", image.copy())

        # --- Step 1: Bubble Detection ---
        tracker.start_step(0)
        try:
            processed_image, scale = resize_for_processing(image, max_dimension=4096)
            bubbles = self.detector.detect_bubbles(processed_image)

            # Scale bboxes back to original size if resized
            if scale != 1.0:
                for b in bubbles:
                    b.bbox = scale_bbox(b.bbox, scale)

            logger.info("Detected %d bubbles", len(bubbles))
        except Exception as e:
            logger.error("Bubble detection failed: %s", e, exc_info=True)
            errors.append(f"Bubble detection failed: {e}")
            bubbles = []
        tracker.complete_step(0)

        if not bubbles:
            return PageTranslationResult(
                original_image=image,
                final_image=image.copy(),
                cleaned_image=image.copy(),
                errors=errors or ["No speech bubbles detected"],
                layer_stack=layer_stack,
            )

        # --- Step 2: OCR ---
        tracker.start_step(1)
        ocr_results: List[OCRResult] = []
        try:
            for bubble in bubbles:
                x, y, w, h = bubble.bbox
                region = image[y : y + h, x : x + w]
                if region.size == 0:
                    ocr_results.append(
                        OCRResult(text="", confidence=0.0, language=src, engine_used="none")
                    )
                    continue
                result = self.ocr.extract_text(region, language_hint=src)
                ocr_results.append(result)
                logger.debug("OCR [%d]: '%s' (%.2f)", bubble.id, result.text[:50], result.confidence)
        except Exception as e:
            logger.error("OCR failed: %s", e, exc_info=True)
            errors.append(f"OCR failed: {e}")
        tracker.complete_step(1)

        # Filter bubbles with no detected text
        valid_pairs = [
            (b, o)
            for b, o in zip(bubbles, ocr_results)
            if o.text.strip() and o.confidence > 0.1
        ]

        if not valid_pairs:
            return PageTranslationResult(
                original_image=image,
                final_image=image.copy(),
                cleaned_image=image.copy(),
                errors=errors or ["No text detected in bubbles"],
                layer_stack=layer_stack,
            )

        valid_bubbles, valid_ocr = zip(*valid_pairs)

        # --- Step 3: Translation ---
        tracker.start_step(2)
        translations: List[TranslationResult] = []
        try:
            texts = [o.text for o in valid_ocr]
            translations = self.translator.translate_batch(texts, src, tgt)
        except Exception as e:
            logger.error("Translation failed: %s", e, exc_info=True)
            errors.append(f"Translation failed: {e}")
            translations = [
                TranslationResult(
                    source_text=o.text,
                    translated_text="",
                    source_language=src,
                    target_language=tgt,
                    engine_used="none",
                    confidence=0.0,
                    error=str(e),
                )
                for o in valid_ocr
            ]
        tracker.complete_step(2)

        # --- Step 4: Inpainting (text removal) ---
        tracker.start_step(3)
        cleaned = image.copy()
        inpaint_results: List[Optional[InpaintResult]] = []
        try:
            for bubble in valid_bubbles:
                x, y, w, h = bubble.bbox
                region = cleaned[y : y + h, x : x + w]
                if region.size == 0:
                    inpaint_results.append(None)
                    continue

                # Create text mask for this bubble
                bubble_mask = bubble.mask
                if bubble_mask is not None:
                    # Crop mask to bbox
                    local_mask = bubble_mask[y : y + h, x : x + w]
                else:
                    local_mask = np.ones((h, w), dtype=np.uint8) * 255

                text_mask = self.inpainter.create_text_mask(region, local_mask)
                result = self.inpainter.remove_text(region, text_mask)
                cleaned[y : y + h, x : x + w] = result.image
                inpaint_results.append(result)
        except Exception as e:
            logger.error("Inpainting failed: %s", e, exc_info=True)
            errors.append(f"Inpainting failed: {e}")
        tracker.complete_step(3)

        layer_stack.add_layer("Cleaned", cleaned.copy())

        # --- Step 5: Typesetting ---
        tracker.start_step(4)
        final = cleaned.copy()
        typeset_results: List[Optional[TypesetResult]] = []
        try:
            for bubble, translation in zip(valid_bubbles, translations):
                if not translation.translated_text:
                    typeset_results.append(None)
                    continue

                result = self.typesetter.typeset_text(
                    final,
                    translation.translated_text,
                    bubble.bbox,
                    bubble_mask=bubble.mask,
                )
                final = result.image
                typeset_results.append(result)
        except Exception as e:
            logger.error("Typesetting failed: %s", e, exc_info=True)
            errors.append(f"Typesetting failed: {e}")
        tracker.complete_step(4)

        # --- Step 6: Assemble result ---
        tracker.start_step(5)
        layer_stack.add_layer("Translated", final.copy())

        bubble_translations = []
        for i, (bubble, ocr) in enumerate(zip(valid_bubbles, valid_ocr)):
            bt = BubbleTranslation(
                bubble=bubble,
                ocr_result=ocr,
                translation=translations[i] if i < len(translations) else TranslationResult(
                    source_text=ocr.text, translated_text="", source_language=src,
                    target_language=tgt, engine_used="none", confidence=0.0,
                ),
                inpaint_result=inpaint_results[i] if i < len(inpaint_results) else None,
                typeset_result=typeset_results[i] if i < len(typeset_results) else None,
            )
            bubble_translations.append(bt)

        tracker.complete_step(5)

        result = PageTranslationResult(
            original_image=image,
            final_image=final,
            cleaned_image=cleaned,
            bubbles=bubble_translations,
            layer_stack=layer_stack,
            errors=errors,
        )

        logger.info(
            "Translation complete: %d bubbles, %.0f%% success rate",
            len(bubble_translations),
            result.success_rate * 100,
        )
        print(tracker.get_summary())

        return result


def translate_file(
    input_path: str,
    output_path: Optional[str] = None,
    source_lang: str = "ja",
    target_lang: str = "en",
    settings: Optional[PluginSettings] = None,
) -> PageTranslationResult:
    """Convenience function: translate a manga image file.

    Args:
        input_path: Path to the manga image.
        output_path: Where to save the result. If None, appends '_translated'.
        source_lang: Source language code.
        target_lang: Target language code.
        settings: Plugin settings. If None, loads from config.

    Returns:
        PageTranslationResult
    """
    from manga_translator.core.image_processor import load_image, save_image
    import os

    image = load_image(input_path)

    if settings is None:
        settings = SettingsManager().get_settings()
    settings.translation.source_language = source_lang
    settings.translation.target_language = target_lang

    pipeline = MangaTranslationPipeline(settings)

    def console_progress(step, total, message):
        print(f"  [{step}/{total}] {message}")

    result = pipeline.translate_page(image, progress_callback=console_progress)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_translated{ext}"

    save_image(result.final_image, output_path)
    logger.info("Saved translated image to %s", output_path)

    return result
