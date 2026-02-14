"""Main translation pipeline — wires all components together."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import numpy as np

from manga_translator.perf_monitor import PerfMonitor
from manga_translator.quality_control import QualityChecker, QCReport
from manga_translator.translation_memory import TranslationMemory

from manga_translator.components.bubble_detector import BubbleDetector, BubbleRegion
from manga_translator.components.ocr_engine import OCREngine, OCRResult
from manga_translator.components.translator import TranslationManager, TranslationResult
from manga_translator.components.inpainter import Inpainter, InpaintResult
from manga_translator.components.typesetter import Typesetter, TypesetResult
from manga_translator.components.text_region_filter import TextRegionFilter
from manga_translator.components.bubble_classifier import BubbleClassifier, BubbleType
from manga_translator.components.reading_order import ReadingOrderOptimizer
from manga_translator.components.font_matcher import FontMatcher
from manga_translator.components.sfx_detector import SFXDetector, SFXRegion
from manga_translator.translation_context import ContextBuilder
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
    sfx_regions: List['SFXRegion'] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    qc_report: Optional['QCReport'] = None
    perf_summary: str = ""

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

    def __init__(
        self,
        settings: Optional[PluginSettings] = None,
        perf_monitor: Optional[PerfMonitor] = None,
        quality_checker: Optional[QualityChecker] = None,
        translation_memory: Optional[TranslationMemory] = None,
        reading_direction: str = "rtl",
        detect_sfx: bool = False,
        fonts_dir: Optional[str] = None,
    ):
        if settings is None:
            settings = SettingsManager().get_settings()
        self.settings = settings
        self._perf = perf_monitor
        self._qc = quality_checker
        self._tm = translation_memory

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
        # Determine inpainting method — prefer LaMa when neural is enabled
        inpaint_method = settings.inpainting.method
        lama_device = "auto" if settings.gpu_enabled else "cpu"
        model_dir = settings.model_cache_dir or None

        self.inpainter = Inpainter(
            method=inpaint_method,
            inpaint_radius=settings.inpainting.inpaint_radius,
            blur_kernel_size=settings.inpainting.blur_kernel_size,
            mask_dilation=settings.inpainting.mask_dilation,
            lama_model_dir=model_dir,
            lama_device=lama_device,
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
            alignment=settings.typesetting.alignment,
            font_category=settings.typesetting.font_category,
        )
        self._quality_threshold = settings.inpainting.quality_threshold
        self.text_filter = TextRegionFilter()
        self.context_builder = ContextBuilder(context_window=2)

        # Phase 10 components
        self.classifier = BubbleClassifier()
        self.reading_order = ReadingOrderOptimizer(reading_direction=reading_direction)
        self.font_matcher = FontMatcher(fonts_dir=fonts_dir)
        self._detect_sfx = detect_sfx
        self.sfx_detector = SFXDetector() if detect_sfx else None

    def translate_page(
        self,
        image: np.ndarray,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> PageTranslationResult:
        """Translate a full manga page.

        Args:
            image: BGR numpy array of the manga page.
            source_lang: Override source language.
            target_lang: Override target language.
            progress_callback: Called with (step, total_steps, message).
            exclusion_mask: Optional binary mask (0 = excluded, 255 = included).
                Bubbles whose center falls in an excluded region are skipped.

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
        with self._perf.track("detection") if self._perf else nullcontext():
            try:
                processed_image, scale = resize_for_processing(image, max_dimension=4096)
                bubbles = self.detector.detect_bubbles(processed_image)

                # Scale bboxes back to original size if resized
                if scale != 1.0:
                    for b in bubbles:
                        b.bbox = scale_bbox(b.bbox, scale)

                logger.info("Detected %d bubbles", len(bubbles))

                # Apply region exclusion filtering
                if exclusion_mask is not None:
                    from manga_translator.region_mask import filter_bubbles_by_mask
                    bubbles = filter_bubbles_by_mask(bubbles, exclusion_mask)
                    logger.info("After exclusion filtering: %d bubbles", len(bubbles))

                # Classify bubble types
                if bubbles:
                    for bubble in bubbles:
                        try:
                            cr = self.classifier.classify(
                                bubble.contour, image_shape=image.shape[:2], bbox=bubble.bbox,
                            )
                            bubble.bubble_type = cr.bubble_type.value
                        except Exception:
                            bubble.bubble_type = "unknown"

                    # Sort in reading order
                    bubbles = self.reading_order.sort_bubbles(bubbles)
                    logger.info("Sorted %d bubbles in %s reading order", len(bubbles), self.reading_order.reading_direction)

            except Exception as e:
                logger.error("Bubble detection failed: %s", e, exc_info=True)
                errors.append(f"Bubble detection failed: {e}")
                bubbles = []
        tracker.complete_step(0)

        # Optional SFX detection
        sfx_regions: List[SFXRegion] = []
        if self.sfx_detector and bubbles:
            try:
                sfx_regions = self.sfx_detector.detect_sfx(image, bubbles)
                logger.info("Detected %d SFX regions", len(sfx_regions))
            except Exception as e:
                logger.warning("SFX detection failed: %s", e)
                errors.append(f"SFX detection failed: {e}")

        if not bubbles:
            return PageTranslationResult(
                original_image=image,
                final_image=image.copy(),
                cleaned_image=image.copy(),
                errors=errors or ["No speech bubbles detected"],
                layer_stack=layer_stack,
            )

        # --- Step 2: OCR (parallel) ---
        tracker.start_step(1)
        ocr_results: List[OCRResult] = [None] * len(bubbles)  # type: ignore[list-item]
        with self._perf.track("ocr") if self._perf else nullcontext():
            def _ocr_one(idx_bubble):
                idx, bubble = idx_bubble
                try:
                    x, y, w, h = bubble.bbox
                    region = image[y : y + h, x : x + w]
                    if region.size == 0:
                        return idx, OCRResult(text="", confidence=0.0, language=src, engine_used="none"), None
                    # Pre-filter: skip OCR on non-text regions
                    score = self.text_filter.analyze_region(region)
                    if not score.has_text:
                        logger.debug("Filtered non-text region [%d]: %s (%.2f)", bubble.id, score.region_type, score.confidence)
                        return idx, OCRResult(text="", confidence=0.0, language=src, engine_used="filtered"), None
                    result = self.ocr.extract_text(region, language_hint=src)
                    logger.debug("OCR [%d]: '%s' (%.2f)", bubble.id, result.text[:50], result.confidence)
                    return idx, result, None
                except Exception as e:
                    logger.warning("OCR failed for bubble %d: %s", bubble.id, e)
                    return idx, OCRResult(text="", confidence=0.0, language=src, engine_used="error"), f"OCR failed for bubble {bubble.id}: {e}"

            try:
                max_workers = min(4, len(bubbles))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_ocr_one, (i, b)): i for i, b in enumerate(bubbles)}
                    for future in as_completed(futures):
                        idx, result, error = future.result()
                        ocr_results[idx] = result
                        if error:
                            errors.append(error)
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
        with self._perf.track("translation") if self._perf else nullcontext():
            try:
                texts = [o.text for o in valid_ocr]

                # Build page context for better translations
                page_ctx = self.context_builder.build_page_context(texts)
                page_prompt = self.context_builder.format_page_prompt(page_ctx)

                # Temporarily enhance context prompt on the actual engine
                openai_engine = self.translator._engines.get("openai")
                original_prompt = (
                    openai_engine._context_prompt if openai_engine else None
                )
                if page_prompt and openai_engine:
                    enhanced = (
                        f"{original_prompt}\n{page_prompt}"
                        if original_prompt
                        else page_prompt
                    )
                    openai_engine._context_prompt = enhanced
                    logger.debug("Enhanced context prompt with page context")

                try:
                    # Check TM for cached translations
                    tm_hits: dict = {}  # index -> target_text
                    texts_to_translate: List[str] = []
                    indices_to_translate: List[int] = []
                    if self._tm:
                        for idx, text in enumerate(texts):
                            match = self._tm.lookup_exact(text, src, tgt)
                            if match:
                                tm_hits[idx] = match.target_text
                                logger.debug("TM hit for: '%s'", text[:50])
                            else:
                                texts_to_translate.append(text)
                                indices_to_translate.append(idx)
                    else:
                        texts_to_translate = texts
                        indices_to_translate = list(range(len(texts)))

                    # Translate texts not found in TM
                    if texts_to_translate:
                        batch_results = self.translator.translate_batch(
                            texts_to_translate, src, tgt
                        )
                    else:
                        batch_results = []

                    # Merge TM hits and fresh translations
                    fresh_iter = iter(batch_results)
                    for idx in range(len(texts)):
                        if idx in tm_hits:
                            translations.append(TranslationResult(
                                source_text=texts[idx],
                                translated_text=tm_hits[idx],
                                source_language=src,
                                target_language=tgt,
                                engine_used="translation_memory",
                                confidence=1.0,
                            ))
                        else:
                            tr = next(fresh_iter)
                            translations.append(tr)
                            # Store new translation in TM
                            if self._tm and tr.translated_text and tr.confidence > 0.3:
                                self._tm.add_entry(
                                    source_text=tr.source_text,
                                    target_text=tr.translated_text,
                                    source_lang=src,
                                    target_lang=tgt,
                                    quality_score=tr.confidence,
                                )
                finally:
                    # Restore original context prompt
                    if openai_engine and original_prompt is not None:
                        openai_engine._context_prompt = original_prompt

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
        with self._perf.track("inpainting") if self._perf else nullcontext():
            try:
                for bubble in valid_bubbles:
                    try:
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
                        result = self.inpainter.remove_text_with_fallback(
                            region, text_mask,
                            quality_threshold=self._quality_threshold,
                        )
                        cleaned[y : y + h, x : x + w] = result.image
                        inpaint_results.append(result)
                    except Exception as e:
                        logger.warning("Inpainting failed for bubble %d: %s", bubble.id, e)
                        errors.append(f"Inpainting failed for bubble {bubble.id}: {e}")
                        inpaint_results.append(None)
            except Exception as e:
                logger.error("Inpainting failed: %s", e, exc_info=True)
                errors.append(f"Inpainting failed: {e}")
        tracker.complete_step(3)

        layer_stack.add_layer("Cleaned", cleaned.copy())

        # --- Step 5: Typesetting ---
        tracker.start_step(4)
        final = cleaned.copy()
        typeset_results: List[Optional[TypesetResult]] = []
        with self._perf.track("typesetting") if self._perf else nullcontext():
            try:
                for bubble, translation in zip(valid_bubbles, translations):
                    try:
                        if not translation.translated_text:
                            typeset_results.append(None)
                            continue

                        # Select font based on bubble type
                        font_override = None
                        if bubble.bubble_type:
                            try:
                                bt = BubbleType(bubble.bubble_type)
                                font_profile = self.font_matcher.match_font(bt, translation.translated_text, tgt)
                                if font_profile.path:
                                    font_override = font_profile.path
                            except (ValueError, Exception):
                                pass

                        result = self.typesetter.typeset_text(
                            final,
                            translation.translated_text,
                            bubble.bbox,
                            bubble_mask=bubble.mask,
                            orientation=self.settings.typesetting.orientation,
                            source_lang=src,
                            font_override=font_override,
                        )
                        final = result.image
                        typeset_results.append(result)
                    except Exception as e:
                        logger.warning("Typesetting failed for bubble %d: %s", bubble.id, e)
                        errors.append(f"Typesetting failed for bubble {bubble.id}: {e}")
                        typeset_results.append(None)
            except Exception as e:
                logger.error("Typesetting failed: %s", e, exc_info=True)
                errors.append(f"Typesetting failed: {e}")
        tracker.complete_step(4)

        # --- Quality Control (optional) ---
        qc_report = None
        if self._qc:
            with self._perf.track("quality_control") if self._perf else nullcontext():
                pairs = [
                    {"source": bt_ocr.text, "target": translations[i].translated_text}
                    for i, bt_ocr in enumerate(valid_ocr)
                    if i < len(translations)
                ]
                qc_report = self._qc.check_page(pairs, target_lang=tgt, page_num=1)

        # --- Step 6: Assemble result ---
        tracker.start_step(5)
        with self._perf.track("assembly") if self._perf else nullcontext():
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

        # Build performance summary
        perf_summary = ""
        if self._perf:
            perf_summary = self._perf.report().summary()

        result = PageTranslationResult(
            original_image=image,
            final_image=final,
            cleaned_image=cleaned,
            bubbles=bubble_translations,
            sfx_regions=sfx_regions,
            layer_stack=layer_stack,
            errors=errors,
            qc_report=qc_report,
            perf_summary=perf_summary,
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
    tm_db_path: Optional[str] = None,
    enable_qc: bool = False,
    enable_perf: bool = False,
    exclude_regions: Optional[str] = None,
    reading_direction: str = "rtl",
    detect_sfx: bool = False,
    fonts_dir: Optional[str] = None,
) -> PageTranslationResult:
    """Convenience function: translate a manga image file.

    Args:
        input_path: Path to the manga image.
        output_path: Where to save the result. If None, appends '_translated'.
        source_lang: Source language code.
        target_lang: Target language code.
        settings: Plugin settings. If None, loads from config.
        tm_db_path: Path to translation memory database. If set, enables TM.
        enable_qc: Enable quality control checks.
        enable_perf: Enable performance monitoring.
        exclude_regions: Semicolon-separated exclusion regions (x,y,w,h;...).

    Returns:
        PageTranslationResult
    """
    from manga_translator.core.image_processor import load_image, save_image
    from manga_translator.input_validator import InputValidator, ValidationError
    import os

    InputValidator.validate_image_path(input_path)
    InputValidator.validate_language(source_lang, "source_lang")
    InputValidator.validate_language(target_lang, "target_lang")
    if output_path:
        InputValidator.validate_output_path(output_path)

    image = load_image(input_path)

    # Build exclusion mask from region string if provided
    exclusion_mask = None
    if exclude_regions:
        from manga_translator.region_mask import parse_exclusion_regions, create_exclusion_mask
        regions = parse_exclusion_regions(exclude_regions)
        if regions:
            h, w = image.shape[:2]
            exclusion_mask = create_exclusion_mask(h, w, regions)
            logger.info("Created exclusion mask with %d regions", len(regions))

    if settings is None:
        settings = SettingsManager().get_settings()
    settings.translation.source_language = source_lang
    settings.translation.target_language = target_lang

    # Create optional Phase 2-3 objects
    perf_monitor = PerfMonitor() if enable_perf else None
    quality_checker = QualityChecker() if enable_qc else None
    translation_memory = TranslationMemory(db_path=tm_db_path) if tm_db_path else None

    if perf_monitor:
        perf_monitor.start()

    pipeline = MangaTranslationPipeline(
        settings,
        perf_monitor=perf_monitor,
        quality_checker=quality_checker,
        translation_memory=translation_memory,
        reading_direction=reading_direction,
        detect_sfx=detect_sfx,
        fonts_dir=fonts_dir,
    )

    def console_progress(step, total, message):
        print(f"  [{step}/{total}] {message}")

    result = pipeline.translate_page(
        image, progress_callback=console_progress, exclusion_mask=exclusion_mask
    )

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_translated{ext}"

    save_image(result.final_image, output_path)
    logger.info("Saved translated image to %s", output_path)

    return result
