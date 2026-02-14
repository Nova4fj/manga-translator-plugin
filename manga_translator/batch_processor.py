"""Multi-threaded batch processing engine for manga translation.

Processes multiple manga pages in parallel with error isolation,
progress tracking, and configurable concurrency.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Callable, Dict


from manga_translator.config.settings import PluginSettings, SettingsManager
from manga_translator.core.image_processor import load_image, save_image
from manga_translator.manga_translator import MangaTranslationPipeline, PageTranslationResult
from manga_translator.cross_page_context import CrossPageContext

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result for a single page in a batch."""
    input_path: str
    output_path: Optional[str] = None
    status: str = "pending"  # pending, processing, complete, failed, skipped
    translation_result: Optional[PageTranslationResult] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    bubble_count: int = 0
    success_rate: float = 0.0


@dataclass
class BatchResult:
    """Aggregated result for an entire batch."""
    pages: List[PageResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.pages)

    @property
    def completed(self) -> int:
        return sum(1 for p in self.pages if p.status == "complete")

    @property
    def failed(self) -> int:
        return sum(1 for p in self.pages if p.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for p in self.pages if p.status == "skipped")

    @property
    def avg_success_rate(self) -> float:
        completed = [p for p in self.pages if p.status == "complete"]
        if not completed:
            return 0.0
        return sum(p.success_rate for p in completed) / len(completed)

    def summary(self) -> str:
        lines = [
            f"Batch complete: {self.completed}/{self.total} succeeded, "
            f"{self.failed} failed, {self.skipped} skipped",
            f"Total time: {self.total_time:.1f}s",
            f"Average success rate: {self.avg_success_rate:.0%}",
        ]
        if self.failed > 0:
            lines.append("Failures:")
            for p in self.pages:
                if p.status == "failed":
                    lines.append(f"  {p.input_path}: {p.error}")
        return "\n".join(lines)


# Type for progress callback: (completed, total, current_file, status_message)
ProgressCallback = Callable[[int, int, str, str], None]


class BatchProcessor:
    """Multi-threaded manga page batch processor.

    Processes multiple pages in parallel using a thread pool. Each page
    is processed independently — failures are isolated and don't affect
    other pages.

    Args:
        settings: Plugin settings. If None, loads defaults.
        max_workers: Maximum concurrent processing threads. Defaults to 2.
        output_dir: Directory for output files. If None, saves next to input.
        output_format: Output format extension (e.g. "png", "jpg").
    """

    def __init__(
        self,
        settings: Optional[PluginSettings] = None,
        max_workers: int = 2,
        output_dir: Optional[str] = None,
        output_format: str = "png",
        enable_cross_page_context: bool = False,
    ):
        if settings is None:
            settings = SettingsManager().get_settings()
        self.settings = settings
        self.enable_cross_page_context = enable_cross_page_context
        # Force sequential when cross-page context is enabled
        if enable_cross_page_context:
            self.max_workers = 1
            logger.info("Cross-page context enabled: forcing sequential processing")
        else:
            self.max_workers = max(max_workers, 1)
        self.output_dir = output_dir
        self.output_format = output_format

        self._lock = Lock()
        self._completed_count = 0

    def process_batch(
        self,
        input_paths: List[str],
        source_lang: str = "ja",
        target_lang: str = "en",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Process a batch of manga pages.

        Args:
            input_paths: List of image file paths.
            source_lang: Source language code.
            target_lang: Target language code.
            progress_callback: Called with (completed, total, current_file, message).

        Returns:
            BatchResult with per-page results and aggregate metrics.
        """
        batch_start = time.time()
        result = BatchResult()

        # Validate input paths
        valid_paths = []
        for path in input_paths:
            if not os.path.isfile(path):
                page_result = PageResult(
                    input_path=path, status="skipped",
                    error=f"File not found: {path}",
                )
                result.pages.append(page_result)
                logger.warning("Skipping %s: file not found", path)
            else:
                valid_paths.append(path)

        if not valid_paths:
            result.total_time = time.time() - batch_start
            return result

        self._completed_count = 0
        total = len(valid_paths)

        # Create output directory if needed
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Create cross-page context if enabled
        cross_page_ctx = CrossPageContext() if self.enable_cross_page_context else None

        # Process pages in parallel (or sequentially if cross-page context)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[Future, str] = {}
            for path in valid_paths:
                output_path = self._get_output_path(path)
                future = executor.submit(
                    self._process_single_page,
                    path, output_path, source_lang, target_lang,
                    cross_page_ctx,
                )
                futures[future] = path

                # When cross-page context is on, wait for each page before starting the next
                if cross_page_ctx:
                    future.result()  # block until done

            for future in as_completed(futures):
                path = futures[future]
                try:
                    page_result = future.result()
                except Exception as e:
                    page_result = PageResult(
                        input_path=path, status="failed",
                        error=str(e),
                    )
                    logger.error("Unexpected error processing %s: %s", path, e)

                result.pages.append(page_result)

                with self._lock:
                    self._completed_count += 1
                    if progress_callback:
                        progress_callback(
                            self._completed_count, total,
                            os.path.basename(path),
                            page_result.status,
                        )

        # Sort results by input order
        path_order = {p: i for i, p in enumerate(input_paths)}
        result.pages.sort(key=lambda r: path_order.get(r.input_path, 999))
        result.total_time = time.time() - batch_start

        logger.info(result.summary())
        return result

    def _process_single_page(
        self,
        input_path: str,
        output_path: str,
        source_lang: str,
        target_lang: str,
        cross_page_context: Optional[CrossPageContext] = None,
    ) -> PageResult:
        """Process a single page (runs in a worker thread)."""
        page_result = PageResult(input_path=input_path, output_path=output_path)
        page_result.status = "processing"
        start = time.time()

        try:
            image = load_image(input_path)
            pipeline = MangaTranslationPipeline(self.settings)
            translation = pipeline.translate_page(
                image, source_lang=source_lang, target_lang=target_lang,
                cross_page_context=cross_page_context,
            )

            save_image(translation.final_image, output_path)

            page_result.status = "complete"
            page_result.translation_result = translation
            page_result.bubble_count = len(translation.bubbles)
            page_result.success_rate = translation.success_rate

            if translation.errors:
                page_result.error = "; ".join(translation.errors)

        except Exception as e:
            page_result.status = "failed"
            page_result.error = str(e)
            logger.error("Failed to process %s: %s", input_path, e)

        page_result.processing_time = time.time() - start
        return page_result

    def _get_output_path(self, input_path: str) -> str:
        """Determine the output path for a given input file."""
        base = os.path.basename(input_path)
        name, _ = os.path.splitext(base)
        output_name = f"{name}_translated.{self.output_format}"

        if self.output_dir:
            return os.path.join(self.output_dir, output_name)
        else:
            return os.path.join(os.path.dirname(input_path), output_name)
