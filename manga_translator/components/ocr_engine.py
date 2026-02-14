"""Multi-engine OCR for manga text extraction."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float  # 0.0 to 1.0
    language: str  # detected or hinted language
    engine_used: str
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Return True if the result contains meaningful text."""
        return bool(self.text and self.text.strip()) and self.error is None

    @staticmethod
    def empty(engine: str = "none", error: Optional[str] = None) -> "OCRResult":
        """Create an empty result, typically used for errors."""
        return OCRResult(
            text="",
            confidence=0.0,
            language="unknown",
            engine_used=engine,
            error=error,
        )


class BaseOCREngine(ABC):
    """Abstract base for OCR engines."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this engine's dependencies are installed."""
        ...

    @abstractmethod
    def extract_text(
        self, image: np.ndarray, language_hint: str = "ja"
    ) -> OCRResult:
        """Run OCR on a single image array (H, W, C) or (H, W) grayscale."""
        ...


# ---------------------------------------------------------------------------
# manga-ocr engine -- specialised for Japanese manga
# ---------------------------------------------------------------------------

class MangaOCREngine(BaseOCREngine):
    """manga-ocr, specialised for Japanese manga text bubbles.

    Handles mixed hiragana / katakana / kanji well and does not require
    Tesseract.  The underlying transformer model is lazy-loaded on first
    use so that import time stays low.
    """

    name = "manga-ocr"

    def __init__(self) -> None:
        self._model = None  # lazy
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import manga_ocr as _mocr  # noqa: F401
            self._available = True
        except ImportError:
            logger.debug("manga-ocr is not installed.")
            self._available = False
        return self._available

    def _load_model(self):
        """Lazy-load the MangaOcr model (downloads weights on first run)."""
        if self._model is not None:
            return
        from manga_ocr import MangaOcr

        logger.info("Loading manga-ocr model (first call may download weights) ...")
        self._model = MangaOcr()
        logger.info("manga-ocr model loaded.")

    def extract_text(
        self, image: np.ndarray, language_hint: str = "ja"
    ) -> OCRResult:
        if not self.is_available():
            return OCRResult.empty(self.name, error="manga-ocr is not installed")

        try:
            self._load_model()

            from PIL import Image

            # Convert numpy array to PIL Image for manga-ocr
            if image.ndim == 2:
                pil_image = Image.fromarray(image, mode="L")
            elif image.ndim == 3 and image.shape[2] == 4:
                pil_image = Image.fromarray(image, mode="RGBA").convert("RGB")
            elif image.ndim == 3:
                pil_image = Image.fromarray(image, mode="RGB")
            else:
                return OCRResult.empty(
                    self.name,
                    error=f"Unsupported image shape: {image.shape}",
                )

            text: str = self._model(pil_image)
            text = text.strip()

            # manga-ocr does not provide a native confidence score, so we
            # derive a heuristic one: non-empty text gets 0.90 and empty text
            # gets 0.0.
            confidence = 0.90 if text else 0.0

            return OCRResult(
                text=text,
                confidence=confidence,
                language="ja",
                engine_used=self.name,
            )
        except Exception as exc:
            logger.warning("manga-ocr extraction failed: %s", exc, exc_info=True)
            return OCRResult.empty(self.name, error=str(exc))


# ---------------------------------------------------------------------------
# PaddleOCR engine -- good for CJK (zh, ko, ja)
# ---------------------------------------------------------------------------

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine supporting Chinese, Korean, and Japanese."""

    name = "paddleocr"

    # PaddleOCR language code mapping
    _LANG_MAP: Dict[str, str] = {
        "ja": "japan",
        "zh": "ch",
        "zh-cn": "ch",
        "zh-tw": "chinese_cht",
        "ko": "korean",
        "en": "en",
    }

    def __init__(self) -> None:
        # Cache one PaddleOCR instance per language to avoid re-init
        self._engines: Dict[str, object] = {}
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import paddleocr as _pocr  # noqa: F401
            self._available = True
        except ImportError:
            logger.debug("paddleocr is not installed.")
            self._available = False
        return self._available

    def _get_engine(self, language_hint: str):
        """Return (or create) a PaddleOCR instance for the requested lang."""
        paddle_lang = self._LANG_MAP.get(language_hint, "japan")
        if paddle_lang in self._engines:
            return self._engines[paddle_lang]

        from paddleocr import PaddleOCR

        logger.info("Initialising PaddleOCR for lang=%s ...", paddle_lang)
        engine = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            show_log=False,
        )
        self._engines[paddle_lang] = engine
        return engine

    def extract_text(
        self, image: np.ndarray, language_hint: str = "ja"
    ) -> OCRResult:
        if not self.is_available():
            return OCRResult.empty(self.name, error="paddleocr is not installed")

        try:
            engine = self._get_engine(language_hint)

            # PaddleOCR expects BGR or grayscale numpy arrays.
            # If the image has an alpha channel, drop it.
            img = image
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            results = engine.ocr(img, cls=True)

            if not results or not results[0]:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    language=language_hint,
                    engine_used=self.name,
                )

            # Aggregate text lines and compute average confidence
            lines: List[str] = []
            confidences: List[float] = []
            for line_info in results[0]:
                # Each line_info is (bbox, (text, confidence))
                if line_info and len(line_info) >= 2:
                    text_conf = line_info[1]
                    if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                        lines.append(str(text_conf[0]))
                        confidences.append(float(text_conf[1]))

            full_text = "\n".join(lines).strip()
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=language_hint,
                engine_used=self.name,
            )
        except Exception as exc:
            logger.warning("PaddleOCR extraction failed: %s", exc, exc_info=True)
            return OCRResult.empty(self.name, error=str(exc))


# ---------------------------------------------------------------------------
# Tesseract engine -- widely available fallback
# ---------------------------------------------------------------------------

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR via pytesseract -- universal fallback."""

    name = "tesseract"

    # Map common language hints to Tesseract language codes
    _LANG_MAP: Dict[str, str] = {
        "ja": "jpn",
        "zh": "chi_sim",
        "zh-cn": "chi_sim",
        "zh-tw": "chi_tra",
        "ko": "kor",
        "en": "eng",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "it": "ita",
        "pt": "por",
        "ru": "rus",
    }

    def __init__(self, tesseract_cmd: str = "") -> None:
        self._available: Optional[bool] = None
        self._tesseract_cmd = tesseract_cmd

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract

            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

            # Quick smoke-test: ask Tesseract for its version
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception:
            logger.debug("pytesseract / Tesseract is not available.")
            self._available = False
        return self._available

    def extract_text(
        self, image: np.ndarray, language_hint: str = "ja"
    ) -> OCRResult:
        if not self.is_available():
            return OCRResult.empty(self.name, error="pytesseract is not available")

        try:
            import pytesseract
            from PIL import Image

            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

            tess_lang = self._LANG_MAP.get(language_hint, "eng")

            # Convert to PIL
            if image.ndim == 2:
                pil_image = Image.fromarray(image, mode="L")
            elif image.ndim == 3 and image.shape[2] == 4:
                pil_image = Image.fromarray(image, mode="RGBA").convert("RGB")
            elif image.ndim == 3:
                pil_image = Image.fromarray(image, mode="RGB")
            else:
                return OCRResult.empty(
                    self.name,
                    error=f"Unsupported image shape: {image.shape}",
                )

            # Use image_to_data for per-word confidence
            data = pytesseract.image_to_data(
                pil_image,
                lang=tess_lang,
                output_type=pytesseract.Output.DICT,
            )

            words: List[str] = []
            confidences: List[float] = []
            for i, word in enumerate(data.get("text", [])):
                word = word.strip()
                if not word:
                    continue
                conf = float(data["conf"][i])
                # Tesseract uses -1 for non-text blocks
                if conf < 0:
                    continue
                words.append(word)
                confidences.append(conf / 100.0)  # normalise to 0-1

            full_text = " ".join(words).strip()
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=language_hint,
                engine_used=self.name,
            )
        except Exception as exc:
            logger.warning("Tesseract extraction failed: %s", exc, exc_info=True)
            return OCRResult.empty(self.name, error=str(exc))


# ---------------------------------------------------------------------------
# Multi-engine orchestrator
# ---------------------------------------------------------------------------

class OCREngine:
    """Multi-engine OCR with automatic fallback.

    The orchestrator tries the *primary* engine first.  If the result
    confidence is below ``confidence_threshold`` it walks through every
    other available engine and returns the result with the highest
    confidence.
    """

    def __init__(
        self,
        primary_engine: str = "manga-ocr",
        confidence_threshold: float = 0.7,
        tesseract_path: str = "",
        language_hint: str = "ja",
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._language_hint = language_hint

        # Build the engine registry (order matters for fallback priority)
        self._engines: Dict[str, BaseOCREngine] = {}
        self._register_engine(MangaOCREngine())
        self._register_engine(PaddleOCREngine())
        self._register_engine(TesseractEngine(tesseract_cmd=tesseract_path))

        if primary_engine not in self._engines:
            logger.warning(
                "Requested primary engine '%s' is not registered. "
                "Falling back to first available engine.",
                primary_engine,
            )
        self._primary_engine = primary_engine

        available = [n for n, e in self._engines.items() if e.is_available()]
        logger.info(
            "OCREngine initialised.  primary=%s  available=%s",
            self._primary_engine,
            available or "(none)",
        )
        if not available:
            logger.warning(
                "No OCR engines are available!  Install at least one of: "
                "manga-ocr, paddleocr, pytesseract."
            )

    # -- internal helpers ---------------------------------------------------

    def _register_engine(self, engine: BaseOCREngine) -> None:
        self._engines[engine.name] = engine

    @property
    def available_engines(self) -> List[str]:
        """Return names of engines whose dependencies are installed."""
        return [n for n, e in self._engines.items() if e.is_available()]

    def _fallback_order(self) -> List[BaseOCREngine]:
        """Return engines in fallback order: primary first, then the rest."""
        primary = self._engines.get(self._primary_engine)
        others = [
            e
            for name, e in self._engines.items()
            if name != self._primary_engine and e.is_available()
        ]
        ordered: List[BaseOCREngine] = []
        if primary is not None and primary.is_available():
            ordered.append(primary)
        ordered.extend(others)
        return ordered

    # -- public API ---------------------------------------------------------

    def extract_text(
        self, image: np.ndarray, language_hint: Optional[str] = None
    ) -> OCRResult:
        """Extract text with automatic engine fallback.

        1. Try the primary engine.
        2. If confidence < threshold, try every remaining available engine.
        3. Return the result with the highest confidence across all attempts.
        """
        hint = language_hint or self._language_hint
        engines = self._fallback_order()

        if not engines:
            return OCRResult.empty(
                engine="none",
                error="No OCR engines available. Install manga-ocr, paddleocr, or pytesseract.",
            )

        best: Optional[OCRResult] = None

        for engine in engines:
            logger.debug("Trying OCR engine: %s", engine.name)
            result = engine.extract_text(image, language_hint=hint)

            if best is None or result.confidence > best.confidence:
                best = result

            # If we already exceed the threshold, no need to try more engines
            if result.is_valid and result.confidence >= self._confidence_threshold:
                logger.debug(
                    "Engine '%s' returned text with confidence %.2f (>= %.2f threshold).",
                    engine.name,
                    result.confidence,
                    self._confidence_threshold,
                )
                return result

            logger.debug(
                "Engine '%s' confidence %.2f is below threshold %.2f; trying next.",
                engine.name,
                result.confidence,
                self._confidence_threshold,
            )

        # Return the best we found, even if below threshold
        assert best is not None  # engines list was non-empty
        logger.info(
            "Best OCR result from '%s' with confidence %.2f (below threshold %.2f).",
            best.engine_used,
            best.confidence,
            self._confidence_threshold,
        )
        return best

    def extract_text_batch(
        self,
        images: List[np.ndarray],
        language_hint: Optional[str] = None,
    ) -> List[OCRResult]:
        """Extract text from multiple image regions.

        Each image is processed independently through the fallback chain.
        """
        return [self.extract_text(img, language_hint=language_hint) for img in images]

    def detect_language(self, image: np.ndarray) -> str:
        """Detect the dominant script type from an image.

        Uses a simple heuristic based on character density and aspect ratios
        of connected components.  This is intentionally lightweight; for
        production accuracy consider a dedicated language-ID model.

        Returns one of: ``"ja"``, ``"zh"``, ``"ko"``, ``"en"``, ``"unknown"``.
        """
        try:
            gray = image
            if gray.ndim == 3:
                # Convert to grayscale via luminance
                gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                    np.uint8
                )

            h, w = gray.shape[:2]
            total_pixels = h * w
            if total_pixels == 0:
                return "unknown"

            # Binarise with a simple threshold (Otsu-like mean)
            threshold = int(gray.mean())
            dark_pixels = int(np.sum(gray < threshold))
            ink_ratio = dark_pixels / total_pixels

            # CJK scripts tend to have higher ink density than Latin scripts
            # because of complex strokes filling square character cells.
            # Very rough thresholds derived from typical manga bubbles:
            #   ink_ratio > 0.25  -> likely CJK
            #   ink_ratio < 0.12  -> likely Latin / sparse
            if ink_ratio > 0.25:
                # Try to distinguish ja / zh / ko by running a quick OCR
                # through whatever is available and checking the characters.
                quick = self.extract_text(image, language_hint="ja")
                if quick.is_valid:
                    return self._guess_cjk_language(quick.text)
                return "ja"  # default for manga context
            elif ink_ratio < 0.12:
                return "en"
            else:
                # Ambiguous -- attempt OCR-based detection
                quick = self.extract_text(image, language_hint="ja")
                if quick.is_valid:
                    return self._guess_cjk_language(quick.text)
                return "unknown"
        except Exception as exc:
            logger.warning("Language detection failed: %s", exc)
            return "unknown"

    @staticmethod
    def _guess_cjk_language(text: str) -> str:
        """Guess CJK language from character Unicode ranges.

        Simplified heuristic:
        - Hangul block -> Korean
        - Hiragana / Katakana presence -> Japanese
        - Otherwise assume Chinese
        """
        hangul = 0
        kana = 0  # hiragana + katakana
        cjk_ideograph = 0
        latin = 0

        for ch in text:
            cp = ord(ch)
            if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
                hangul += 1
            elif 0x3040 <= cp <= 0x30FF:
                kana += 1
            elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                cjk_ideograph += 1
            elif 0x0041 <= cp <= 0x007A:
                latin += 1

        total = hangul + kana + cjk_ideograph + latin
        if total == 0:
            return "unknown"

        if hangul / total > 0.3:
            return "ko"
        if kana / total > 0.15:
            return "ja"
        if cjk_ideograph / total > 0.3:
            return "zh"
        if latin / total > 0.5:
            return "en"
        return "unknown"
