"""Multi-backend translation engine for manga text."""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator that retries a function with exponential backoff.

    Handles rate limits (HTTP 429), network timeouts, and transient failures.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    # Check for rate limit hint in the exception
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    error_str = str(e).lower()
                    if "429" in error_str or "rate" in error_str:
                        delay = min(delay * 2, max_delay)
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s. Retrying in %.1fs",
                        attempt + 1, max_retries + 1, func.__name__, e, delay,
                    )
                    time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


@dataclass
class TranslationResult:
    """Result from a translation operation."""

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    engine_used: str
    confidence: float  # 0.0 to 1.0
    error: Optional[str] = None


class BaseTranslationEngine(ABC):
    """Abstract base for translation engines."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is ready to use."""
        ...

    @abstractmethod
    def translate(
        self, text: str, source_lang: str = "ja", target_lang: str = "en"
    ) -> TranslationResult:
        """Translate a single text string."""
        ...

    def translate_batch(
        self, texts: List[str], source_lang: str = "ja", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple texts. Override for batch-optimized engines."""
        return [self.translate(t, source_lang, target_lang) for t in texts]


class OpenAIEngine(BaseTranslationEngine):
    """OpenAI GPT-based translation with context awareness."""

    name = "openai"

    def __init__(self, api_key: str = "", model: str = "gpt-4",
                 context_prompt: str = ""):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._context_prompt = context_prompt or (
            "You are a professional manga translator. Translate the following "
            "Japanese manga dialogue into natural, conversational English. "
            "Preserve the tone and emotion of the original text. "
            "Return ONLY the translated text, nothing else."
        )
        self._client = None

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _api_call(self, messages, max_tokens=1000):
        """Make an API call with retry logic."""
        client = self._get_client()
        return client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )

    def translate(
        self, text: str, source_lang: str = "ja", target_lang: str = "en"
    ) -> TranslationResult:
        try:
            response = self._api_call(
                messages=[
                    {"role": "system", "content": self._context_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=1000,
            )
            translated = response.choices[0].message.content.strip()
            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.9,
            )
        except Exception as e:
            logger.error("OpenAI translation failed: %s", e)
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.0,
                error=str(e),
            )

    def translate_batch(
        self, texts: List[str], source_lang: str = "ja", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Batch translate by sending all texts in one prompt for efficiency."""
        if len(texts) <= 1:
            return [self.translate(t, source_lang, target_lang) for t in texts]

        try:
            numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
            batch_prompt = (
                f"Translate each numbered line from {source_lang} to {target_lang}. "
                f"Return each translation on its own line with the same numbering.\n\n"
                f"{numbered}"
            )
            response = self._api_call(
                messages=[
                    {"role": "system", "content": self._context_prompt},
                    {"role": "user", "content": batch_prompt},
                ],
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            lines = raw.split("\n")

            results = []
            for i, text in enumerate(texts):
                translated = ""
                tag = f"[{i+1}]"
                for line in lines:
                    if line.strip().startswith(tag):
                        translated = line.strip()[len(tag):].strip()
                        break

                if not translated and i < len(lines):
                    translated = lines[i].strip()
                    # Strip numbering if present
                    for prefix in [f"{i+1}.", f"{i+1})", tag]:
                        if translated.startswith(prefix):
                            translated = translated[len(prefix):].strip()
                            break

                results.append(TranslationResult(
                    source_text=text,
                    translated_text=translated,
                    source_language=source_lang,
                    target_language=target_lang,
                    engine_used=self.name,
                    confidence=0.85 if translated else 0.0,
                ))
            return results

        except Exception as e:
            logger.error("OpenAI batch translation failed, falling back to individual: %s", e)
            return [self.translate(t, source_lang, target_lang) for t in texts]


class DeepLEngine(BaseTranslationEngine):
    """DeepL API translation."""

    name = "deepl"

    LANG_MAP = {
        "ja": "JA",
        "zh": "ZH",
        "ko": "KO",
        "en": "EN-US",
        "de": "DE",
        "fr": "FR",
        "es": "ES",
    }

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("DEEPL_AUTH_KEY", "")
        self._translator = None

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import deepl  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_translator(self):
        if self._translator is None:
            import deepl
            self._translator = deepl.Translator(self._api_key)
        return self._translator

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _api_call(self, texts, src, tgt):
        """Make a DeepL API call with retry logic."""
        translator = self._get_translator()
        return translator.translate_text(texts, source_lang=src, target_lang=tgt)

    def translate(
        self, text: str, source_lang: str = "ja", target_lang: str = "en"
    ) -> TranslationResult:
        try:
            src = self.LANG_MAP.get(source_lang, source_lang.upper())
            tgt = self.LANG_MAP.get(target_lang, target_lang.upper())
            result = self._api_call(text, src, tgt)
            return TranslationResult(
                source_text=text,
                translated_text=result.text,
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.85,
            )
        except Exception as e:
            logger.error("DeepL translation failed: %s", e)
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.0,
                error=str(e),
            )

    def translate_batch(
        self, texts: List[str], source_lang: str = "ja", target_lang: str = "en"
    ) -> List[TranslationResult]:
        try:
            src = self.LANG_MAP.get(source_lang, source_lang.upper())
            tgt = self.LANG_MAP.get(target_lang, target_lang.upper())
            results = self._api_call(texts, src, tgt)

            return [
                TranslationResult(
                    source_text=texts[i],
                    translated_text=r.text,
                    source_language=source_lang,
                    target_language=target_lang,
                    engine_used=self.name,
                    confidence=0.85,
                )
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error("DeepL batch failed, falling back to individual: %s", e)
            return [self.translate(t, source_lang, target_lang) for t in texts]


class ArgosEngine(BaseTranslationEngine):
    """Argos Translate for offline translation."""

    name = "argos"

    def __init__(self):
        self._installed_packages = None

    def is_available(self) -> bool:
        try:
            import argostranslate.translate  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_language_package(self, source_lang: str, target_lang: str) -> bool:
        """Check/install the required language package."""
        try:
            import argostranslate.package
            import argostranslate.translate

            languages = argostranslate.translate.get_installed_languages()
            src_langs = [l for l in languages if l.code == source_lang]
            tgt_langs = [l for l in languages if l.code == target_lang]

            if src_langs and tgt_langs:
                translation = src_langs[0].get_translation(tgt_langs[0])
                if translation:
                    return True

            # Try to install
            logger.info("Installing Argos language package %s→%s", source_lang, target_lang)
            argostranslate.package.update_package_index()
            packages = argostranslate.package.get_available_packages()
            pkg = next(
                (p for p in packages
                 if p.from_code == source_lang and p.to_code == target_lang),
                None,
            )
            if pkg:
                argostranslate.package.install_from_path(pkg.download())
                return True
            return False
        except Exception as e:
            logger.error("Argos package setup failed: %s", e)
            return False

    def translate(
        self, text: str, source_lang: str = "ja", target_lang: str = "en"
    ) -> TranslationResult:
        try:
            if not self._ensure_language_package(source_lang, target_lang):
                return TranslationResult(
                    source_text=text,
                    translated_text="",
                    source_language=source_lang,
                    target_language=target_lang,
                    engine_used=self.name,
                    confidence=0.0,
                    error=f"Language package {source_lang}→{target_lang} not available",
                )

            import argostranslate.translate
            translated = argostranslate.translate.translate(text, source_lang, target_lang)

            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.65,
            )
        except Exception as e:
            logger.error("Argos translation failed: %s", e)
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                engine_used=self.name,
                confidence=0.0,
                error=str(e),
            )


class TranslationManager:
    """Multi-engine translation manager with automatic fallback."""

    def __init__(
        self,
        primary_engine: str = "deepl",
        openai_api_key: str = "",
        openai_model: str = "gpt-4",
        deepl_api_key: str = "",
        context_prompt: str = "",
    ):
        self._engines: Dict[str, BaseTranslationEngine] = {}
        self._primary = primary_engine

        # Register engines
        self._engines["openai"] = OpenAIEngine(
            api_key=openai_api_key, model=openai_model, context_prompt=context_prompt
        )
        self._engines["deepl"] = DeepLEngine(api_key=deepl_api_key)
        self._engines["argos"] = ArgosEngine()

        # Log availability
        for name, engine in self._engines.items():
            avail = engine.is_available()
            logger.info("Translation engine %s: %s", name, "available" if avail else "not available")

    @property
    def available_engines(self) -> List[str]:
        return [name for name, eng in self._engines.items() if eng.is_available()]

    def _get_engine_order(self) -> List[BaseTranslationEngine]:
        """Get engines in priority order: primary first, then others."""
        order = []
        if self._primary in self._engines and self._engines[self._primary].is_available():
            order.append(self._engines[self._primary])

        for name, engine in self._engines.items():
            if name != self._primary and engine.is_available():
                order.append(engine)

        return order

    def translate(
        self, text: str, source_lang: str = "ja", target_lang: str = "en"
    ) -> TranslationResult:
        """Translate text with automatic engine fallback."""
        if not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                engine_used="none",
                confidence=0.0,
            )

        # Same-language passthrough
        if source_lang == target_lang:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                engine_used="passthrough",
                confidence=1.0,
            )

        engines = self._get_engine_order()
        if not engines:
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                engine_used="none",
                confidence=0.0,
                error="No translation engines available. Configure API keys or install argostranslate.",
            )

        best_result = None
        for engine in engines:
            result = engine.translate(text, source_lang, target_lang)
            if result.confidence > 0.5 and result.translated_text:
                return result
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result

    def translate_batch(
        self, texts: List[str], source_lang: str = "ja", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple texts with the best available engine."""
        if not texts:
            return []

        engines = self._get_engine_order()
        if not engines:
            return [
                TranslationResult(
                    source_text=t,
                    translated_text="",
                    source_language=source_lang,
                    target_language=target_lang,
                    engine_used="none",
                    confidence=0.0,
                    error="No translation engines available.",
                )
                for t in texts
            ]

        for engine in engines:
            try:
                results = engine.translate_batch(texts, source_lang, target_lang)
                # Check if results are usable
                avg_confidence = sum(r.confidence for r in results) / len(results)
                if avg_confidence > 0.3:
                    return results
            except Exception as e:
                logger.warning("Batch translation with %s failed: %s", engine.name, e)
                continue

        # Last resort: translate individually with fallback
        return [self.translate(t, source_lang, target_lang) for t in texts]
