"""Tests for manga_translator.ocr_cache."""

import numpy as np
import pytest

from manga_translator.components.ocr_engine import OCRResult
from manga_translator.ocr_cache import OCRCache


@pytest.fixture
def cache(tmp_path):
    """Create an OCRCache backed by a temporary database."""
    db_path = str(tmp_path / "test_ocr_cache.db")
    c = OCRCache(db_path=db_path)
    yield c
    c.close()


def _make_result(text="hello", confidence=0.95, language="ja", engine="manga-ocr"):
    return OCRResult(text=text, confidence=confidence, language=language, engine_used=engine)


class TestOCRCache:
    def test_store_and_lookup(self, cache):
        """Stored OCR result can be retrieved by the same image."""
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        result = _make_result()
        cache.store(image, result)

        cached = cache.lookup(image)
        assert cached is not None
        assert cached.text == result.text
        assert cached.confidence == result.confidence
        assert cached.language == result.language
        assert cached.engine_used == f"cached:{result.engine_used}"

    def test_cache_miss(self, cache):
        """Lookup on an image not in the cache returns None."""
        image = np.ones((16, 16), dtype=np.uint8) * 128
        assert cache.lookup(image) is None

    def test_hash_region_deterministic(self):
        """Same image data always produces the same hash."""
        image = np.random.RandomState(42).randint(0, 256, (20, 20, 3), dtype=np.uint8)
        h1 = OCRCache.hash_region(image)
        h2 = OCRCache.hash_region(image)
        assert h1 == h2

    def test_hash_region_different(self):
        """Different images produce different hashes."""
        img_a = np.zeros((10, 10), dtype=np.uint8)
        img_b = np.ones((10, 10), dtype=np.uint8) * 255
        assert OCRCache.hash_region(img_a) != OCRCache.hash_region(img_b)

    def test_clear(self, cache):
        """Clear removes all cached entries."""
        image = np.zeros((8, 8), dtype=np.uint8)
        cache.store(image, _make_result())
        assert cache.size == 1

        cache.clear()
        assert cache.size == 0
        assert cache.lookup(image) is None

    def test_size(self, cache):
        """Size reflects the number of stored entries."""
        assert cache.size == 0

        for i in range(3):
            img = np.full((8, 8), fill_value=i, dtype=np.uint8)
            cache.store(img, _make_result(text=f"text_{i}"))

        assert cache.size == 3
