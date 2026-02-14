"""OCR result cache -- avoids re-running OCR on identical image regions."""

import hashlib
import logging
import sqlite3
from typing import Optional

import numpy as np

from manga_translator.components.ocr_engine import OCRResult

logger = logging.getLogger(__name__)


class OCRCache:
    """SQLite-backed cache for OCR results, keyed by image region hash."""

    def __init__(self, db_path: str = "ocr_cache.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_cache (
                region_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                language TEXT NOT NULL,
                engine_used TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    @staticmethod
    def hash_region(image: np.ndarray) -> str:
        """Compute a hash of an image region for cache lookup."""
        return hashlib.sha256(image.tobytes()).hexdigest()

    def lookup(self, image: np.ndarray) -> Optional[OCRResult]:
        """Look up cached OCR result for an image region."""
        h = self.hash_region(image)
        row = self._conn.execute(
            "SELECT text, confidence, language, engine_used FROM ocr_cache WHERE region_hash = ?",
            (h,),
        ).fetchone()
        if row:
            logger.debug("OCR cache hit: %s", h[:12])
            return OCRResult(
                text=row[0],
                confidence=row[1],
                language=row[2],
                engine_used=f"cached:{row[3]}",
            )
        return None

    def store(self, image: np.ndarray, result: OCRResult) -> None:
        """Store an OCR result in the cache."""
        h = self.hash_region(image)
        self._conn.execute(
            "INSERT OR REPLACE INTO ocr_cache (region_hash, text, confidence, language, engine_used) VALUES (?, ?, ?, ?, ?)",
            (h, result.text, result.confidence, result.language, result.engine_used),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Clear all cached OCR results."""
        self._conn.execute("DELETE FROM ocr_cache")
        self._conn.commit()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return self._conn.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()[0]

    def close(self):
        """Close the database connection."""
        self._conn.close()
