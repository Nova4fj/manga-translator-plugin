"""Translation memory system with exact and fuzzy matching.

Stores previous translations in a SQLite database for reuse.
Supports exact lookup, fuzzy matching via SequenceMatcher,
context-aware ranking, and a terminology dictionary.
"""

import difflib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".manga-translator"
_DEFAULT_DB_NAME = "translation_memory.db"


@dataclass
class TMEntry:
    """A single translation memory entry."""
    id: Optional[int] = None
    source_text: str = ""
    target_text: str = ""
    source_lang: str = "ja"
    target_lang: str = "en"
    context: str = ""  # character name, scene, etc.
    series: str = ""
    quality_score: float = 1.0
    usage_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class TMMatch:
    """A match result from translation memory lookup."""
    entry: TMEntry
    similarity: float  # 0.0 to 1.0
    match_type: str  # "exact", "fuzzy", "term"


@dataclass
class TermEntry:
    """A terminology dictionary entry."""
    id: Optional[int] = None
    term: str = ""
    translation: str = ""
    category: str = ""  # character_name, place_name, technique, etc.
    series: str = ""
    notes: str = ""


class TranslationMemory:
    """SQLite-backed translation memory with fuzzy matching.

    Args:
        db_path: Path to SQLite database. If None, uses default location.
        fuzzy_threshold: Minimum similarity for fuzzy matches (0.0-1.0).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        fuzzy_threshold: float = 0.85,
    ):
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            self._db_path = str(_DEFAULT_DB_DIR / _DEFAULT_DB_NAME)
        else:
            self._db_path = db_path
        self.fuzzy_threshold = fuzzy_threshold
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    source_lang TEXT DEFAULT 'ja',
                    target_lang TEXT DEFAULT 'en',
                    context TEXT DEFAULT '',
                    series TEXT DEFAULT '',
                    quality_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    series TEXT DEFAULT '',
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_source
                ON entries (source_text, source_lang, target_lang)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_terms_term
                ON terms (term)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Translation entries
    # ------------------------------------------------------------------

    def add_entry(
        self,
        source_text: str,
        target_text: str,
        source_lang: str = "ja",
        target_lang: str = "en",
        context: str = "",
        series: str = "",
        quality_score: float = 1.0,
    ) -> TMEntry:
        """Add or update a translation memory entry.

        If an exact match (source_text + langs) already exists, updates it.
        """
        now = time.time()
        existing = self.lookup_exact(source_text, source_lang, target_lang)

        with self._connect() as conn:
            if existing:
                conn.execute("""
                    UPDATE entries SET target_text=?, context=?, series=?,
                    quality_score=?, updated_at=?
                    WHERE id=?
                """, (target_text, context, series, quality_score, now, existing.id))
                existing.target_text = target_text
                existing.updated_at = now
                return existing
            else:
                cursor = conn.execute("""
                    INSERT INTO entries
                    (source_text, target_text, source_lang, target_lang,
                     context, series, quality_score, usage_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """, (source_text, target_text, source_lang, target_lang,
                      context, series, quality_score, now, now))
                return TMEntry(
                    id=cursor.lastrowid,
                    source_text=source_text, target_text=target_text,
                    source_lang=source_lang, target_lang=target_lang,
                    context=context, series=series,
                    quality_score=quality_score,
                    created_at=now, updated_at=now,
                )

    def lookup_exact(
        self,
        source_text: str,
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> Optional[TMEntry]:
        """Find an exact match in the translation memory."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, source_text, target_text, source_lang, target_lang,
                       context, series, quality_score, usage_count,
                       created_at, updated_at
                FROM entries
                WHERE source_text=? AND source_lang=? AND target_lang=?
                ORDER BY quality_score DESC, updated_at DESC
                LIMIT 1
            """, (source_text, source_lang, target_lang)).fetchone()

        if row is None:
            return None
        return self._row_to_entry(row)

    def lookup(
        self,
        source_text: str,
        source_lang: str = "ja",
        target_lang: str = "en",
        context: str = "",
        max_results: int = 5,
    ) -> List[TMMatch]:
        """Look up matches including both exact and fuzzy.

        Returns matches sorted by relevance (exact first, then by similarity).
        """
        matches: List[TMMatch] = []

        # Exact match
        exact = self.lookup_exact(source_text, source_lang, target_lang)
        if exact:
            self._increment_usage(exact.id)
            matches.append(TMMatch(entry=exact, similarity=1.0, match_type="exact"))

        # Fuzzy matches
        fuzzy = self.lookup_fuzzy(
            source_text, source_lang, target_lang,
            max_results=max_results,
        )
        for m in fuzzy:
            if exact and m.entry.id == exact.id:
                continue  # skip duplicate
            matches.append(m)

        # Context boost: entries with matching context get a small boost
        if context:
            for m in matches:
                if m.entry.context and context.lower() in m.entry.context.lower():
                    m.similarity = min(m.similarity + 0.05, 1.0)

        matches.sort(key=lambda m: (-m.similarity, -m.entry.quality_score))
        return matches[:max_results]

    def lookup_fuzzy(
        self,
        source_text: str,
        source_lang: str = "ja",
        target_lang: str = "en",
        max_results: int = 5,
    ) -> List[TMMatch]:
        """Find fuzzy matches above the similarity threshold."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, source_text, target_text, source_lang, target_lang,
                       context, series, quality_score, usage_count,
                       created_at, updated_at
                FROM entries
                WHERE source_lang=? AND target_lang=?
            """, (source_lang, target_lang)).fetchall()

        matches: List[TMMatch] = []
        for row in rows:
            entry = self._row_to_entry(row)
            similarity = difflib.SequenceMatcher(
                None, source_text, entry.source_text
            ).ratio()
            if similarity >= self.fuzzy_threshold:
                matches.append(TMMatch(
                    entry=entry, similarity=similarity, match_type="fuzzy",
                ))

        matches.sort(key=lambda m: -m.similarity)
        return matches[:max_results]

    def _increment_usage(self, entry_id: int):
        """Increment the usage count for an entry."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE entries SET usage_count = usage_count + 1 WHERE id=?",
                (entry_id,),
            )

    def entry_count(self) -> int:
        """Return the total number of entries."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Terminology dictionary
    # ------------------------------------------------------------------

    def add_term(
        self,
        term: str,
        translation: str,
        category: str = "",
        series: str = "",
        notes: str = "",
    ) -> TermEntry:
        """Add a terminology entry."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO terms (term, translation, category, series, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (term, translation, category, series, notes))
            return TermEntry(
                id=cursor.lastrowid, term=term, translation=translation,
                category=category, series=series, notes=notes,
            )

    def lookup_term(self, term: str, series: str = "") -> Optional[TermEntry]:
        """Look up a terminology entry."""
        with self._connect() as conn:
            if series:
                row = conn.execute("""
                    SELECT id, term, translation, category, series, notes
                    FROM terms WHERE term=? AND series=?
                    LIMIT 1
                """, (term, series)).fetchone()
            else:
                row = conn.execute("""
                    SELECT id, term, translation, category, series, notes
                    FROM terms WHERE term=?
                    LIMIT 1
                """, (term,)).fetchone()

        if row is None:
            return None
        return TermEntry(
            id=row[0], term=row[1], translation=row[2],
            category=row[3], series=row[4], notes=row[5],
        )

    def list_terms(self, series: str = "") -> List[TermEntry]:
        """List all terminology entries, optionally filtered by series."""
        with self._connect() as conn:
            if series:
                rows = conn.execute(
                    "SELECT id, term, translation, category, series, notes FROM terms WHERE series=?",
                    (series,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, term, translation, category, series, notes FROM terms"
                ).fetchall()

        return [
            TermEntry(id=r[0], term=r[1], translation=r[2],
                      category=r[3], series=r[4], notes=r[5])
            for r in rows
        ]

    def term_count(self) -> int:
        """Return the total number of terminology entries."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM terms").fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> int:
        """Export the entire translation memory to JSON.

        Returns the number of entries exported.
        """
        with self._connect() as conn:
            entries = conn.execute("""
                SELECT source_text, target_text, source_lang, target_lang,
                       context, series, quality_score, usage_count
                FROM entries
            """).fetchall()
            terms = conn.execute("""
                SELECT term, translation, category, series, notes
                FROM terms
            """).fetchall()

        data = {
            "entries": [
                {
                    "source_text": r[0], "target_text": r[1],
                    "source_lang": r[2], "target_lang": r[3],
                    "context": r[4], "series": r[5],
                    "quality_score": r[6], "usage_count": r[7],
                }
                for r in entries
            ],
            "terms": [
                {
                    "term": r[0], "translation": r[1],
                    "category": r[2], "series": r[3], "notes": r[4],
                }
                for r in terms
            ],
        }

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return len(entries) + len(terms)

    def import_json(self, path: str) -> int:
        """Import translation memory from JSON.

        Returns the number of entries imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for entry in data.get("entries", []):
            self.add_entry(
                source_text=entry["source_text"],
                target_text=entry["target_text"],
                source_lang=entry.get("source_lang", "ja"),
                target_lang=entry.get("target_lang", "en"),
                context=entry.get("context", ""),
                series=entry.get("series", ""),
                quality_score=entry.get("quality_score", 1.0),
            )
            count += 1

        for term in data.get("terms", []):
            self.add_term(
                term=term["term"],
                translation=term["translation"],
                category=term.get("category", ""),
                series=term.get("series", ""),
                notes=term.get("notes", ""),
            )
            count += 1

        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row) -> TMEntry:
        return TMEntry(
            id=row[0], source_text=row[1], target_text=row[2],
            source_lang=row[3], target_lang=row[4],
            context=row[5], series=row[6],
            quality_score=row[7], usage_count=row[8],
            created_at=row[9], updated_at=row[10],
        )
