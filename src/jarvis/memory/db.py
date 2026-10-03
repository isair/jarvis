from __future__ import annotations
import sqlite3
import re
from typing import Sequence, Optional
from pathlib import Path
import threading
from datetime import datetime, timezone
from ..debug import debug_log

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Structured meals log (optional feature)
CREATE TABLE IF NOT EXISTS meals (
  id            INTEGER PRIMARY KEY,
  ts_utc        TEXT NOT NULL,
  source_app    TEXT NOT NULL,
  description   TEXT NOT NULL,
  calories_kcal REAL,
  protein_g     REAL,
  carbs_g       REAL,
  fat_g         REAL,
  fiber_g       REAL,
  sugar_g       REAL,
  sodium_mg     REAL,
  potassium_mg  REAL,
  micros_json   TEXT,
  confidence    REAL
);

-- Conversation summaries for diary/memory system
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id         INTEGER PRIMARY KEY,
  date_utc   TEXT NOT NULL,  -- YYYY-MM-DD format
  ts_utc     TEXT NOT NULL,  -- When summary was created
  summary    TEXT NOT NULL,  -- Concise summary of the day's conversations
  topics     TEXT,           -- Comma-separated list of main topics discussed
  source_app TEXT NOT NULL,  -- Source app that generated the conversation
  UNIQUE(date_utc, source_app)
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
  summary,
  topics,
  content='conversation_summaries',
  content_rowid='id',
  tokenize='porter'
);

-- Triggers for conversation summaries FTS
CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
"""

_VSS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vss0(
  vec(768) factory="Flat,IDMap2"
);

CREATE TABLE IF NOT EXISTS summary_vec (
  summary_id INTEGER PRIMARY KEY REFERENCES conversation_summaries(id) ON DELETE CASCADE,
  emb_id     INTEGER NOT NULL
);
"""


def _normalize_fts_query(raw: str) -> str:
    # Use improved fuzzy search query generation
    try:
        from .fuzzy_search import generate_flexible_fts_query
        flexible_query = generate_flexible_fts_query(raw)
        if flexible_query:
            return flexible_query
    except ImportError:
        pass
    
    # Fallback: Extract alphanumeric tokens and join them with spaces (logical AND)
    tokens = re.findall(r"[A-Za-z0-9_]+", raw)
    return " ".join(tokens)


def _reciprocal_rank_fusion(vector_rows, fts_rows):
    """Weighted RRF, with equal scores sharing a rank and absent hits scoring zero."""
    scores = {}
    for rows, weight in ((vector_rows, 0.6), (fts_rows, 0.4)):
        previous = None
        rank = 0
        for position, (summary_id, value) in enumerate(sorted(rows, key=lambda row: (row[1], row[0])), 1):
            if previous is None or value != previous:
                rank = position
            previous = value
            scores[summary_id] = scores.get(summary_id, 0.0) + weight / (60 + rank)
    return scores


class Database:
    def __init__(self, db_path: str, sqlite_vss_path: Optional[str] = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.is_vss_enabled = False
        self._python_vector_store = None
        
        if sqlite_vss_path:
            try:
                self.conn.enable_load_extension(True)
                self.conn.load_extension(sqlite_vss_path)
                self.is_vss_enabled = True
            except Exception:
                self.is_vss_enabled = False
        
        # If sqlite-vss is not available, use best available vector store (FAISS or Python fallback)
        if not self.is_vss_enabled:
            from ..utils.vector_store import get_best_vector_store
            self._python_vector_store = get_best_vector_store(db_path, dimension=768)
            
            # Log which vector store implementation is being used
            import sys
            store_type = type(self._python_vector_store).__name__
            if store_type == "FAISSVectorStore":
                debug_log("Using FAISS vector store for fast search", "jarvis")
            else:
                debug_log("Using Python fallback vector store", "jarvis")
        
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(_SCHEMA_SQL)
            if self.is_vss_enabled:
                cur.executescript(_VSS_SCHEMA_SQL)
            self.conn.commit()

    

    def search_hybrid(self, fts_query: str, query_vec_json: Optional[str], top_k: int = 8) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            safe_q = _normalize_fts_query(fts_query)

            # Fuse ranked candidates rather than mixing BM25 and distance scales.
            if query_vec_json is not None and safe_q and (self.is_vss_enabled or self._python_vector_store):
                import json as _json

                search_limit = max(top_k * 3, 50)
                fts_rows = cur.execute("""
                    SELECT s.id, bm25(summaries_fts) AS bm
                    FROM summaries_fts
                    JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                    WHERE summaries_fts MATCH ?
                    ORDER BY bm, s.id LIMIT ?
                """, (safe_q, search_limit)).fetchall()
                if self.is_vss_enabled:
                    vector_rows = cur.execute("""
                        SELECT sv.summary_id, candidates.distance
                        FROM (
                            SELECT rowid, distance FROM embeddings
                            WHERE vss_search(vec, vss_search_params(?, ?))
                        ) AS candidates
                        JOIN summary_vec sv ON sv.emb_id = candidates.rowid
                        ORDER BY candidates.distance, sv.summary_id
                    """, (query_vec_json, search_limit)).fetchall()
                else:
                    vector_rows = self._python_vector_store.search(
                        _json.loads(query_vec_json), top_k=search_limit,
                    )

                scores = _reciprocal_rank_fusion(vector_rows, [(r['id'], r['bm']) for r in fts_rows])
                # Fetch before limiting so stale embedding IDs cannot crowd out live memories.
                if not scores:
                    return []
                placeholders = ','.join('?' for _ in scores)
                rows = cur.execute(f"""
                    SELECT s.id,
                           '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                           'summary' AS result_type
                    FROM conversation_summaries s WHERE s.id IN ({placeholders})
                """, list(scores)).fetchall()
                results = [dict(row, score=scores[row['id']]) for row in rows]
                results.sort(key=lambda row: (-row['score'], row['id']))
                debug_log(f"Hybrid memory search: {len(vector_rows)} vector, {len(fts_rows)} keyword candidates", "jarvis")
                return results[:top_k]

            elif safe_q:
                # FTS-only search over conversation summaries
                summary_sql = f"""
                SELECT s.id, bm25(summaries_fts) AS score,
                       '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY score
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql, (safe_q,)).fetchall()

            else:
                # Fallback: latest conversation summaries
                summary_sql = f"""
                SELECT id, 0.0 AS score,
                       '[' || date_utc || '] ' || summary || ' (Topics: ' || COALESCE(topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM conversation_summaries
                ORDER BY date_utc DESC
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql).fetchall()

            return rows

    # --- Meals API ---
    def insert_meal(
        self,
        ts_utc: str,
        source_app: str,
        description: str,
        calories_kcal: Optional[float] = None,
        protein_g: Optional[float] = None,
        carbs_g: Optional[float] = None,
        fat_g: Optional[float] = None,
        fiber_g: Optional[float] = None,
        sugar_g: Optional[float] = None,
        sodium_mg: Optional[float] = None,
        potassium_mg: Optional[float] = None,
        micros_json: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO meals(ts_utc, source_app, description, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, potassium_mg, micros_json, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    source_app,
                    description,
                    calories_kcal,
                    protein_g,
                    carbs_g,
                    fat_g,
                    fiber_g,
                    sugar_g,
                    sodium_mg,
                    potassium_mg,
                    micros_json,
                    confidence,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_meals_between(self, ts_utc_min: str, ts_utc_max: str) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM meals
                WHERE ts_utc >= ? AND ts_utc <= ?
                ORDER BY ts_utc ASC
                """,
                (ts_utc_min, ts_utc_max),
            ).fetchall()
            return rows

    def delete_meal(self, meal_id: int) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # --- Conversation Summaries API ---
    def upsert_conversation_summary(
        self,
        date_utc: str,  # YYYY-MM-DD format
        summary: str,
        topics: Optional[str] = None,
        source_app: str = "jarvis",
        ts_utc: Optional[str] = None,
    ) -> int:
        """Insert or update a conversation summary for a given date.

        ``ts_utc`` defaults to "now". Maintenance ops that rewrite an
        existing row's content without changing what it represents (e.g.
        the deflection scrub bulk sweep) should pass through the row's
        original ``ts_utc`` so the audit trail is preserved.
        """
        if ts_utc is None:
            ts_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO conversation_summaries(date_utc, ts_utc, summary, topics, source_app)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date_utc, ts_utc, summary, topics, source_app),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_conversation_summary(self, date_utc: str, source_app: str = "jarvis") -> Optional[sqlite3.Row]:
        """Get conversation summary for a specific date."""
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc = ? AND source_app = ?
                """,
                (date_utc, source_app),
            ).fetchone()
            return row

    def get_recent_conversation_summaries(self, days: int = 7) -> list[sqlite3.Row]:
        """Get conversation summaries from the last N days."""
        from datetime import datetime, timedelta, timezone
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc >= ?
                ORDER BY date_utc DESC
                """,
                (cutoff_date,),
            ).fetchall()
            return rows

    def get_all_conversation_summaries(self) -> list[sqlite3.Row]:
        """Get all conversation summaries, ordered by date ascending (oldest first).

        Used for bulk import into graph memory — processes diary entries
        chronologically so the graph builds up naturally.
        """
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                ORDER BY date_utc ASC
                """,
            ).fetchall()
            return rows

    def upsert_summary_embedding(self, summary_id: int, vec: Sequence[float]) -> Optional[int]:
        """Store or update embedding for a conversation summary."""
        if self.is_vss_enabled:
            # Use sqlite-vss
            import json
            with self._lock:
                cur = self.conn.cursor()
                emb_id = cur.execute("SELECT COALESCE(MAX(rowid), 0) + 1 FROM embeddings").fetchone()[0]
                cur.execute("INSERT INTO embeddings(rowid, vec) VALUES (?, ?)", (emb_id, json.dumps([float(x) for x in vec])))
                cur.execute(
                    "INSERT OR REPLACE INTO summary_vec(summary_id, emb_id) VALUES (?, ?)",
                    (summary_id, emb_id),
                )
                self.conn.commit()
                return int(emb_id)
        elif self._python_vector_store:
            # Use Python vector store
            self._python_vector_store.add_vector(summary_id, list(vec))
            return summary_id  # Return summary_id as a placeholder for emb_id
        else:
            return None

    def close(self) -> None:
        try:
            with self._lock:
                self.conn.close()
        except Exception:
            pass
