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
  id INTEGER PRIMARY KEY,
  vec FLOAT[768]
);

CREATE TABLE IF NOT EXISTS summary_vec (
  summary_id INTEGER PRIMARY KEY REFERENCES conversation_summaries(id) ON DELETE CASCADE,
  emb_id     INTEGER NOT NULL REFERENCES embeddings(id)
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

            # Use Python vector store if sqlite-vss is not available
            if not self.is_vss_enabled and self._python_vector_store and query_vec_json is not None and safe_q:
                # Parse query vector
                import json as _json
                query_vec = _json.loads(query_vec_json)
                
                # Get vector search results (use max of top_k*3 and 50 for good hybrid scoring)
                vector_search_limit = max(top_k * 3, 50)
                vector_results = self._python_vector_store.search(query_vec, top_k=vector_search_limit)
                
                # Get FTS results (use max of top_k*3 and 50 for good hybrid scoring)
                fts_search_limit = max(top_k * 3, 50)
                fts_sql = f"""
                SELECT s.id, bm25(summaries_fts) AS bm
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY bm
                LIMIT {fts_search_limit}
                """
                fts_rows = cur.execute(fts_sql, (safe_q,)).fetchall()
                fts_scores = {row['id']: row['bm'] for row in fts_rows}
                
                # Combine scores
                combined_scores = {}
                
                # Add vector scores (60% weight)
                for summary_id, distance in vector_results:
                    combined_scores[summary_id] = (1.0 / (1.0 + distance)) * 0.6
                
                # Add FTS scores (40% weight)
                for summary_id, bm_score in fts_scores.items():
                    if summary_id in combined_scores:
                        combined_scores[summary_id] += (1.0 / (1.0 + bm_score)) * 0.4
                    else:
                        combined_scores[summary_id] = (1.0 / (1.0 + bm_score)) * 0.4
                
                # Sort by combined score and fetch summaries
                sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                if sorted_ids:
                    # Fetch summaries for top results
                    placeholders = ','.join('?' * len(sorted_ids))
                    summary_sql = f"""
                    SELECT s.id, 
                           '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                           'summary' AS result_type
                    FROM conversation_summaries s
                    WHERE s.id IN ({placeholders})
                    """
                    rows = cur.execute(summary_sql, [sid for sid, _ in sorted_ids]).fetchall()
                    
                    # Create result rows with scores
                    results = []
                    id_to_score = {sid: score for sid, score in sorted_ids}
                    for row in rows:
                        # Create a new row dict with score
                        result = dict(row)
                        result['score'] = id_to_score.get(row['id'], 0.0)
                        results.append(result)
                    
                    # Sort by score again (in case DB returned in different order)
                    results.sort(key=lambda x: x['score'], reverse=True)
                    return results
                else:
                    return []
                    
            elif self.is_vss_enabled and query_vec_json is not None and safe_q:
                # Hybrid search: 60% vector similarity (semantic) + 40% FTS (exact terms)
                # This balances finding semantically related content with keyword matches
                # Use dynamic limits for efficiency on large datasets
                search_limit = max(top_k * 3, 50)
                summary_sql = f"""
                WITH fts_sum AS (
                  SELECT s.id, bm25(summaries_fts) AS bm
                  FROM summaries_fts
                  JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                  WHERE summaries_fts MATCH ?
                  ORDER BY bm LIMIT {search_limit}
                ),
                v_sum AS (
                  SELECT sv.summary_id AS id, distance
                  FROM vss_search(embeddings, 'vec', ?)
                  JOIN summary_vec sv ON sv.emb_id = rowid
                  LIMIT {search_limit}
                )
                SELECT s.id, (
                    (1.0/(1.0+COALESCE(v_sum.distance, 1))) * 0.6 +
                    (1.0/(1.0+COALESCE(fts_sum.bm, 10))) * 0.4
                  ) AS score,
                  '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                  'summary' AS result_type
                FROM conversation_summaries s
                LEFT JOIN v_sum     ON v_sum.id = s.id
                LEFT JOIN fts_sum   ON fts_sum.id = s.id
                WHERE v_sum.id IS NOT NULL OR fts_sum.id IS NOT NULL
                ORDER BY score DESC
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql, (safe_q, query_vec_json)).fetchall()

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

    @staticmethod
    def _pack_vector(vec: Sequence[float]) -> bytes:
        # SQLite-vss expects a float array; packing via array('f') ensures binary blob layout.
        import array
        arr = array.array('f', [float(x) for x in vec])
        return arr.tobytes()

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
    ) -> int:
        """Insert or update a conversation summary for a given date."""
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

    def upsert_summary_embedding(self, summary_id: int, vec: Sequence[float]) -> Optional[int]:
        """Store or update embedding for a conversation summary."""
        if self.is_vss_enabled:
            # Use sqlite-vss
            with self._lock:
                cur = self.conn.cursor()
                cur.execute("INSERT INTO embeddings(vec) VALUES (?)", (sqlite3.Binary(self._pack_vector(vec)),))
                emb_id = cur.lastrowid
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
