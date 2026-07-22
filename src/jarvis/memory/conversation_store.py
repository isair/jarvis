"""Persistent chat conversations for the modern UI (Phase 3A).

A browsable list of chat conversations + their message transcripts, stored in
the SAME profile ``jarvis.db`` as diary/learning/meals but in NEW, additive
tables. This is deliberately SEPARATE from:
  * ``DialogueMemory`` (conversation.py) — an in-memory rolling window used by
    the reply engine / diary; single-conversation, never persisted as text;
  * ``conversation_summaries`` — the LLM diary.

Self-contained by design (maximum safety for the shared DB): this module owns
its own DDL and creates its tables lazily via ``CREATE TABLE IF NOT EXISTS`` on
the shared connection. It does NOT modify ``db.py``, does NOT change any PRAGMA,
and does NOT rely on foreign-key enforcement — ``delete(hard=True)`` removes
child rows explicitly inside one transaction. It reuses the ``Database`` RLock so
its writes serialise with diary writes on the shared connection.

Idempotent + non-destructive: only ``CREATE IF NOT EXISTS`` and parameterized
INSERT/UPDATE. No DROP, no ALTER, no data deletion on init. A component-keyed
``schema_migrations`` row records the version so future migrations have a single
choke point (the backup helper is the forward story for any destructive step).

All queries are parameterized; ``search_titles`` escapes LIKE wildcards; message
content is never written to debug logs by default.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

try:  # debug_log is optional; never let logging break a store call
    from ..debug import debug_log
except Exception:  # pragma: no cover - defensive
    def debug_log(*_a, **_k):  # type: ignore
        return None

__all__ = ["ConversationStore", "CONVERSATIONS_SCHEMA_VERSION"]

CONVERSATIONS_SCHEMA_VERSION = 1

# Content is never emitted to debug logs by default (privacy first). Flip only
# for local forensic debugging.
_LOG_CONTENT = False

_MAX_METADATA_CHARS = 4096
_MAX_TITLE_CHARS = 60
_AUTO_TITLE_WORDS = 8
_LIKE_ESCAPE = "\\"

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
  id            TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  archived      INTEGER NOT NULL DEFAULT 0,
  deleted       INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  -- Monotonic activity counter, incremented by the store on every recency-
  -- bumping write. Ordering keys off this, NOT wall-clock, so recency is
  -- deterministic regardless of system-clock resolution.
  updated_seq   INTEGER NOT NULL DEFAULT 0
);
-- NOTE: the index on updated_seq is created in _ensure_schema AFTER the guarded
-- ADD COLUMN, so a legacy conversations table (missing updated_seq) cannot make
-- executescript() fail on "no such column".

CREATE TABLE IF NOT EXISTS conversation_messages (
  id              TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_id         TEXT,
  correlation_id  TEXT,
  role            TEXT NOT NULL
                    CHECK (role IN ('user','assistant','system')),
  source          TEXT
                    CHECK (source IS NULL OR source IN ('text','voice')),
  content         TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL DEFAULT 'completed'
                    CHECK (status IN ('pending','streaming','completed','cancelled','failed')),
  created_at      TEXT NOT NULL,
  tts_status      TEXT,
  error_code      TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_created
  ON conversation_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS schema_migrations (
  component  TEXT PRIMARY KEY,
  version    INTEGER NOT NULL,
  applied_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    # Microsecond resolution so updated_at is strictly increasing across
    # rapid operations (create/touch within the same second must still order
    # correctly). ISO-8601 UTC sorts lexically == chronologically.
    return datetime.now(timezone.utc).isoformat()


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so user input is matched literally. Use with
    ``LIKE ? ESCAPE '\\'`` and the pattern ``f"%{_escape_like(q)}%"``."""
    return (
        term.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


def _auto_title(text: str) -> str:
    """MVP auto-title: first words of the first user message. No model call.
    Unicode/diacritics-safe (character slicing), whitespace-collapsed,
    control-char stripped, length-capped, ``"New chat"`` fallback."""
    if not text:
        return "New chat"
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch == " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "New chat"
    words = cleaned.split(" ")[:_AUTO_TITLE_WORDS]
    title = " ".join(words)
    if len(title) > _MAX_TITLE_CHARS:
        title = title[:_MAX_TITLE_CHARS].rstrip() + "…"
    return title or "New chat"


def _validate_metadata(meta: Optional[dict]) -> str:
    """Return a bounded, canonical JSON object string; reject non-objects and
    oversized payloads (defends the row and any downstream ``json.loads``)."""
    if meta is None:
        return "{}"
    if not isinstance(meta, dict):
        raise ValueError("metadata must be a JSON object (dict)")
    encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _MAX_METADATA_CHARS:
        raise ValueError(f"metadata_json exceeds {_MAX_METADATA_CHARS} chars")
    return encoded


class ConversationStore:
    """CRUD over ``conversations`` / ``conversation_messages`` on the shared
    profile ``Database``. Separate from ``DialogueMemory`` and the diary."""

    def __init__(self, db) -> None:
        self._db = db
        # Reuse the Database RLock so writes serialise with diary writes on the
        # SAME connection. Fall back to a private lock for a bare-connection test.
        self._lock = getattr(db, "_lock", None) or threading.RLock()
        self._schema_ready = False

    # -- connection / schema ----------------------------------------------
    def _conn(self):
        return self._db.conn

    def _next_seq(self) -> int:
        """Monotonic activity counter (called under the lock). Independent of
        wall-clock resolution so recency ordering is always deterministic."""
        row = self._conn().execute(
            "SELECT COALESCE(MAX(updated_seq), 0) + 1 FROM conversations"
        ).fetchone()
        return int(row[0])

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if self._schema_ready:
                return
            conn = self._conn()
            conn.executescript(_DDL)
            # Idempotent additive migration: a conversations table created by an
            # earlier in-session build may lack updated_seq. Add it if missing
            # (ADD COLUMN is additive + non-destructive; never drops data).
            cols = {r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
            if "updated_seq" not in cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN updated_seq INTEGER NOT NULL DEFAULT 0"
                )
            # Build the updated_seq index only AFTER the column is guaranteed to
            # exist (fresh CREATE or legacy ADD COLUMN above).
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_active "
                "ON conversations(deleted, archived, updated_seq DESC)"
            )
            row = conn.execute(
                "SELECT version FROM schema_migrations WHERE component = 'conversations'"
            ).fetchone()
            current = int(row[0]) if row else 0
            if current < CONVERSATIONS_SCHEMA_VERSION:
                # v0 -> v1 is purely additive (tables ensured above). Record it.
                conn.execute(
                    """INSERT INTO schema_migrations (component, version, applied_at)
                       VALUES ('conversations', ?, ?)
                       ON CONFLICT(component) DO UPDATE SET
                         version = excluded.version, applied_at = excluded.applied_at""",
                    (CONVERSATIONS_SCHEMA_VERSION, _utc_now()),
                )
            conn.commit()
            self._schema_ready = True

    # -- conversations -----------------------------------------------------
    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        self._ensure_schema()
        cid = conversation_id or str(uuid.uuid4())
        now = _utc_now()
        meta = _validate_metadata(metadata)
        with self._lock:
            try:
                self._conn().execute(
                    """INSERT INTO conversations
                         (id, title, created_at, updated_at, archived, deleted, metadata_json, updated_seq)
                       VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
                    (cid, (title or "New chat")[:_MAX_TITLE_CHARS], now, now, meta, self._next_seq()),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        return cid

    def ensure_conversation(
        self, conversation_id: str, title_seed: Optional[str] = None
    ) -> str:
        """Idempotent create for a UI-minted id. Sets the auto-title from the
        first user message once; ``INSERT OR IGNORE`` means a later rename is
        never clobbered by subsequent turns."""
        self._ensure_schema()
        now = _utc_now()
        title = _auto_title(title_seed) if title_seed else "New chat"
        with self._lock:
            try:
                self._conn().execute(
                    """INSERT OR IGNORE INTO conversations
                         (id, title, created_at, updated_at, archived, deleted, metadata_json, updated_seq)
                       VALUES (?, ?, ?, ?, 0, 0, '{}', ?)""",
                    (conversation_id, title, now, now, self._next_seq()),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        self._ensure_schema()
        with self._lock:
            row = self._conn().execute(
                """SELECT id, title, created_at, updated_at, archived, deleted, metadata_json
                   FROM conversations WHERE id = ?""",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_conversations(
        self,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> List[dict]:
        self._ensure_schema()
        clauses = []
        if not include_deleted:
            clauses.append("deleted = 0")
        if not include_archived:
            clauses.append("archived = 0")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn().execute(
                f"""SELECT id, title, created_at, updated_at, archived, deleted
                    FROM conversations {where}
                    ORDER BY updated_seq DESC, rowid DESC
                    LIMIT ? OFFSET ?""",
                (int(limit), int(offset)),
            ).fetchall()
        return [dict(r) for r in rows]

    def select_active(self) -> Optional[str]:
        """Most-recently-updated live conversation (UI resume target)."""
        self._ensure_schema()
        with self._lock:
            row = self._conn().execute(
                """SELECT id FROM conversations
                   WHERE deleted = 0 AND archived = 0
                   ORDER BY updated_seq DESC, rowid DESC LIMIT 1"""
            ).fetchone()
        return row["id"] if row is not None else None

    def rename(self, conversation_id: str, new_title: str) -> bool:
        self._ensure_schema()
        title = re.sub(r"\s+", " ", (new_title or "")).strip()[:_MAX_TITLE_CHARS]
        if not title:
            raise ValueError("title must be non-empty")
        with self._lock:
            try:
                cur = self._conn().execute(
                    "UPDATE conversations SET title = ?, updated_at = ?, updated_seq = ? "
                    "WHERE id = ? AND deleted = 0",
                    (title, _utc_now(), self._next_seq(), conversation_id),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        return cur.rowcount > 0

    def touch(self, conversation_id: str, ts: Optional[str] = None) -> None:
        self._ensure_schema()
        with self._lock:
            try:
                self._conn().execute(
                    "UPDATE conversations SET updated_at = ?, updated_seq = ? WHERE id = ?",
                    (ts or _utc_now(), self._next_seq(), conversation_id),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise

    def clear_messages(self, conversation_id: str, confirm: bool) -> int:
        """Delete every message in a conversation (keeps the conversation row).
        Requires explicit confirmation from the caller."""
        if not confirm:
            raise ValueError("clear_messages requires explicit confirm=True")
        self._ensure_schema()
        with self._lock:
            try:
                self._conn().execute("BEGIN")
                cur = self._conn().execute(
                    "DELETE FROM conversation_messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self._conn().execute(
                    "UPDATE conversations SET updated_at = ?, updated_seq = ? WHERE id = ?",
                    (_utc_now(), self._next_seq(), conversation_id),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        return cur.rowcount

    def delete(self, conversation_id: str, confirm: bool, hard: bool = False) -> bool:
        """Delete-with-confirm. ``confirm=False`` refuses. Default is reversible
        soft-delete; ``hard=True`` removes rows (children deleted explicitly in
        one transaction, so correctness does not depend on FK enforcement)."""
        if not confirm:
            raise ValueError("delete requires explicit confirm=True")
        self._ensure_schema()
        with self._lock:
            try:
                self._conn().execute("BEGIN")
                if hard:
                    self._conn().execute(
                        "DELETE FROM conversation_messages WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    cur = self._conn().execute(
                        "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                    )
                else:
                    cur = self._conn().execute(
                        "UPDATE conversations SET deleted = 1, updated_at = ?, updated_seq = ? "
                        "WHERE id = ? AND deleted = 0",
                        (_utc_now(), self._next_seq(), conversation_id),
                    )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        return cur.rowcount > 0

    def search_titles(self, query: str, limit: int = 100) -> List[dict]:
        self._ensure_schema()
        pattern = f"%{_escape_like((query or '').strip())}%"
        with self._lock:
            rows = self._conn().execute(
                """SELECT id, title, updated_at FROM conversations
                   WHERE deleted = 0 AND title LIKE ? ESCAPE '\\'
                   ORDER BY updated_seq DESC, rowid DESC LIMIT ?""",
                (pattern, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- messages ----------------------------------------------------------
    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        after_created_at: Optional[str] = None,
    ) -> List[dict]:
        self._ensure_schema()
        clauses = ["conversation_id = ?"]
        params: List[Any] = [conversation_id]
        if after_created_at:
            clauses.append("created_at > ?")
            params.append(after_created_at)
        sql = (
            "SELECT id, conversation_id, turn_id, correlation_id, role, source, "
            "content, status, created_at, tts_status, error_code "
            "FROM conversation_messages WHERE " + " AND ".join(clauses) +
            " ORDER BY created_at ASC, rowid ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def append_user_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        source: str,
        turn_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        status: str = "completed",
        created_at: Optional[str] = None,
    ) -> str:
        return self._insert_message(
            conversation_id, message_id, "user", content, source,
            turn_id, correlation_id, status, created_at, None, None,
        )

    def append_or_update_assistant_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        status: str,
        turn_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        source: str = "text",
        tts_status: Optional[str] = None,
        error_code: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """UPSERT: insert a pending/streaming row, then finalize the SAME row
        (preserving ``created_at``) when the reply completes/fails."""
        self._ensure_schema()
        now = created_at or _utc_now()
        with self._lock:
            try:
                self._conn().execute("BEGIN")
                self._conn().execute(
                    """INSERT INTO conversation_messages
                         (id, conversation_id, turn_id, correlation_id, role, source,
                          content, status, created_at, tts_status, error_code)
                       VALUES (?, ?, ?, ?, 'assistant', ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         content    = excluded.content,
                         status     = excluded.status,
                         tts_status = excluded.tts_status,
                         error_code = excluded.error_code""",
                    (message_id, conversation_id, turn_id, correlation_id, source,
                     content, status, now, tts_status, error_code),
                )
                self._conn().execute(
                    "UPDATE conversations SET updated_at = ?, updated_seq = ? WHERE id = ?",
                    (_utc_now(), self._next_seq(), conversation_id),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        self._log("assistant", conversation_id, status)
        return message_id

    def _insert_message(
        self, conversation_id, message_id, role, content, source,
        turn_id, correlation_id, status, created_at, tts_status, error_code,
    ) -> str:
        self._ensure_schema()
        now = created_at or _utc_now()
        with self._lock:
            try:
                self._conn().execute("BEGIN")
                self._conn().execute(
                    """INSERT INTO conversation_messages
                         (id, conversation_id, turn_id, correlation_id, role, source,
                          content, status, created_at, tts_status, error_code)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (message_id, conversation_id, turn_id, correlation_id, role,
                     source, content, status, now, tts_status, error_code),
                )
                self._conn().execute(
                    "UPDATE conversations SET updated_at = ?, updated_seq = ? WHERE id = ?",
                    (_utc_now(), self._next_seq(), conversation_id),
                )
                self._conn().commit()
            except Exception:
                self._conn().rollback()
                raise
        self._log(role, conversation_id, status)
        return message_id

    def _log(self, role: str, conversation_id: str, status: str) -> None:
        if _LOG_CONTENT:
            return
        try:
            debug_log(
                f"conv_store: {role} msg -> {str(conversation_id)[:8]} ({status})",
                "memory",
            )
        except Exception:
            pass
