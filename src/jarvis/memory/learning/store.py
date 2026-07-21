"""SQLite persistence for learning lessons — same profile Database connection."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from .types import (
    CONFIDENCE_CAPS,
    NAMESPACE_FOR_TYPE,
    Lesson,
    LessonStatus,
    LessonType,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_key(key: str) -> str:
    return " ".join((key or "").strip().lower().split())


class LearningStore:
    """CRUD over ``learning_lessons`` using an existing ``Database`` or raw path."""

    def __init__(self, db) -> None:
        # Accept jarvis.memory.db.Database or a sqlite3 connection-like wrapper.
        self._db = db
        self._lock = threading.RLock()

    def _conn(self) -> sqlite3.Connection:
        return self._db.conn

    def _ensure_schema(self) -> None:
        # Schema is created by Database._init_schema; no-op if already present.
        # For tests that open a bare connection, create tables defensively.
        with self._lock:
            cur = self._conn().cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='learning_lessons'"
            )
            if cur.fetchone():
                return
            # Re-run additive fragment via Database if available.
            if hasattr(self._db, "_init_schema"):
                self._db._init_schema()

    def was_conversation_processed(self, conversation_id: str) -> bool:
        self._ensure_schema()
        with self._lock:
            row = self._conn().execute(
                "SELECT 1 FROM learning_processed_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return row is not None

    def mark_conversation_processed(self, conversation_id: str, lesson_count: int) -> None:
        self._ensure_schema()
        with self._lock:
            self._conn().execute(
                """
                INSERT OR REPLACE INTO learning_processed_conversations
                  (conversation_id, processed_at, lesson_count)
                VALUES (?, ?, ?)
                """,
                (conversation_id, _utc_now(), int(lesson_count)),
            )
            self._conn().commit()

    def get_active_by_key(self, lesson_type: str, subject_key: str) -> List[Lesson]:
        self._ensure_schema()
        key = _normalize_key(subject_key)
        with self._lock:
            rows = self._conn().execute(
                """
                SELECT * FROM learning_lessons
                WHERE status = 'active' AND lesson_type = ? AND subject_key = ?
                ORDER BY updated_at DESC
                """,
                (lesson_type, key),
            ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def find_duplicate(self, lesson_type: str, subject_key: str, value: str) -> Optional[Lesson]:
        self._ensure_schema()
        key = _normalize_key(subject_key)
        val = (value or "").strip()
        with self._lock:
            row = self._conn().execute(
                """
                SELECT * FROM learning_lessons
                WHERE status = 'active' AND lesson_type = ? AND subject_key = ?
                  AND lower(value) = lower(?)
                LIMIT 1
                """,
                (lesson_type, key, val),
            ).fetchone()
        return self._row_to_lesson(row) if row else None

    def upsert_lesson(self, lesson: Lesson) -> Lesson:
        """Insert or update respecting dedupe / contradiction rules."""
        self._ensure_schema()
        lesson.subject_key = _normalize_key(lesson.subject_key)
        lesson.value = (lesson.value or "").strip()
        lesson.confidence = min(
            float(lesson.confidence),
            float(CONFIDENCE_CAPS.get(lesson.provenance, 0.95)),
        )
        if not lesson.namespace:
            try:
                lt = LessonType(lesson.lesson_type)
                lesson.namespace = NAMESPACE_FOR_TYPE[lt].value
            except Exception:
                lesson.namespace = "profile"

        now = _utc_now()
        with self._lock:
            dup = self.find_duplicate(lesson.lesson_type, lesson.subject_key, lesson.value)
            if dup:
                self._conn().execute(
                    """
                    UPDATE learning_lessons
                    SET source_quote = ?, conversation_id = ?, turn_id = ?,
                        updated_at = ?, confidence = ?,
                        occurrence_count = occurrence_count + 1
                    WHERE id = ?
                    """,
                    (
                        lesson.source_quote,
                        lesson.conversation_id,
                        lesson.turn_id,
                        now,
                        max(float(dup.confidence), float(lesson.confidence)),
                        dup.id,
                    ),
                )
                self._conn().commit()
                dup.source_quote = lesson.source_quote
                dup.updated_at = now
                dup.occurrence_count += 1
                return dup

            # Same key, different value → supersede older actives of same type+key
            # when provenance is a direct user correction/preference.
            if lesson.provenance in ("user_explicit_correction", "user_direct"):
                self._conn().execute(
                    """
                    UPDATE learning_lessons
                    SET status = 'superseded', updated_at = ?
                    WHERE status = 'active'
                      AND lesson_type = ? AND subject_key = ?
                      AND lower(value) != lower(?)
                    """,
                    (now, lesson.lesson_type, lesson.subject_key, lesson.value),
                )

            if not lesson.id:
                lesson.id = str(uuid.uuid4())
            if not lesson.created_at:
                lesson.created_at = now
            lesson.updated_at = now

            self._conn().execute(
                """
                INSERT INTO learning_lessons (
                  id, lesson_type, subject_key, value, source_quote,
                  conversation_id, turn_id, created_at, updated_at,
                  confidence, sensitivity, status, expires_at,
                  namespace, occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson.id,
                    lesson.lesson_type,
                    lesson.subject_key,
                    lesson.value,
                    lesson.source_quote,
                    lesson.conversation_id,
                    lesson.turn_id,
                    lesson.created_at,
                    lesson.updated_at,
                    lesson.confidence,
                    lesson.sensitivity,
                    lesson.status,
                    lesson.expires_at,
                    lesson.namespace,
                    lesson.occurrence_count,
                ),
            )
            self._conn().commit()
            return lesson

    def increment_recurring_failure(self, subject_key: str, value: str, quote: str,
                                    conversation_id: str, turn_id: str,
                                    min_recurrences: int = 3) -> Optional[Lesson]:
        """Bump recurring_failure; emit improvement_candidate only at threshold."""
        from .types import LessonType, Namespace

        self._ensure_schema()
        key = _normalize_key(subject_key)
        now = _utc_now()
        with self._lock:
            row = self._conn().execute(
                """
                SELECT * FROM learning_lessons
                WHERE status = 'active' AND lesson_type = ?
                  AND subject_key = ? AND lower(value) = lower(?)
                LIMIT 1
                """,
                (LessonType.RECURRING_FAILURE.value, key, value.strip()),
            ).fetchone()
            if row:
                self._conn().execute(
                    """
                    UPDATE learning_lessons
                    SET occurrence_count = occurrence_count + 1,
                        updated_at = ?, source_quote = ?,
                        conversation_id = ?, turn_id = ?
                    WHERE id = ?
                    """,
                    (now, quote, conversation_id, turn_id, row["id"]),
                )
                self._conn().commit()
                count = int(row["occurrence_count"]) + 1
                lesson = self._row_to_lesson(
                    self._conn().execute(
                        "SELECT * FROM learning_lessons WHERE id = ?", (row["id"],)
                    ).fetchone()
                )
            else:
                lesson = Lesson(
                    id=str(uuid.uuid4()),
                    lesson_type=LessonType.RECURRING_FAILURE.value,
                    subject_key=key,
                    value=value.strip(),
                    source_quote=quote,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    created_at=now,
                    updated_at=now,
                    confidence=0.80,
                    namespace=Namespace.IMPROVEMENTS.value,
                    occurrence_count=1,
                    provenance="tool_confirmed",
                )
                self.upsert_lesson(lesson)
                count = 1

            if count >= int(min_recurrences):
                # Create improvement_candidate once (dedupe by key+value).
                cand = Lesson(
                    id=str(uuid.uuid4()),
                    lesson_type=LessonType.IMPROVEMENT_CANDIDATE.value,
                    subject_key=key,
                    value=f"Proposed improvement after {count} similar failures: {value.strip()}",
                    source_quote=quote,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    created_at=now,
                    updated_at=now,
                    confidence=0.80,
                    namespace=Namespace.IMPROVEMENTS.value,
                    occurrence_count=count,
                    provenance="tool_confirmed",
                    sensitivity="proposal_only",
                )
                return self.upsert_lesson(cand)
            return lesson

    def soft_delete(self, lesson_id: str) -> bool:
        self._ensure_schema()
        with self._lock:
            cur = self._conn().execute(
                """
                UPDATE learning_lessons
                SET status = 'deleted', updated_at = ?
                WHERE id = ? AND status != 'deleted'
                """,
                (_utc_now(), lesson_id),
            )
            self._conn().commit()
            return cur.rowcount > 0

    def soft_delete_by_subject(self, subject_key: str) -> List[str]:
        """Logical delete all active lessons matching subject; return ids."""
        self._ensure_schema()
        key = _normalize_key(subject_key)
        with self._lock:
            rows = self._conn().execute(
                """
                SELECT id FROM learning_lessons
                WHERE status = 'active' AND subject_key = ?
                """,
                (key,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._conn().execute(
                    f"""
                    UPDATE learning_lessons
                    SET status = 'deleted', updated_at = ?
                    WHERE id IN ({','.join('?' * len(ids))})
                    """,
                    [_utc_now(), *ids],
                )
                self._conn().commit()
            return ids

    def list_active(
        self,
        namespaces: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[Lesson]:
        self._ensure_schema()
        with self._lock:
            if namespaces:
                placeholders = ",".join("?" * len(namespaces))
                rows = self._conn().execute(
                    f"""
                    SELECT * FROM learning_lessons
                    WHERE status = 'active' AND namespace IN ({placeholders})
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    [*namespaces, int(limit)],
                ).fetchall()
            else:
                rows = self._conn().execute(
                    """
                    SELECT * FROM learning_lessons
                    WHERE status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def list_for_viewer(self, limit: int = 200) -> List[dict]:
        """Safe projection for Memory Viewer — no sensitive quotes/values."""
        from .safety import looks_sensitive_for_ui

        lessons = self.list_active(limit=limit)
        # Also show superseded/deleted lightly for audit? Spec: type, subject,
        # confidence, date, status — hide sensitive values.
        self._ensure_schema()
        with self._lock:
            rows = self._conn().execute(
                """
                SELECT id, lesson_type, subject_key, value, confidence,
                       status, updated_at, namespace, sensitivity, source_quote
                FROM learning_lessons
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            sensitive = looks_sensitive_for_ui(r["value"] or "", r["source_quote"] or "")
            out.append({
                "id": r["id"],
                "lesson_type": r["lesson_type"],
                "subject_key": r["subject_key"],
                "confidence": r["confidence"],
                "status": r["status"],
                "updated_at": r["updated_at"],
                "namespace": r["namespace"],
                "value_preview": "[hidden]" if sensitive else (r["value"] or "")[:120],
            })
        return out

    @staticmethod
    def _row_to_lesson(row) -> Lesson:
        d = dict(row)
        return Lesson(
            id=d["id"],
            lesson_type=d["lesson_type"],
            subject_key=d["subject_key"],
            value=d["value"],
            source_quote=d["source_quote"],
            conversation_id=d["conversation_id"],
            turn_id=d["turn_id"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            confidence=float(d["confidence"]),
            sensitivity=d.get("sensitivity") or "normal",
            status=d.get("status") or "active",
            expires_at=d.get("expires_at"),
            namespace=d.get("namespace") or "profile",
            occurrence_count=int(d.get("occurrence_count") or 1),
        )
