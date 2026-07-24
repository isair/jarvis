"""State / provenance memory store (Cora Brain Foundation, Phase 4 — Module E).

A NEW, additive sqlite store for *durable* facts, rules and preferences that Cora
learns about her owner/projects/world, with an explicit **confirmation lifecycle**
and a full **audit trail**. It is deliberately SEPARATE from:

  * ``learning_lessons`` (learning/store.py) — the lightweight learning-loop store;
  * ``conversation_summaries`` / ``ConversationStore`` — diary + chat transcripts.

Design (mirrors ``conversation_store``'s maximum-safety pattern for the shared DB):
this module owns its own DDL and creates its tables lazily via
``CREATE TABLE IF NOT EXISTS`` on the shared ``Database`` connection. It does NOT
modify ``db.py``, does NOT touch ``learning_lessons``, changes no PRAGMA, and never
drops/alters existing data. A component-keyed ``schema_migrations`` row records the
version. All writes serialise on the shared ``Database`` RLock so they interleave
safely with diary / learning / conversation writes on the one connection.

Safety posture (why this store exists):
  * Everything enters as a ``candidate``. Nothing is trusted until ``confirmed``.
  * **ASR-sourced** values (``add_from_asr``) are NEVER auto-confirmed: on low
    confidence or missing quality metrics they are routed straight to
    ``quarantined`` — a mis-heard word must never become a durable "fact".
  * **Auto-promotion** to ``confirmed`` is allowed ONLY for a *non-sensitive*
    ``user_preference`` that is explicit, high-confidence, and recurs across ≥3
    INDEPENDENT conversations, does not contradict a confirmed ``owner_rule``, and
    passes the secret / injection filter. Rules, corrections, identity, web facts,
    and anything sensitive ALWAYS require an explicit human ``confirm``.
  * When the policy flag ``memory_require_confirmation`` is set (the default),
    auto-promotion is disabled entirely — even eligible preferences wait.

This store NEVER authorises tools or development. It only remembers. There is no
method here that grants execution, and none should be added — reading a memory is
not permission to act on it.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .learning.safety import (
    looks_sensitive_for_ui,
    scrub_lesson_text,
    should_reject_lesson_value,
)
from .learning.types import CONFIDENCE_CAPS

try:  # debug_log is optional; never let logging break a store call
    from ..debug import debug_log
except Exception:  # pragma: no cover - defensive
    def debug_log(*_a, **_k):  # type: ignore
        return None

__all__ = [
    "StateStore",
    "asr_quality_ok",
    "STATE_MEMORY_SCHEMA_VERSION",
    "STATE_TYPES",
    "STATE_STATUSES",
]

STATE_MEMORY_SCHEMA_VERSION = 1

# --- vocabulary -------------------------------------------------------------
TYPE_OWNER_RULE = "owner_rule"
TYPE_USER_PREFERENCE = "user_preference"
TYPE_USER_FACT = "user_fact"
TYPE_PROJECT_FACT = "project_fact"
TYPE_CORRECTION = "correction"
TYPE_WEB_FACT = "web_fact"
TYPE_IMPROVEMENT_CANDIDATE = "improvement_candidate"

STATE_TYPES = frozenset({
    TYPE_OWNER_RULE,
    TYPE_USER_PREFERENCE,
    TYPE_USER_FACT,
    TYPE_PROJECT_FACT,
    TYPE_CORRECTION,
    TYPE_WEB_FACT,
    TYPE_IMPROVEMENT_CANDIDATE,
})

STATUS_CANDIDATE = "candidate"
STATUS_PENDING = "pending_confirmation"
STATUS_CONFIRMED = "confirmed"
STATUS_QUARANTINED = "quarantined"
STATUS_SUPERSEDED = "superseded"
STATUS_FORGOTTEN = "forgotten"

STATE_STATUSES = frozenset({
    STATUS_CANDIDATE,
    STATUS_PENDING,
    STATUS_CONFIRMED,
    STATUS_QUARANTINED,
    STATUS_SUPERSEDED,
    STATUS_FORGOTTEN,
})

# States a live row can still be worked with (dedupe target / mutable).
_LIVE_STATUSES = (STATUS_CANDIDATE, STATUS_PENDING, STATUS_CONFIRMED)
# Terminal states — never a dedupe target.
_TERMINAL_STATUSES = (STATUS_SUPERSEDED, STATUS_FORGOTTEN)

# Provenance that counts as an *explicit* human statement (required to auto-promote).
_EXPLICIT_PROVENANCES = frozenset({"user_explicit_correction", "user_direct"})

# Auto-promotion thresholds.
AUTO_PROMOTE_MIN_CONFIDENCE = 0.9
AUTO_PROMOTE_MIN_CONVERSATIONS = 3

# ASR quality gate.
ASR_MIN_CONFIDENCE = 0.6

_NOSET = object()  # sentinel: "do not change this column" in a transition


_DDL = """
CREATE TABLE IF NOT EXISTS memory_states (
  id               TEXT PRIMARY KEY,
  type             TEXT NOT NULL,
  subject_key      TEXT NOT NULL,
  value            TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'candidate',
  source           TEXT,
  conversation_id  TEXT,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  confidence       REAL NOT NULL DEFAULT 0.0,
  recurrence_count INTEGER NOT NULL DEFAULT 1,
  confirmed_by     TEXT,
  provenance       TEXT NOT NULL DEFAULT 'user_direct',
  source_url       TEXT,
  fetched_at       TEXT,
  expires_at       TEXT,
  supersedes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_states_status
  ON memory_states(status, type, subject_key);

CREATE TABLE IF NOT EXISTS memory_audit (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id     TEXT NOT NULL,
  actor       TEXT,
  action      TEXT NOT NULL,
  from_status TEXT,
  to_status   TEXT,
  reason      TEXT,
  at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_audit_item
  ON memory_audit(item_id, id);

-- Distinct conversations that independently produced the SAME candidate value.
-- The unique (item_id, conversation_id) key is what makes "recurrence across N
-- INDEPENDENT conversations" honest: re-stating a preference twice inside one
-- conversation cannot inflate the count.
--
-- ``provenance`` records HOW that conversation produced the value. Only
-- *explicit* mentions (see :data:`_EXPLICIT_PROVENANCES`) may count toward the
-- auto-promotion total: a non-explicit / ASR echo (``model_inference``,
-- ``allow_auto_promote=False``) contributes to recurrence history but must never
-- pad the independent-conversation count that promotes a preference. A row's
-- provenance is only ever *upgraded* to explicit — never downgraded.
CREATE TABLE IF NOT EXISTS memory_state_sources (
  item_id         TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  at              TEXT NOT NULL,
  provenance      TEXT NOT NULL DEFAULT 'model_inference',
  PRIMARY KEY (item_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
  component  TEXT PRIMARY KEY,
  version    INTEGER NOT NULL,
  applied_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    # Microsecond resolution so updated_at / audit ``at`` are strictly increasing
    # across rapid transitions. ISO-8601 UTC sorts lexically == chronologically.
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    """Lower-case + whitespace-collapse for keys and value comparisons."""
    return " ".join((text or "").strip().lower().split())


def asr_quality_ok(asr_confidence: Optional[float], asr_metrics: Optional[dict]) -> bool:
    """Return ``True`` only when an ASR transcript is trustworthy enough to keep.

    Fails closed: a missing confidence, a missing / non-dict / empty metrics
    payload, a confidence below :data:`ASR_MIN_CONFIDENCE`, or any standard
    faster-whisper quality signal (``avg_logprob``, ``no_speech_prob``,
    ``compression_ratio``) in the danger zone all return ``False``. A ``False``
    result never means "confirm" — it routes the value to quarantine.
    """
    if asr_confidence is None:
        return False
    try:
        conf = float(asr_confidence)
    except (TypeError, ValueError):
        return False
    if conf < ASR_MIN_CONFIDENCE:
        return False
    if not asr_metrics or not isinstance(asr_metrics, dict):
        return False
    try:
        if "avg_logprob" in asr_metrics and float(asr_metrics["avg_logprob"]) < -1.0:
            return False
        if "no_speech_prob" in asr_metrics and float(asr_metrics["no_speech_prob"]) > 0.6:
            return False
        if "compression_ratio" in asr_metrics and float(asr_metrics["compression_ratio"]) > 2.4:
            return False
    except (TypeError, ValueError):
        return False
    return True


class StateStore:
    """Confirmation-gated state/provenance memory over ``memory_states``.

    Build against a ``jarvis.memory.db.Database`` (shared profile DB) or any
    object exposing ``.conn`` (an sqlite3 connection with ``row_factory =
    sqlite3.Row``) and optionally ``._lock``.

    ``require_confirmation`` mirrors the ``memory_require_confirmation`` setting:
    when ``True`` (the default, safest) auto-promotion is disabled and every
    ``confirmed`` state must be reached through an explicit ``confirm`` /
    ``correct`` call.
    """

    def __init__(self, db, *, require_confirmation: bool = True) -> None:
        self._db = db
        # Reuse the Database RLock so writes serialise with every other writer on
        # the shared connection; fall back to a private lock for a bare double.
        self._lock = getattr(db, "_lock", None) or threading.RLock()
        self._require_confirmation = bool(require_confirmation)
        self._schema_ready = False

    # -- connection / schema --------------------------------------------------
    def _conn(self):
        return self._db.conn

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if self._schema_ready:
                return
            conn = self._conn()
            conn.executescript(_DDL)
            # Idempotent upgrade for a pre-existing memory_state_sources table
            # created before the per-source ``provenance`` column existed. The
            # CREATE above is a no-op on such tables, so add the column here.
            # Old rows default to the non-explicit provenance (fail-closed: they
            # never inflate the auto-promotion count).
            src_cols = {
                r[1] for r in conn.execute(
                    "PRAGMA table_info(memory_state_sources)"
                ).fetchall()
            }
            if "provenance" not in src_cols:
                conn.execute(
                    "ALTER TABLE memory_state_sources "
                    "ADD COLUMN provenance TEXT NOT NULL DEFAULT 'model_inference'"
                )
            row = conn.execute(
                "SELECT version FROM schema_migrations WHERE component = 'memory_states'"
            ).fetchone()
            current = int(row[0]) if row else 0
            if current < STATE_MEMORY_SCHEMA_VERSION:
                conn.execute(
                    """INSERT INTO schema_migrations (component, version, applied_at)
                       VALUES ('memory_states', ?, ?)
                       ON CONFLICT(component) DO UPDATE SET
                         version = excluded.version, applied_at = excluded.applied_at""",
                    (STATE_MEMORY_SCHEMA_VERSION, _utc_now()),
                )
            conn.commit()
            self._schema_ready = True

    # -- internal helpers -----------------------------------------------------
    def _cap_confidence(self, confidence: Optional[float], provenance: str) -> float:
        cap = float(CONFIDENCE_CAPS.get(provenance, 0.95))
        if confidence is None:
            return cap
        try:
            return min(float(confidence), cap)
        except (TypeError, ValueError):
            return cap

    def _distinct_conversations(self, item_id: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM memory_state_sources WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def _distinct_explicit_conversations(self, item_id: str) -> int:
        """Count only conversations that produced this value via an *explicit*
        human statement. This — not :meth:`_distinct_conversations` — is what
        gates auto-promotion, so a non-explicit / ASR echo can never pad the
        independent-conversation total toward :data:`AUTO_PROMOTE_MIN_CONVERSATIONS`.
        """
        placeholders = ",".join("?" * len(_EXPLICIT_PROVENANCES))
        row = self._conn().execute(
            f"""SELECT COUNT(*) FROM memory_state_sources
                WHERE item_id = ? AND provenance IN ({placeholders})""",
            (item_id, *sorted(_EXPLICIT_PROVENANCES)),
        ).fetchone()
        return int(row[0]) if row else 0

    def _record_source(self, conn, item_id: str, conversation_id: str, at: str,
                       provenance: str) -> None:
        """Record (or upgrade) the per-conversation source row for an item.

        A conversation is inserted with its own ``provenance``. If the same
        conversation was already recorded, its provenance is *upgraded* to
        explicit when this mention is explicit, but is NEVER downgraded — so an
        explicit statement that also happens to be re-heard by ASR still counts,
        while an ASR echo can never demote an explicit source.
        """
        is_explicit = 1 if provenance in _EXPLICIT_PROVENANCES else 0
        conn.execute(
            """INSERT INTO memory_state_sources
                 (item_id, conversation_id, at, provenance)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(item_id, conversation_id) DO UPDATE SET
                 provenance = CASE WHEN ? = 1 THEN excluded.provenance
                                   ELSE memory_state_sources.provenance END""",
            (item_id, conversation_id, at, provenance, is_explicit),
        )

    def _find_live_duplicate(self, item_type: str, subject_key: str, value: str):
        """Existing non-terminal row with the same type+key+value (case-fold)."""
        placeholders = ",".join("?" * len(_TERMINAL_STATUSES))
        return self._conn().execute(
            f"""
            SELECT * FROM memory_states
            WHERE type = ? AND subject_key = ? AND lower(value) = lower(?)
              AND status NOT IN ({placeholders})
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (item_type, subject_key, value, *_TERMINAL_STATUSES),
        ).fetchone()

    def _audit(self, conn, item_id, actor, action, from_status, to_status, reason, at) -> None:
        conn.execute(
            """INSERT INTO memory_audit
                 (item_id, actor, action, from_status, to_status, reason, at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (item_id, actor, action, from_status, to_status, reason, at),
        )

    # -- ingestion ------------------------------------------------------------
    def add_candidate(
        self,
        item_type: str,
        subject_key: str,
        value: str,
        *,
        source: Optional[str] = None,
        conversation_id: Optional[str] = None,
        confidence: Optional[float] = None,
        provenance: str = "user_direct",
        sensitivity: str = "normal",
        source_url: Optional[str] = None,
        fetched_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        actor: str = "system",
    ) -> Optional[str]:
        """Record a new ``candidate`` (or fold into an existing live duplicate).

        Returns the item id, or ``None`` when the value is rejected by the
        secret / injection / PII filter (nothing is persisted in that case). A
        non-sensitive ``user_preference`` may be auto-promoted to ``confirmed``
        by this call — see :meth:`_maybe_auto_promote`.
        """
        return self._add_core(
            item_type,
            subject_key,
            value,
            source=source,
            conversation_id=conversation_id,
            confidence=confidence,
            provenance=provenance,
            sensitivity=sensitivity,
            source_url=source_url,
            fetched_at=fetched_at,
            expires_at=expires_at,
            actor=actor,
            allow_auto_promote=True,
        )

    def add_from_asr(
        self,
        value: str,
        *,
        asr_confidence: Optional[float],
        asr_metrics: Optional[dict],
        item_type: str = TYPE_USER_FACT,
        subject_key: str = "asr",
        source: str = "asr",
        conversation_id: Optional[str] = None,
        actor: str = "system:asr",
    ) -> Optional[str]:
        """Ingest a value that originated from speech recognition.

        NEVER auto-confirms. When :func:`asr_quality_ok` is ``False`` (low
        confidence OR missing/degraded metrics) the value is added and then
        immediately ``quarantined`` — a mis-heard word must not become a durable
        fact. When quality is OK it is left as a plain ``candidate`` for the
        normal (explicit-confirmation) path. Returns the id, or ``None`` if the
        value was rejected by the secret filter.
        """
        item_id = self._add_core(
            item_type,
            subject_key,
            value,
            source=source,
            conversation_id=conversation_id,
            confidence=asr_confidence,
            provenance="model_inference",  # ASR is inference, never explicit
            sensitivity="normal",
            actor=actor,
            allow_auto_promote=False,  # belt-and-suspenders: ASR never auto-promotes
        )
        if item_id is None:
            return None
        if not asr_quality_ok(asr_confidence, asr_metrics):
            # Only quarantine a freshly-added / still-candidate row. If this
            # value merely deduped into an already confirmed (or pending) item, a
            # noisy ASR echo must NOT be allowed to un-confirm good, human-vetted
            # memory — leave that item untouched.
            #
            # The get() pre-check is a fast path only; correctness rests on the
            # atomic ``allowed_from=(STATUS_CANDIDATE,)`` below. A concurrent
            # confirm() landing between this read and the quarantine would leave
            # the item CONFIRMED, and the guarded transition then becomes a no-op
            # instead of un-confirming it (TOCTOU-safe).
            current = self.get(item_id)
            if current is not None and current["status"] == STATUS_CANDIDATE:
                self.quarantine(
                    item_id, reason="asr_low_quality", actor=actor,
                    allowed_from=(STATUS_CANDIDATE,),
                )
        return item_id

    def _add_core(
        self,
        item_type: str,
        subject_key: str,
        value: str,
        *,
        source: Optional[str],
        conversation_id: Optional[str],
        confidence: Optional[float],
        provenance: str,
        sensitivity: str,
        source_url: Optional[str] = None,
        fetched_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        actor: str,
        allow_auto_promote: bool,
    ) -> Optional[str]:
        if item_type not in STATE_TYPES:
            raise ValueError(f"unknown memory type: {item_type!r}")
        raw = (value or "").strip()
        if not raw:
            raise ValueError("value must be non-empty")
        # Reject secrets / injection / heavy PII on the RAW text first — never
        # persist them, not even scrubbed.
        reject, reason = should_reject_lesson_value(raw, item_type)
        if reject:
            debug_log(f"state_store: rejected candidate ({reason})", "memory")
            return None
        stored_value = scrub_lesson_text(raw)
        key = _norm(subject_key)
        prov = provenance or "user_direct"
        conf = self._cap_confidence(confidence, prov)
        now = _utc_now()

        self._ensure_schema()
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN")
                dup = self._find_live_duplicate(item_type, key, stored_value)
                if dup is not None:
                    item_id = dup["id"]
                    if conversation_id:
                        self._record_source(conn, item_id, conversation_id, now, prov)
                    rc = self._distinct_conversations(item_id)
                    conn.execute(
                        """UPDATE memory_states
                           SET updated_at = ?, recurrence_count = ?,
                               confidence = MAX(confidence, ?),
                               source = COALESCE(?, source),
                               conversation_id = COALESCE(?, conversation_id)
                           WHERE id = ?""",
                        (now, max(rc, 1), conf, source, conversation_id, item_id),
                    )
                else:
                    item_id = str(uuid.uuid4())
                    conn.execute(
                        """INSERT INTO memory_states
                             (id, type, subject_key, value, status, source,
                              conversation_id, created_at, updated_at, confidence,
                              recurrence_count, confirmed_by, provenance,
                              source_url, fetched_at, expires_at, supersedes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item_id, item_type, key, stored_value, STATUS_CANDIDATE,
                            source, conversation_id, now, now, conf, 1, None, prov,
                            source_url, fetched_at, expires_at, None,
                        ),
                    )
                    if conversation_id:
                        self._record_source(conn, item_id, conversation_id, now, prov)
                    self._audit(
                        conn, item_id, actor, "add", None, STATUS_CANDIDATE,
                        f"provenance:{prov}", now,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        # Auto-promotion runs in its own transaction (after the insert commits).
        if allow_auto_promote:
            self._maybe_auto_promote(item_id)
        return item_id

    # -- auto-promotion -------------------------------------------------------
    def _contradicts_owner_rule(self, subject_key: str, value: str) -> bool:
        """True iff a CONFIRMED ``owner_rule`` exists on the same subject with a
        different value — a preference must never silently override a rule."""
        rows = self._conn().execute(
            """SELECT value FROM memory_states
               WHERE type = ? AND subject_key = ? AND status = ?""",
            (TYPE_OWNER_RULE, subject_key, STATUS_CONFIRMED),
        ).fetchall()
        target = _norm(value)
        return any(_norm(r["value"]) != target for r in rows)

    def _maybe_auto_promote(self, item_id: str) -> bool:
        """Promote a candidate ``user_preference`` to ``confirmed`` only when every
        guard passes. Returns ``True`` if it promoted.

        Guards: policy allows it; still a fresh ``candidate``; type is
        ``user_preference``; provenance is explicit; confidence ≥ threshold;
        recurs across ≥3 independent conversations; value is not sensitive; passes
        the secret filter; does not contradict a confirmed ``owner_rule``.
        """
        if self._require_confirmation:
            return False
        item = self.get(item_id)
        if not item or item["status"] != STATUS_CANDIDATE:
            return False
        if item["type"] != TYPE_USER_PREFERENCE:
            return False
        if item["provenance"] not in _EXPLICIT_PROVENANCES:
            return False
        if float(item["confidence"]) < AUTO_PROMOTE_MIN_CONFIDENCE:
            return False
        # Hold the lock across BOTH the guards AND the promoting transition so no
        # concurrent write can slip in between them (e.g. a confirmed owner_rule
        # inserted after the contradiction check but before the commit). The
        # shared lock is a re-entrant RLock, so the inner ``_transition`` re-locks
        # safely inside this critical section.
        with self._lock:
            # Only EXPLICIT independent conversations count toward promotion — an
            # ASR echo / non-explicit mention adds recurrence history but must not
            # pad the threshold.
            if self._distinct_explicit_conversations(item_id) < AUTO_PROMOTE_MIN_CONVERSATIONS:
                return False
            value = item["value"]
            if looks_sensitive_for_ui(value):
                return False
            reject, _ = should_reject_lesson_value(value, TYPE_USER_PREFERENCE)
            if reject:
                return False
            subject_key = item["subject_key"]
            if self._contradicts_owner_rule(subject_key, value):
                return False
            # The promoting write re-checks the owner_rule invariant INSIDE its
            # own transaction (precommit_guard), so a contradicting owner_rule
            # confirmed after the guard above but before the commit still blocks
            # promotion instead of silently overriding a rule.
            return self._transition(
                item_id,
                STATUS_CONFIRMED,
                actor="system:auto",
                action="auto_confirm",
                reason="auto_promoted_recurrence",
                confirmed_by="auto",
                allowed_from=(STATUS_CANDIDATE,),
                precommit_guard=lambda: not self._contradicts_owner_rule(subject_key, value),
            )

    # -- transitions ----------------------------------------------------------
    def _transition(
        self,
        item_id: str,
        to_status: str,
        *,
        actor: Optional[str],
        action: str,
        reason: Optional[str] = None,
        confirmed_by: Any = _NOSET,
        allowed_from: Optional[tuple] = None,
        precommit_guard: Optional[Any] = None,
    ) -> bool:
        """Atomically move an item to ``to_status`` and record one audit row.

        Returns ``False`` (no-op) when the item is missing, its current status is
        not in ``allowed_from``, or ``precommit_guard`` (a zero-arg callable, when
        supplied) returns falsey. ``precommit_guard`` is evaluated INSIDE the same
        locked transaction, immediately before the write, so an invariant it
        checks (e.g. "no contradicting owner_rule exists") is verified atomically
        with the commit — closing any check-then-act (TOCTOU) window.
        """
        self._ensure_schema()
        now = _utc_now()
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN")
                row = conn.execute(
                    "SELECT status FROM memory_states WHERE id = ?", (item_id,)
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return False
                from_status = row["status"]
                if allowed_from is not None and from_status not in allowed_from:
                    conn.rollback()
                    return False
                if precommit_guard is not None and not precommit_guard():
                    conn.rollback()
                    return False
                sets = "status = ?, updated_at = ?"
                params: List[Any] = [to_status, now]
                if confirmed_by is not _NOSET:
                    sets += ", confirmed_by = ?"
                    params.append(confirmed_by)
                params.append(item_id)
                conn.execute(
                    f"UPDATE memory_states SET {sets} WHERE id = ?", params
                )
                self._audit(conn, item_id, actor, action, from_status, to_status, reason, now)
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def promote_to_pending(self, item_id: str, *, actor: str = "system") -> bool:
        """candidate → pending_confirmation (awaiting explicit human confirm)."""
        return self._transition(
            item_id, STATUS_PENDING, actor=actor, action="promote",
            reason="promote_to_pending", allowed_from=(STATUS_CANDIDATE,),
        )

    def confirm(self, item_id: str, confirmed_by: str, *, actor: Optional[str] = None) -> bool:
        """candidate|pending → confirmed. ``confirmed_by`` records who confirmed."""
        return self._transition(
            item_id, STATUS_CONFIRMED, actor=actor or confirmed_by, action="confirm",
            reason="confirmed", confirmed_by=confirmed_by,
            allowed_from=(STATUS_CANDIDATE, STATUS_PENDING),
        )

    def quarantine(
        self,
        item_id: str,
        reason: str,
        *,
        actor: str = "system",
        allowed_from: tuple = (STATUS_CANDIDATE, STATUS_PENDING, STATUS_CONFIRMED),
    ) -> bool:
        """Route an item to ``quarantined`` (excluded from all retrieval).

        ``allowed_from`` restricts which current statuses may be quarantined; the
        status check and the write happen atomically in one locked transaction
        (see :meth:`_transition`). Callers that must NOT un-confirm human-vetted
        memory — e.g. the ASR low-quality path racing a concurrent ``confirm`` —
        pass ``allowed_from=(STATUS_CANDIDATE,)`` so the quarantine is a no-op the
        instant the item is no longer a bare candidate.
        """
        return self._transition(
            item_id, STATUS_QUARANTINED, actor=actor, action="quarantine",
            reason=reason, allowed_from=allowed_from,
        )

    def forget(self, item_id: str, *, actor: str = "system", reason: Optional[str] = None) -> bool:
        """Soft-forget: status → ``forgotten``. The row and its audit trail are
        KEPT (recoverable / auditable); retrieval excludes it."""
        return self._transition(
            item_id, STATUS_FORGOTTEN, actor=actor, action="forget",
            reason=reason or "forgotten",
            allowed_from=(
                STATUS_CANDIDATE, STATUS_PENDING, STATUS_CONFIRMED,
                STATUS_QUARANTINED, STATUS_SUPERSEDED,
            ),
        )

    def correct(
        self,
        item_id: str,
        new_value: str,
        *,
        confirmed_by: str = "user",
        actor: Optional[str] = None,
        source: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Explicit correction: the old item becomes ``superseded`` and a NEW
        ``confirmed`` item (same type/subject, ``supersedes`` → old id) is created.

        Raises ``KeyError`` if the item is unknown and ``ValueError`` if the new
        value fails the secret filter. Returns the new item id.
        """
        raw = (new_value or "").strip()
        if not raw:
            raise ValueError("new_value must be non-empty")
        reject, reason = should_reject_lesson_value(raw, TYPE_CORRECTION)
        if reject:
            raise ValueError(f"rejected new_value: {reason}")
        stored_value = scrub_lesson_text(raw)
        prov = "user_explicit_correction"
        conf = self._cap_confidence(1.0, prov)
        now = _utc_now()
        new_id = str(uuid.uuid4())

        self._ensure_schema()
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN")
                old = conn.execute(
                    "SELECT * FROM memory_states WHERE id = ?", (item_id,)
                ).fetchone()
                if old is None:
                    conn.rollback()
                    raise KeyError(item_id)
                old = dict(old)
                conn.execute(
                    """INSERT INTO memory_states
                         (id, type, subject_key, value, status, source,
                          conversation_id, created_at, updated_at, confidence,
                          recurrence_count, confirmed_by, provenance,
                          source_url, fetched_at, expires_at, supersedes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id, old["type"], old["subject_key"], stored_value,
                        STATUS_CONFIRMED, source, conversation_id, now, now, conf,
                        1, confirmed_by, prov, None, None, None, item_id,
                    ),
                )
                if conversation_id:
                    self._record_source(conn, new_id, conversation_id, now, prov)
                conn.execute(
                    "UPDATE memory_states SET status = ?, updated_at = ? WHERE id = ?",
                    (STATUS_SUPERSEDED, now, item_id),
                )
                self._audit(
                    conn, item_id, actor or confirmed_by, "supersede",
                    old["status"], STATUS_SUPERSEDED, f"corrected_by:{new_id}", now,
                )
                self._audit(
                    conn, new_id, actor or confirmed_by, "correct",
                    None, STATUS_CONFIRMED, f"correction_of:{item_id}", now,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return new_id

    # -- reads ----------------------------------------------------------------
    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_schema()
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM memory_states WHERE id = ?", (item_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_by_state(self, status: str, *, limit: int = 200) -> List[Dict[str, Any]]:
        self._ensure_schema()
        with self._lock:
            rows = self._conn().execute(
                """SELECT * FROM memory_states WHERE status = ?
                   ORDER BY updated_at DESC, rowid DESC LIMIT ?""",
                (status, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def retrieve_confirmed(
        self,
        item_type: Optional[str] = None,
        subject_key: Optional[str] = None,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return ONLY ``confirmed``, non-expired items. Candidates, pending,
        quarantined, superseded and forgotten rows are excluded by construction —
        this is the single safe read path for "what does Cora actually know"."""
        self._ensure_schema()
        now = _utc_now()
        clauses = ["status = ?", "(expires_at IS NULL OR expires_at > ?)"]
        params: List[Any] = [STATUS_CONFIRMED, now]
        if item_type:
            clauses.append("type = ?")
            params.append(item_type)
        if subject_key:
            clauses.append("subject_key = ?")
            params.append(_norm(subject_key))
        params.append(int(limit))
        sql = (
            "SELECT * FROM memory_states WHERE " + " AND ".join(clauses) +
            " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_audit(self, item_id: str) -> List[Dict[str, Any]]:
        """Full audit trail for an item, oldest transition first."""
        self._ensure_schema()
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM memory_audit WHERE item_id = ? ORDER BY id ASC",
                (item_id,),
            ).fetchall()
        return [dict(r) for r in rows]
