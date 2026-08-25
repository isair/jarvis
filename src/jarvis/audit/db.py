"""
Audit database — SQLite schema and low-level access layer.

Tables
------
tasks            – one row per user query (top-level task)
task_steps       – one row per tool invocation within a task
policy_decisions – one row per policy evaluation (linked to a step)
approvals        – one row per user approval or denial
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..debug import debug_log

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- ── Tasks ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    intent          TEXT NOT NULL DEFAULT '',
    request_type    TEXT NOT NULL DEFAULT '',
    selected_profile TEXT NOT NULL DEFAULT '',
    selected_tools  TEXT NOT NULL DEFAULT '[]',   -- JSON array
    status          TEXT NOT NULL DEFAULT 'planning',
    started_at      REAL NOT NULL,
    finished_at     REAL,
    duration_ms     REAL,
    final_status    TEXT NOT NULL DEFAULT 'unknown',
    error           TEXT
);

-- ── Task steps ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS task_steps (
    step_id         TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    tool_name       TEXT NOT NULL DEFAULT '',
    args_hash       TEXT NOT NULL DEFAULT '',
    policy_audit_id TEXT NOT NULL DEFAULT '',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    result_summary  TEXT NOT NULL DEFAULT '',
    success         INTEGER NOT NULL DEFAULT 1,   -- 0/1 boolean
    started_at      REAL NOT NULL,
    finished_at     REAL,
    duration_ms     REAL
);

CREATE INDEX IF NOT EXISTS idx_steps_task_id ON task_steps(task_id);

-- ── Policy decisions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS policy_decisions (
    audit_id          TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL DEFAULT '',
    step_id           TEXT NOT NULL DEFAULT '',
    tool_name         TEXT NOT NULL DEFAULT '',
    tool_class        TEXT NOT NULL DEFAULT '',
    risk_level        TEXT NOT NULL DEFAULT '',
    allowed           INTEGER NOT NULL DEFAULT 1,
    approval_required INTEGER NOT NULL DEFAULT 0,
    decision_reason   TEXT NOT NULL DEFAULT '',
    denied_reason     TEXT NOT NULL DEFAULT '',
    constraints_json  TEXT NOT NULL DEFAULT '[]',
    decided_at        REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pd_task_id ON policy_decisions(task_id);

-- ── Approvals ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL DEFAULT '',
    step_id     TEXT NOT NULL DEFAULT '',
    tool_name   TEXT NOT NULL DEFAULT '',
    operation   TEXT NOT NULL DEFAULT '*',
    path_prefix TEXT NOT NULL DEFAULT '*',
    decision    TEXT NOT NULL DEFAULT 'granted',
    granted_by  TEXT NOT NULL DEFAULT 'user',
    decided_at  REAL NOT NULL,
    expires_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_approvals_task_id ON approvals(task_id);
"""


class AuditDB:
    """
    Thread-safe SQLite wrapper for the audit database.

    The audit tables are written to a *separate* database from the main Jarvis
    database so that audit data is never accidentally cleared alongside
    user memories.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=10,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
            debug_log(f"audit db opened at {self._db_path}", "audit")
        except Exception as exc:
            debug_log(f"audit db init failed: {exc}", "audit")
            self._conn = None

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a single write statement inside a lock."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except Exception as exc:
                debug_log(f"audit db write error: {exc}", "audit")

    def executemany(self, sql: str, params_seq) -> None:
        """Execute a batch write statement inside a lock."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.executemany(sql, params_seq)
                self._conn.commit()
            except Exception as exc:
                debug_log(f"audit db batch write error: {exc}", "audit")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def fetchall(self, sql: str, params: tuple = ()):
        """Execute a query and return all rows as dicts."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
            except Exception as exc:
                debug_log(f"audit db query error: {exc}", "audit")
                return []
