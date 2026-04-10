"""
Audit recorder — high-level API for writing task, step, policy, and approval records.

Usage::

    from jarvis.audit import configure_audit, get_recorder

    # At daemon startup:
    configure_audit("/path/to/audit.db")

    # In the reply engine:
    recorder = get_recorder()
    recorder.begin_task(task_record)
    recorder.record_step(step_record)
    recorder.record_policy_decision(decision_record)
    recorder.finish_task(task_id, final_status="done")
"""

from __future__ import annotations

import json
import time
from typing import Optional

from ..debug import debug_log
from ..utils.redact import redact as _redact
from .db import AuditDB
from .models import ApprovalRecord, PolicyDecisionRecord, TaskRecord, TaskStepRecord


class AuditRecorder:
    """
    High-level facade over :class:`~jarvis.audit.db.AuditDB`.

    All methods are safe to call even when the database is unavailable —
    they log a debug message and return silently.
    """

    def __init__(self, db: AuditDB) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, record: TaskRecord) -> None:
        """Insert a new task row."""
        self._db.execute(
            """
            INSERT OR IGNORE INTO tasks
              (task_id, intent, request_type, selected_profile, selected_tools,
               status, started_at, finished_at, duration_ms, final_status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.task_id,
                _redact(record.intent, max_len=500),
                record.request_type,
                record.selected_profile,
                json.dumps(record.selected_tools),
                record.status,
                record.started_at,
                record.finished_at,
                record.duration_ms,
                record.final_status,
                record.error,
            ),
        )
        debug_log(f"audit: task started task_id={record.task_id}", "audit")

    def finish_task(
        self,
        task_id: str,
        final_status: str,
        selected_tools: Optional[list] = None,
        selected_profile: str = "",
        error: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> None:
        """Update a task row with completion details.

        Pass *started_at* to avoid a round-trip query for the value
        recorded in :meth:`begin_task`.  When omitted the recorder
        falls back to reading from the database.
        """
        finished = time.time()
        if started_at is not None:
            started = started_at
        else:
            rows = self._db.fetchall(
                "SELECT started_at FROM tasks WHERE task_id = ?", (task_id,)
            )
            started = rows[0]["started_at"] if rows else finished
        duration = (finished - started) * 1000

        self._db.execute(
            """
            UPDATE tasks
               SET status = ?,
                   final_status = ?,
                   finished_at = ?,
                   duration_ms = ?,
                   selected_tools = COALESCE(NULLIF(?, ''), selected_tools),
                   selected_profile = COALESCE(NULLIF(?, ''), selected_profile),
                   error = ?
             WHERE task_id = ?
            """,
            (
                final_status,  # status column
                final_status,  # final_status column
                finished,
                duration,
                json.dumps(selected_tools or []),
                selected_profile,
                error,
                task_id,
            ),
        )
        debug_log(f"audit: task finished task_id={task_id} status={final_status}", "audit")

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def record_step(self, record: TaskStepRecord) -> None:
        """Insert a task-step row (or update if already exists)."""
        finished = record.finished_at
        duration = (
            (finished - record.started_at) * 1000
            if finished is not None
            else None
        )
        # Redact result_summary before persisting to the audit database
        safe_summary = _redact(record.result_summary, max_len=500) if record.result_summary else ""
        self._db.execute(
            """
            INSERT OR REPLACE INTO task_steps
              (step_id, task_id, tool_name, args_hash, policy_audit_id,
               retry_count, result_summary, success, started_at, finished_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.step_id,
                record.task_id,
                record.tool_name,
                record.args_hash,
                record.policy_audit_id,
                record.retry_count,
                safe_summary,
                int(record.success),
                record.started_at,
                finished,
                duration,
            ),
        )

    # ------------------------------------------------------------------
    # Policy decisions
    # ------------------------------------------------------------------

    def record_policy_decision(self, record: PolicyDecisionRecord) -> None:
        """Insert a policy-decision row."""
        self._db.execute(
            """
            INSERT OR IGNORE INTO policy_decisions
              (audit_id, task_id, step_id, tool_name, tool_class, risk_level,
               allowed, approval_required, decision_reason, denied_reason,
               constraints_json, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.audit_id,
                record.task_id,
                record.step_id,
                record.tool_name,
                record.tool_class,
                record.risk_level,
                int(record.allowed),
                int(record.approval_required),
                record.decision_reason[:1000],
                record.denied_reason[:1000],
                record.constraints_json,
                record.decided_at,
            ),
        )

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------

    def record_approval(self, record: ApprovalRecord) -> None:
        """Insert an approval record."""
        self._db.execute(
            """
            INSERT OR IGNORE INTO approvals
              (approval_id, task_id, step_id, tool_name, operation,
               path_prefix, decision, granted_by, decided_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.approval_id,
                record.task_id,
                record.step_id,
                record.tool_name,
                record.operation,
                record.path_prefix,
                record.decision,
                record.granted_by,
                record.decided_at,
                record.expires_at,
            ),
        )
        debug_log(
            f"audit: approval recorded tool={record.tool_name} decision={record.decision}",
            "audit",
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_recorder: Optional[AuditRecorder] = None


def configure(db_path: str) -> AuditRecorder:
    """
    Initialise the module-level :class:`AuditRecorder`.

    Call once at daemon startup.
    """
    global _recorder
    _recorder = AuditRecorder(AuditDB(db_path))
    debug_log(f"audit recorder configured at {db_path}", "audit")
    return _recorder


def get_recorder() -> Optional[AuditRecorder]:
    """Return the module-level recorder, or ``None`` if not configured."""
    return _recorder
