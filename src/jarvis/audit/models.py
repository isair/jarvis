"""
Audit data models.

These are lightweight data-transfer objects that map directly onto the
``tasks``, ``task_steps``, ``policy_decisions``, and ``approvals`` tables
in the audit database.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """
    One logical task — corresponds to a single user query dispatched to the
    reply engine.

    Persisted to the ``tasks`` table.
    """
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Stable unique identifier for this task."""

    intent: str = ""
    """Redacted user query / intent."""

    request_type: str = ""
    """Classification from :func:`jarvis.approval.classify_request`."""

    selected_profile: str = ""
    """Profile chosen by the profile-selector LLM."""

    selected_tools: List[str] = field(default_factory=list)
    """Tools invoked during execution (populated at the end)."""

    status: str = "planning"
    """One of: planning, executing, awaiting_approval, done, failed."""

    started_at: float = field(default_factory=time.time)
    """UNIX timestamp of task creation."""

    finished_at: Optional[float] = None
    """UNIX timestamp of task completion."""

    duration_ms: Optional[float] = None
    """Wall-clock duration in milliseconds."""

    final_status: str = "unknown"
    """done | failed | approval_denied | policy_denied"""

    error: Optional[str] = None
    """Error message if the task failed."""


# ---------------------------------------------------------------------------
# TaskStepRecord
# ---------------------------------------------------------------------------

@dataclass
class TaskStepRecord:
    """
    One execution step within a task — typically one tool invocation.

    Persisted to the ``task_steps`` table.
    """
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""

    tool_name: str = ""
    """Canonical tool name (camelCase or server__tool)."""

    args_hash: str = ""
    """SHA-256 of the canonical JSON-serialised arguments (for dedup detection)."""

    policy_audit_id: str = ""
    """Foreign key into ``policy_decisions``."""

    retry_count: int = 0
    """How many times the tool was retried."""

    result_summary: str = ""
    """Short human-readable summary of the result (≤ 200 chars)."""

    success: bool = True

    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None

    @staticmethod
    def hash_args(args: Optional[Dict[str, Any]]) -> str:
        """Return a SHA-256 hex digest of the canonically serialised arguments."""
        canonical = json.dumps(args or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# PolicyDecisionRecord
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecisionRecord:
    """
    Snapshot of a :class:`~jarvis.policy.PolicyDecision`.

    Persisted to the ``policy_decisions`` table.
    """
    audit_id: str = ""
    task_id: str = ""
    step_id: str = ""

    tool_name: str = ""
    tool_class: str = ""
    risk_level: str = ""
    allowed: bool = True
    approval_required: bool = False
    decision_reason: str = ""
    denied_reason: str = ""
    constraints_json: str = "[]"
    """JSON-serialised list of applied constraint names."""

    decided_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ApprovalRecord
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRecord:
    """
    Records a user approval decision.

    Persisted to the ``approvals`` table.
    """
    approval_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    step_id: str = ""

    tool_name: str = ""
    operation: str = "*"
    path_prefix: str = "*"

    decision: str = "granted"
    """``"granted"`` | ``"denied"``"""

    granted_by: str = "user"
    decided_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
