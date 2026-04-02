"""
Policy data models.

All value types are immutable dataclasses or enums so the policy layer can
be tested and reasoned about without side-effects.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PolicyMode(Enum):
    """Operator-controlled policy enforcement level."""
    ALWAYS_ALLOW = "always_allow"
    """Allow everything automatically (development / demo mode)."""
    ASK_DESTRUCTIVE = "ask_destructive"
    """Ask the user only before destructive or high-risk operations (default)."""
    ASK_EVERY_TIME = "ask_every_time"
    """Require explicit approval for every tool invocation."""
    DENY_ALL = "deny_all"
    """Block all tool execution."""


class ToolClass(Enum):
    """Broad capability class of a tool — used to determine approval requirements."""
    INFORMATIONAL = "informational"
    """Read-only, no side-effects (screenshot, weather, recall)."""
    READ_ONLY_OPERATIONAL = "read_only_operational"
    """Reads from external systems but produces no mutations (web search, fetch page)."""
    WRITE_OPERATIONAL = "write_operational"
    """Creates or updates data (logMeal, localFiles write/append)."""
    DESTRUCTIVE = "destructive"
    """Permanently removes data (deleteMeal, localFiles delete)."""
    EXTERNAL_DELEGATED = "external_delegated"
    """Delegates to an external MCP server — trust depends on declared capabilities."""


class NetworkClass(Enum):
    """Network reach of a tool invocation."""
    NONE = "none"
    """No network access required."""
    LOOPBACK = "loopback"
    """127.0.0.1 / ::1 only (Ollama, local MCP servers)."""
    LAN = "lan"
    """RFC-1918 / CGNAT addresses."""
    PUBLIC = "public"
    """General public Internet."""
    MCP_ENDPOINT = "mcp_endpoint"
    """Outbound to a declared MCP server endpoint."""


class AccessMode(Enum):
    """Read / write intent for path-scoped operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    LIST = "list"


class RiskLevel(Enum):
    """Assessed risk of an action — consistent with :mod:`jarvis.approval`."""
    SAFE = "safe"
    MODERATE = "moderate"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppliedConstraint:
    """A single constraint that was enforced as part of a policy decision."""
    name: str
    description: str


@dataclass(frozen=True)
class PolicyDecision:
    """
    The result of evaluating a tool invocation against the active policy.

    Produced by :func:`jarvis.policy.engine.evaluate` for every tool call
    before execution.
    """
    allowed: bool
    """True when the action may proceed."""

    decision_reason: str
    """Human-readable explanation of the decision."""

    risk_level: RiskLevel
    """Assessed risk of the action."""

    approval_required: bool
    """True when user approval must be obtained before proceeding."""

    tool_class: ToolClass
    """Capability class of the tool being invoked."""

    applied_constraints: List[AppliedConstraint] = field(default_factory=list)
    """Zero or more constraints that were evaluated."""

    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Unique identifier for this decision (used by the audit recorder)."""

    denied_reason: Optional[str] = None
    """Populated when ``allowed`` is False — explains why it was denied."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def assert_allowed(self) -> None:
        """Raise :exc:`PolicyDeniedError` if the decision denies execution."""
        if not self.allowed:
            raise PolicyDeniedError(
                self.denied_reason or self.decision_reason,
                decision=self,
            )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PolicyDeniedError(PermissionError):
    """Raised when a policy decision blocks tool execution."""

    def __init__(self, message: str, decision: Optional[PolicyDecision] = None) -> None:
        super().__init__(message)
        self.decision = decision


class PolicyError(RuntimeError):
    """Raised when the policy engine itself encounters an internal error."""
