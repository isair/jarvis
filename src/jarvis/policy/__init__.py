"""
Jarvis Policy Engine
====================
Provides a formal policy layer between planning and tool execution.

Every tool execution attempt must be evaluated by the policy engine
before it is permitted. The engine produces a :class:`PolicyDecision`
that describes whether the action is allowed, why, and which constraints
apply.

Public API::

    from jarvis.policy import evaluate, PolicyDecision, PolicyDeniedError
"""

from .engine import evaluate, PolicyEngine
from .models import (
    PolicyDecision,
    PolicyMode,
    ToolClass,
    NetworkClass,
    AccessMode,
    RiskLevel,
)
from .path_guard import resolve_and_validate_path, PathGuard
from .approvals import ApprovalStore, ScopedGrant

__all__ = [
    "evaluate",
    "PolicyEngine",
    "PolicyDecision",
    "PolicyMode",
    "ToolClass",
    "NetworkClass",
    "AccessMode",
    "RiskLevel",
    "resolve_and_validate_path",
    "PathGuard",
    "ApprovalStore",
    "ScopedGrant",
]
