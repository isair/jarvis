"""Owner-triggered Development Mode (Phase 4 · Section H).

A default-OFF, owner-only capability: Cora can research, build, and self-test a
change in an isolated workspace, but only when the owner explicitly triggers it
and only within hard authority limits (no push / PR / merge / deploy / delete /
n8n / paid / external). Applying an approved change requires an explicit
two-step confirmation.

This package is inert at import time — no I/O, no subprocess, no network. The
state machine (``mode.py``) is pure in-memory policy; the coding-agent bridge
(``claude_bridge.py``) binds policy to a provider interface and ships the no-op
``DisabledProvider`` as the default, so nothing runs until the owner both
enables the flag and selects a real provider.
"""

from __future__ import annotations

from .claude_bridge import (
    AgentRequest,
    AgentResult,
    ArgvSpec,
    Caps,
    ClaudeCliProvider,
    CodingAgentProvider,
    DangerousFlagRequested,
    DisabledProvider,
    get_provider,
)
from .mode import (
    DEVELOP_AND_TEST_AUTHORITIES,
    FORBIDDEN_AUTHORITIES,
    ActivationNotConfirmed,
    Authority,
    AuthorityDenied,
    DevelopmentMode,
    DevJob,
    DevModeError,
    DevState,
    InvalidTransition,
    NotOwnerTriggered,
    SingleFlightViolation,
)

__all__ = [
    # bridge
    "AgentRequest",
    "AgentResult",
    "ArgvSpec",
    "Caps",
    "ClaudeCliProvider",
    "CodingAgentProvider",
    "DangerousFlagRequested",
    "DisabledProvider",
    "get_provider",
    # state machine
    "DevelopmentMode",
    "DevJob",
    "DevState",
    "Authority",
    "DEVELOP_AND_TEST_AUTHORITIES",
    "FORBIDDEN_AUTHORITIES",
    "DevModeError",
    "NotOwnerTriggered",
    "SingleFlightViolation",
    "ActivationNotConfirmed",
    "AuthorityDenied",
    "InvalidTransition",
]
