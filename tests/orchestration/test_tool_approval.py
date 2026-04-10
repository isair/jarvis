"""Orchestration tests — policy engine evaluation under voice-first undo model.

Tests that PolicyEngine.evaluate() correctly handles the ACTIVE and DENY
policy modes.  Approval gates have been removed — Jarvis uses act-then-undo
for reversible actions and spoken warnings for irreversible ones.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(mode: str = "active"):
    """Return a configured PolicyEngine for the given mode."""
    from jarvis.policy.approvals import ApprovalStore
    from jarvis.policy.engine import PolicyEngine
    from jarvis.policy.models import PolicyMode

    store = ApprovalStore()

    class _FakeCfg:
        policy_mode = mode
        workspace_roots: list = []
        blocked_roots: list = []
        read_only_roots: list = []
        local_files_mode = "home_only"
        mcps: dict = {}

    return PolicyEngine(_FakeCfg(), store)


# ---------------------------------------------------------------------------
# ACTIVE mode (default) — act-then-undo, no blocking gates
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_active_permits_destructive():
    """ACTIVE mode permits destructive tools (undo model handles risk)."""
    engine = _make_engine("active")
    decision = engine.evaluate("localFiles", {"operation": "delete", "path": "/tmp/x.txt"})
    assert decision.allowed is True
    assert decision.approval_required is False


@pytest.mark.unit
def test_active_permits_informational():
    """ACTIVE mode permits informational tools."""
    engine = _make_engine("active")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is True


@pytest.mark.unit
def test_active_permits_write():
    """ACTIVE mode permits write operations (undo model handles risk)."""
    engine = _make_engine("active")
    decision = engine.evaluate("localFiles", {"operation": "write", "path": "/tmp/x.txt"})
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# Legacy mode names map to ACTIVE
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_legacy_always_allow_maps_to_active():
    """Legacy 'always_allow' config value maps to ACTIVE mode."""
    engine = _make_engine("always_allow")
    decision = engine.evaluate("localFiles", {"operation": "delete", "path": "/tmp/x.txt"})
    assert decision.allowed is True


@pytest.mark.unit
def test_legacy_ask_destructive_maps_to_active():
    """Legacy 'ask_destructive' config value maps to ACTIVE mode."""
    engine = _make_engine("ask_destructive")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is True
    assert decision.approval_required is False


# ---------------------------------------------------------------------------
# DENY mode (kill-switch)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_deny_blocks_any_tool():
    """DENY mode blocks every tool call."""
    engine = _make_engine("deny")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is False
    assert decision.denied_reason


@pytest.mark.unit
def test_deny_blocks_write_tool():
    """DENY mode blocks write operations."""
    engine = _make_engine("deny")
    decision = engine.evaluate("localFiles", {"operation": "write", "path": "/tmp/x.txt"})
    assert decision.allowed is False


@pytest.mark.unit
def test_legacy_deny_all_maps_to_deny():
    """Legacy 'deny_all' config value maps to DENY mode."""
    engine = _make_engine("deny_all")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# Decision metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_decision_has_audit_id():
    """Every PolicyDecision carries a non-empty audit_id."""
    engine = _make_engine("active")
    decision = engine.evaluate("getWeather", {})
    assert decision.audit_id and len(decision.audit_id) > 0


@pytest.mark.unit
def test_decision_has_tool_class():
    """Every PolicyDecision carries a ToolClass classification."""
    from jarvis.policy.models import ToolClass
    engine = _make_engine("active")
    decision = engine.evaluate("getWeather", {})
    assert isinstance(decision.tool_class, ToolClass)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_module_evaluate_returns_permissive_when_unconfigured():
    """Module-level evaluate() is permissive when configure() hasn't been called."""
    import jarvis.policy.engine as eng
    # Save and clear the singleton
    original = eng._default_engine
    eng._default_engine = None
    try:
        decision = eng.evaluate("anyTool", {})
        assert decision.allowed is True
    finally:
        eng._default_engine = original


@pytest.mark.unit
def test_configure_returns_engine():
    """configure() returns a PolicyEngine instance."""
    from jarvis.policy.engine import configure
    from jarvis.policy.approvals import ApprovalStore

    class _Cfg:
        policy_mode = "active"
        workspace_roots: list = []
        blocked_roots: list = []
        read_only_roots: list = []
        local_files_mode = "home_only"
        mcps: dict = {}

    engine = configure(_Cfg(), ApprovalStore())
    assert engine is not None
