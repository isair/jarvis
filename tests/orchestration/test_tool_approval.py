"""Orchestration tests — policy engine evaluation and tool approval flow (spec 5.1).

Tests that PolicyEngine.evaluate() correctly gates tool invocations
under different policy modes.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(mode: str = "ask_destructive"):
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
# ALWAYS_ALLOW mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_always_allow_permits_destructive():
    """ALWAYS_ALLOW mode permits destructive tools without approval."""
    engine = _make_engine("always_allow")
    decision = engine.evaluate("localFiles", {"operation": "delete", "path": "/tmp/x.txt"})
    assert decision.allowed is True
    assert decision.approval_required is False


@pytest.mark.unit
def test_always_allow_permits_informational():
    """ALWAYS_ALLOW mode permits informational tools."""
    engine = _make_engine("always_allow")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# DENY_ALL mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_deny_all_blocks_any_tool():
    """DENY_ALL mode blocks every tool call."""
    engine = _make_engine("deny_all")
    decision = engine.evaluate("getWeather", {})
    assert decision.allowed is False
    assert decision.denied_reason


@pytest.mark.unit
def test_deny_all_blocks_write_tool():
    """DENY_ALL mode blocks write operations."""
    engine = _make_engine("deny_all")
    decision = engine.evaluate("localFiles", {"operation": "write", "path": "/tmp/x.txt"})
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# ASK_DESTRUCTIVE mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ask_destructive_allows_read_tool(tmp_path):
    """ASK_DESTRUCTIVE permits read-only operations without approval."""
    from jarvis.policy.approvals import ApprovalStore
    from jarvis.policy.engine import PolicyEngine

    target = tmp_path / "notes.txt"
    target.write_text("data")

    class _FakeCfg:
        policy_mode = "ask_destructive"
        workspace_roots = [str(tmp_path)]
        blocked_roots: list = []
        read_only_roots: list = []
        local_files_mode = "workspace_only"
        mcps: dict = {}

    engine = PolicyEngine(_FakeCfg(), ApprovalStore())
    decision = engine.evaluate("localFiles", {"operation": "read", "path": str(target)})
    assert decision.allowed is True


@pytest.mark.unit
def test_ask_destructive_flags_delete_for_approval():
    """ASK_DESTRUCTIVE marks delete operations as requiring approval."""
    engine = _make_engine("ask_destructive")
    decision = engine.evaluate("localFiles", {"operation": "delete", "path": "/tmp/x.txt"})
    # Either denied outright or flagged for approval
    assert not decision.allowed or decision.approval_required


@pytest.mark.unit
def test_ask_destructive_allows_informational():
    """ASK_DESTRUCTIVE permits informational tools (weather, web search etc.)."""
    engine = _make_engine("ask_destructive")
    for tool in ("getWeather", "webSearch", "screenshot"):
        decision = engine.evaluate(tool, {})
        assert decision.allowed is True, f"Expected {tool} to be allowed"


# ---------------------------------------------------------------------------
# Decision metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_decision_has_audit_id():
    """Every PolicyDecision carries a non-empty audit_id."""
    engine = _make_engine("ask_destructive")
    decision = engine.evaluate("getWeather", {})
    assert decision.audit_id and len(decision.audit_id) > 0


@pytest.mark.unit
def test_decision_has_tool_class():
    """Every PolicyDecision carries a ToolClass classification."""
    from jarvis.policy.models import ToolClass
    engine = _make_engine("ask_destructive")
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
        policy_mode = "ask_destructive"
        workspace_roots: list = []
        blocked_roots: list = []
        read_only_roots: list = []
        local_files_mode = "home_only"
        mcps: dict = {}

    engine = configure(_Cfg(), ApprovalStore())
    assert engine is not None
