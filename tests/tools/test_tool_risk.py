"""Tests for Tool.assess_risk() — each tool declares its own risk level."""

import pytest

from jarvis.tools.types import RiskLevel
from jarvis.tools.registry import BUILTIN_TOOLS


# ---------------------------------------------------------------------------
# RiskLevel is now in tools.types, but approval re-exports it for compat
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_risk_level_importable_from_tools_types():
    from jarvis.tools.types import RiskLevel as RL
    assert RL.SAFE.value == "safe"
    assert RL.MODERATE.value == "moderate"
    assert RL.HIGH.value == "high"


@pytest.mark.unit
def test_risk_level_still_importable_from_approval_for_compat():
    """Backward-compat: RiskLevel must remain importable via jarvis.approval."""
    from jarvis.approval import RiskLevel as RL
    assert RL.SAFE.value == "safe"


# ---------------------------------------------------------------------------
# Safe read-only tools
# ---------------------------------------------------------------------------

_SAFE_TOOLS = [
    "screenshot",
    "webSearch",
    "fetchWebPage",
    "recallConversation",
    "fetchMeals",
    "getWeather",
    "refreshMCPTools",
    "stop",
]


@pytest.mark.unit
@pytest.mark.parametrize("tool_name", _SAFE_TOOLS)
def test_safe_tools_return_safe_risk(tool_name):
    tool = BUILTIN_TOOLS[tool_name]
    assert tool.assess_risk({}) == RiskLevel.SAFE
    assert tool.assess_risk(None) == RiskLevel.SAFE


# ---------------------------------------------------------------------------
# localFiles — operation-specific risk
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("operation", ["list", "read"])
def test_local_files_safe_operations(operation):
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.assess_risk({"operation": operation}) == RiskLevel.SAFE


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["write", "append"])
def test_local_files_moderate_operations(operation):
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.assess_risk({"operation": operation}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_local_files_delete_is_high():
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.assess_risk({"operation": "delete"}) == RiskLevel.HIGH


@pytest.mark.unit
def test_local_files_unknown_operation_is_moderate():
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.assess_risk({"operation": "chmod"}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_local_files_no_args_is_moderate():
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.assess_risk(None) == RiskLevel.MODERATE
    assert tool.assess_risk({}) == RiskLevel.MODERATE


# ---------------------------------------------------------------------------
# deleteMeal — always HIGH
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_delete_meal_is_high():
    tool = BUILTIN_TOOLS["deleteMeal"]
    assert tool.assess_risk({"id": 1}) == RiskLevel.HIGH
    assert tool.assess_risk(None) == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# logMeal — default MODERATE
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_log_meal_is_moderate():
    tool = BUILTIN_TOOLS["logMeal"]
    assert tool.assess_risk({"name": "apple"}) == RiskLevel.MODERATE


# ---------------------------------------------------------------------------
# assess_risk() fully drives approval.assess_risk() for builtins
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_approval_assess_risk_delegates_to_tool():
    """approval.assess_risk() must return the same value as the tool's own method."""
    from jarvis.approval import assess_risk

    for tool_name, tool in BUILTIN_TOOLS.items():
        # Use a generic empty args dict; tool-specific operation tests are above
        expected = tool.assess_risk({})
        actual = assess_risk(tool_name, {})
        assert actual == expected, (
            f"{tool_name}: approval.assess_risk returned {actual}, "
            f"expected {expected} from tool.assess_risk"
        )


@pytest.mark.unit
def test_approval_assess_risk_mcp_tool_is_moderate():
    from jarvis.approval import assess_risk
    assert assess_risk("myserver__doThing", {}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_approval_assess_risk_unknown_tool_is_moderate():
    from jarvis.approval import assess_risk
    assert assess_risk("brandNewTool", {}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_approval_assess_risk_none_is_safe():
    from jarvis.approval import assess_risk
    assert assess_risk(None, {}) == RiskLevel.SAFE
    assert assess_risk("", {}) == RiskLevel.SAFE
