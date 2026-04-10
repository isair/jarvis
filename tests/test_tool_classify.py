"""Unit tests for tool classification via the classify() method on Tool base class."""

import pytest

from jarvis.tools.registry import BUILTIN_TOOLS
from jarvis.policy.models import ToolClass


# ---------------------------------------------------------------------------
# Informational tools
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("tool_name", [
    "screenshot", "recallConversation", "stop", "fetchMeals",
])
def test_informational_tools(tool_name):
    tool = BUILTIN_TOOLS[tool_name]
    assert tool.classify() == ToolClass.INFORMATIONAL


# ---------------------------------------------------------------------------
# Read-only operational tools
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("tool_name", [
    "getWeather", "webSearch", "fetchWebPage", "refreshMCPTools",
])
def test_read_only_operational_tools(tool_name):
    tool = BUILTIN_TOOLS[tool_name]
    assert tool.classify() == ToolClass.READ_ONLY_OPERATIONAL


# ---------------------------------------------------------------------------
# Write operational tools (default)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("tool_name", ["logMeal", "undo"])
def test_write_operational_tools(tool_name):
    tool = BUILTIN_TOOLS[tool_name]
    assert tool.classify() == ToolClass.WRITE_OPERATIONAL


# ---------------------------------------------------------------------------
# Destructive tools
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_delete_meal_is_destructive():
    tool = BUILTIN_TOOLS["deleteMeal"]
    assert tool.classify() == ToolClass.DESTRUCTIVE


# ---------------------------------------------------------------------------
# localFiles operation-dependent classification
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("operation,expected", [
    ("list", ToolClass.INFORMATIONAL),
    ("read", ToolClass.INFORMATIONAL),
    ("write", ToolClass.WRITE_OPERATIONAL),
    ("append", ToolClass.WRITE_OPERATIONAL),
    ("delete", ToolClass.DESTRUCTIVE),
])
def test_local_files_classify_by_operation(operation, expected):
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.classify({"operation": operation}) == expected


@pytest.mark.unit
def test_local_files_unknown_operation_defaults_to_write():
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.classify({"operation": "chmod"}) == ToolClass.WRITE_OPERATIONAL


@pytest.mark.unit
def test_local_files_no_args_defaults_to_write():
    tool = BUILTIN_TOOLS["localFiles"]
    assert tool.classify() == ToolClass.WRITE_OPERATIONAL
    assert tool.classify(None) == ToolClass.WRITE_OPERATIONAL
