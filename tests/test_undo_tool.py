"""Unit tests for the UndoTool builtin."""

import pytest
from unittest.mock import patch, MagicMock

from jarvis.tools.builtin.undo import UndoTool
from jarvis.tools.base import ToolContext
from jarvis.tools.types import RiskLevel, ToolExecutionResult
from jarvis.undo_registry import UndoEntry


@pytest.fixture
def undo_tool():
    return UndoTool()


@pytest.fixture
def context():
    return ToolContext(
        db=MagicMock(),
        cfg=MagicMock(),
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=1,
        user_print=lambda msg: None,
    )


def _make_entry(step_id="abc123", description="wrote notes.txt"):
    return UndoEntry(
        step_id=step_id,
        description=description,
        tool_name="localFiles",
        tool_args={"operation": "write", "path": "notes.txt"},
        undo_tool="localFiles",
        undo_args={"operation": "write", "path": "notes.txt", "content": "original"},
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_undo_tool_name(undo_tool):
    assert undo_tool.name == "undo"


@pytest.mark.unit
def test_undo_tool_risk_is_moderate(undo_tool):
    assert undo_tool.assess_risk() == RiskLevel.MODERATE
    assert undo_tool.assess_risk({"count": 3}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_undo_tool_schema_has_count_and_step_id(undo_tool):
    props = undo_tool.inputSchema["properties"]
    assert "count" in props
    assert "step_id" in props


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo", return_value=[])
def test_nothing_to_undo(mock_pop, undo_tool, context):
    result = undo_tool.run({}, context)
    assert result.success is True
    assert "nothing to undo" in result.reply_text.lower()


# ---------------------------------------------------------------------------
# Single undo (default count=1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo")
@patch("jarvis.tools.registry.run_tool_with_retries")
def test_single_undo_executes_reversal(mock_run, mock_pop, undo_tool, context):
    entry = _make_entry()
    mock_pop.return_value = [entry]
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="File written.", error_message=None
    )

    result = undo_tool.run({}, context)

    assert result.success is True
    assert "Reversed" in result.reply_text
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["tool_name"] == "localFiles"
    assert call_kwargs["tool_args"]["content"] == "original"


# ---------------------------------------------------------------------------
# Count parameter
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo")
@patch("jarvis.tools.registry.run_tool_with_retries")
def test_count_undoes_n_entries(mock_run, mock_pop, undo_tool, context):
    entries = [_make_entry(step_id=f"id_{i}", description=f"action {i}") for i in range(3)]
    mock_pop.return_value = entries
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="done", error_message=None
    )

    result = undo_tool.run({"count": 3}, context)

    assert result.success is True
    assert mock_run.call_count == 3
    mock_pop.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# step_id parameter
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_undo_by_id")
@patch("jarvis.tools.registry.run_tool_with_retries")
def test_step_id_targets_specific_entry(mock_run, mock_pop_id, undo_tool, context):
    entry = _make_entry(step_id="specific-id")
    mock_pop_id.return_value = entry
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="done", error_message=None
    )

    result = undo_tool.run({"step_id": "specific-id"}, context)

    assert result.success is True
    assert "Reversed" in result.reply_text
    mock_pop_id.assert_called_once_with("specific-id")


@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_undo_by_id", return_value=None)
def test_step_id_not_found(mock_pop_id, undo_tool, context):
    result = undo_tool.run({"step_id": "nonexistent"}, context)

    assert result.success is True
    assert "nothing to undo" in result.reply_text.lower()


# ---------------------------------------------------------------------------
# Reversal failure handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo")
@patch("jarvis.tools.registry.run_tool_with_retries")
def test_reversal_tool_error_is_reported(mock_run, mock_pop, undo_tool, context):
    entry = _make_entry()
    mock_pop.return_value = [entry]
    mock_run.return_value = ToolExecutionResult(
        success=False, reply_text=None, error_message="disk full"
    )

    result = undo_tool.run({}, context)

    assert result.success is True  # Tool itself succeeds, reports the failure
    assert "could not reverse" in result.reply_text.lower() or "disk full" in result.reply_text.lower()


@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo")
@patch("jarvis.tools.registry.run_tool_with_retries", side_effect=RuntimeError("boom"))
def test_reversal_exception_is_caught(mock_run, mock_pop, undo_tool, context):
    entry = _make_entry()
    mock_pop.return_value = [entry]

    result = undo_tool.run({}, context)

    assert result.success is True
    assert "boom" in result.reply_text.lower()


# ---------------------------------------------------------------------------
# Count capping
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("jarvis.tools.builtin.undo.pop_last_undo", return_value=[])
def test_count_capped_at_20(mock_pop, undo_tool, context):
    undo_tool.run({"count": 100}, context)
    mock_pop.assert_called_once_with(20)
