"""Unit tests for the approval module."""

import pytest

from jarvis.approval import (
    RiskLevel,
    RequestType,
    assess_risk,
    requires_approval,
    approval_prompt,
    classify_request,
    is_undoable,
    pre_execution_warning,
    post_execution_note,
    build_undo_args,
)


# ---------------------------------------------------------------------------
# assess_risk tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_safe_read_only_tools():
    for tool in ("screenshot", "recallConversation", "fetchMeals", "webSearch",
                 "fetchWebPage", "getWeather", "refreshMCPTools", "stop"):
        assert assess_risk(tool, {}) == RiskLevel.SAFE, f"Expected SAFE for {tool}"


@pytest.mark.unit
def test_log_meal_is_moderate():
    assert assess_risk("logMeal", {"name": "apple"}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_delete_meal_is_high():
    assert assess_risk("deleteMeal", {"id": 42}) == RiskLevel.HIGH


@pytest.mark.unit
def test_local_files_list_is_safe():
    assert assess_risk("localFiles", {"operation": "list"}) == RiskLevel.SAFE


@pytest.mark.unit
def test_local_files_read_is_safe():
    assert assess_risk("localFiles", {"operation": "read"}) == RiskLevel.SAFE


@pytest.mark.unit
def test_local_files_write_is_moderate():
    assert assess_risk("localFiles", {"operation": "write"}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_local_files_append_is_moderate():
    assert assess_risk("localFiles", {"operation": "append"}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_local_files_delete_is_high():
    assert assess_risk("localFiles", {"operation": "delete"}) == RiskLevel.HIGH


@pytest.mark.unit
def test_local_files_unknown_operation_is_moderate():
    assert assess_risk("localFiles", {"operation": "chmod"}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_mcp_tool_is_moderate():
    assert assess_risk("filesystem__readFile", {}) == RiskLevel.MODERATE
    assert assess_risk("myserver__doThing", {"x": 1}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_unknown_tool_is_moderate():
    assert assess_risk("brandNewTool", {}) == RiskLevel.MODERATE


@pytest.mark.unit
def test_empty_tool_name_is_safe():
    assert assess_risk("", {}) == RiskLevel.SAFE
    assert assess_risk(None, {}) == RiskLevel.SAFE


# ---------------------------------------------------------------------------
# requires_approval tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_requires_approval_for_high_risk():
    assert requires_approval("deleteMeal", {"id": 1}) is True
    assert requires_approval("localFiles", {"operation": "delete", "path": "~/notes"}) is True


@pytest.mark.unit
def test_no_approval_for_safe():
    assert requires_approval("webSearch", {"query": "weather"}) is False
    assert requires_approval("screenshot", {}) is False


@pytest.mark.unit
def test_no_approval_for_moderate():
    assert requires_approval("logMeal", {"name": "salad"}) is False
    assert requires_approval("localFiles", {"operation": "write"}) is False


# ---------------------------------------------------------------------------
# approval_prompt tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_approval_prompt_contains_tool_name():
    prompt = approval_prompt("deleteMeal", {"id": 5})
    assert "deleteMeal" in prompt


@pytest.mark.unit
def test_approval_prompt_mentions_risk():
    prompt = approval_prompt("localFiles", {"operation": "delete", "path": "~/data"})
    assert "high" in prompt.lower()


@pytest.mark.unit
def test_approval_prompt_asks_for_confirmation():
    prompt = approval_prompt("deleteMeal", {"id": 1})
    # Should contain some kind of confirmation request (language-neutral)
    lower = prompt.lower()
    assert "confirm" in lower or "approval" in lower or "proceed" in lower


# ---------------------------------------------------------------------------
# classify_request tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_informational_queries():
    informational = [
        "What is the weather in London?",
        "Tell me about Python programming",
        "How many calories are in an apple?",
        "What time is it?",
        "Who is the current president?",
    ]
    for query in informational:
        result = classify_request(query)
        assert result == RequestType.INFORMATIONAL, f"Expected INFORMATIONAL for: '{query}'"


@pytest.mark.unit
def test_operational_queries():
    """With a tool_name supplied, classification is OPERATIONAL regardless of text."""
    tool_invocations = [
        ("delete the old log files", "localFiles"),
        ("write a summary to notes.txt", "localFiles"),
        ("save this conversation", "writeFile"),
        ("create a new file called report.md", "localFiles"),
        ("log meal apple for lunch", "logMeal"),
        ("run the tests", "shell"),
        ("book a restaurant tonight", "mcp_restaurant"),
    ]
    for query, tool_name in tool_invocations:
        result = classify_request(query, tool_name=tool_name)
        assert result == RequestType.OPERATIONAL, (
            f"Expected OPERATIONAL when tool_name='{tool_name}' for: '{query}'"
        )


@pytest.mark.unit
def test_operational_without_tool_is_informational():
    """Without a tool_name, even action-sounding text is classified INFORMATIONAL.

    classify_request() is language-agnostic: it relies on tool presence, not
    keyword matching, so the same text may arrive in any language.
    """
    action_texts = [
        "delete the old log files",
        "write a summary to notes.txt",
        "send an email to Alice",
    ]
    for query in action_texts:
        result = classify_request(query)
        assert result == RequestType.INFORMATIONAL, (
            f"Expected INFORMATIONAL (no tool_name) for: '{query}'"
        )


@pytest.mark.unit
def test_empty_query_is_informational():
    assert classify_request("") == RequestType.INFORMATIONAL
    assert classify_request(None) == RequestType.INFORMATIONAL


# ---------------------------------------------------------------------------
# is_undoable tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_local_files_write_is_undoable():
    assert is_undoable("localFiles", {"operation": "write"}) is True


@pytest.mark.unit
def test_local_files_append_is_undoable():
    assert is_undoable("localFiles", {"operation": "append"}) is True


@pytest.mark.unit
def test_local_files_delete_is_undoable():
    assert is_undoable("localFiles", {"operation": "delete"}) is True


@pytest.mark.unit
def test_local_files_read_is_not_undoable():
    assert is_undoable("localFiles", {"operation": "read"}) is False


@pytest.mark.unit
def test_local_files_list_is_not_undoable():
    assert is_undoable("localFiles", {"operation": "list"}) is False


@pytest.mark.unit
def test_delete_meal_is_not_undoable():
    assert is_undoable("deleteMeal", {"id": 42}) is False


@pytest.mark.unit
def test_log_meal_is_not_undoable():
    assert is_undoable("logMeal", {"name": "apple"}) is False


@pytest.mark.unit
def test_unknown_tool_is_not_undoable():
    assert is_undoable("webSearch", {"query": "hello"}) is False


@pytest.mark.unit
def test_none_tool_is_not_undoable():
    assert is_undoable(None, {}) is False


# ---------------------------------------------------------------------------
# pre_execution_warning tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pre_execution_warning_for_high_irreversible():
    """deleteMeal is HIGH risk and NOT undoable -- must warn."""
    warning = pre_execution_warning("deleteMeal", {"id": 99})
    assert warning is not None
    assert "deleteMeal" in warning


@pytest.mark.unit
def test_pre_execution_warning_none_for_high_undoable():
    """localFiles/delete is HIGH but undoable -- no pre-warning (post-note instead)."""
    warning = pre_execution_warning("localFiles", {"operation": "delete", "path": "notes.txt"})
    assert warning is None


@pytest.mark.unit
def test_pre_execution_warning_none_for_safe_tools():
    assert pre_execution_warning("webSearch", {"query": "weather"}) is None
    assert pre_execution_warning("screenshot", {}) is None


@pytest.mark.unit
def test_pre_execution_warning_none_for_moderate_tools():
    assert pre_execution_warning("logMeal", {"name": "salad"}) is None
    assert pre_execution_warning("localFiles", {"operation": "write"}) is None


# ---------------------------------------------------------------------------
# post_execution_note tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_post_execution_note_for_high_undoable():
    """localFiles/delete is HIGH + undoable -- should return undo hint."""
    note = post_execution_note("localFiles", {"operation": "delete", "path": "notes.txt"})
    assert note is not None
    assert "undo" in note.lower()


@pytest.mark.unit
def test_post_execution_note_none_for_high_irreversible():
    """deleteMeal is HIGH but not undoable -- no post-note."""
    note = post_execution_note("deleteMeal", {"id": 5})
    assert note is None


@pytest.mark.unit
def test_post_execution_note_none_for_safe_tool():
    assert post_execution_note("webSearch", {"query": "hello"}) is None


@pytest.mark.unit
def test_post_execution_note_none_for_moderate_non_undoable():
    assert post_execution_note("logMeal", {"name": "banana"}) is None


# ---------------------------------------------------------------------------
# build_undo_args tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_undo_args_write_with_snapshot():
    result = build_undo_args(
        "localFiles",
        {"operation": "write", "path": "notes.txt"},
        snapshot="original content",
    )
    assert result is not None
    undo_tool, undo_args, description = result
    assert undo_tool == "localFiles"
    assert undo_args["operation"] == "write"
    assert undo_args["path"] == "notes.txt"
    assert undo_args["content"] == "original content"
    assert "notes.txt" in description


@pytest.mark.unit
def test_build_undo_args_append_with_snapshot():
    result = build_undo_args(
        "localFiles",
        {"operation": "append", "path": "log.txt"},
        snapshot="before-append contents",
    )
    assert result is not None
    _, undo_args, _ = result
    assert undo_args["content"] == "before-append contents"


@pytest.mark.unit
def test_build_undo_args_delete_with_snapshot():
    result = build_undo_args(
        "localFiles",
        {"operation": "delete", "path": "shopping_list.txt"},
        snapshot="milk\neggs\nbread",
    )
    assert result is not None
    undo_tool, undo_args, description = result
    assert undo_tool == "localFiles"
    assert undo_args["content"] == "milk\neggs\nbread"
    assert "shopping_list.txt" in description


@pytest.mark.unit
def test_build_undo_args_write_without_snapshot_returns_none():
    result = build_undo_args(
        "localFiles",
        {"operation": "write", "path": "notes.txt"},
        snapshot=None,
    )
    assert result is None


@pytest.mark.unit
def test_build_undo_args_for_non_undoable_tool():
    result = build_undo_args("deleteMeal", {"id": 1}, snapshot=None)
    assert result is None


@pytest.mark.unit
def test_build_undo_args_for_read_operation():
    """localFiles/read is not undoable -- should return None."""
    result = build_undo_args(
        "localFiles",
        {"operation": "read", "path": "notes.txt"},
        snapshot="some content",
    )
    assert result is None

