"""Unit tests for the approval module.
Copyright 2026 sjackson0109

"""

import pytest

from jarvis.approval import (
    RiskLevel,
    RequestType,
    assess_risk,
    requires_approval,
    approval_prompt,
    classify_request,
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
    operational = [
        "delete the old log files",
        "write a summary to notes.txt",
        "save this conversation",
        "create a new file called report.md",
        "send an email to Alice",
        "log meal apple for lunch",
        "run the tests",
        "book a restaurant for tonight",
    ]
    for query in operational:
        result = classify_request(query)
        assert result == RequestType.OPERATIONAL, f"Expected OPERATIONAL for: '{query}'"


@pytest.mark.unit
def test_empty_query_is_informational():
    assert classify_request("") == RequestType.INFORMATIONAL
    assert classify_request(None) == RequestType.INFORMATIONAL
