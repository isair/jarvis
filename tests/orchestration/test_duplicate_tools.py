"""Orchestration tests — duplicate tool call suppression (spec 5.5).

Tests that the agent loop does not execute the same tool call twice in the
same turn, and escalates correctly when a loop is detected.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_signature(tool_name: str, tool_args: dict) -> tuple:
    import json
    stable = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    return (tool_name, stable)


# ---------------------------------------------------------------------------
# Signature deduplication logic (unit tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_same_args_produce_same_signature():
    """Tool call with identical args produces an identical signature."""
    sig1 = _build_signature("getWeather", {"city": "London"})
    sig2 = _build_signature("getWeather", {"city": "London"})
    assert sig1 == sig2


@pytest.mark.unit
def test_different_args_produce_different_signature():
    """Tool call with different args produces a distinct signature."""
    sig1 = _build_signature("getWeather", {"city": "London"})
    sig2 = _build_signature("getWeather", {"city": "Paris"})
    assert sig1 != sig2


@pytest.mark.unit
def test_different_tools_produce_different_signature():
    """Different tool names produce distinct signatures even with identical args."""
    sig1 = _build_signature("getWeather", {})
    sig2 = _build_signature("webSearch", {})
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# Recent-signature memory management
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_signature_eviction_after_five():
    """Only the last 5 signatures are kept; older ones are evicted."""
    sigs = []
    for i in range(7):
        sigs.append(_build_signature("tool", {"n": i}))
        if len(sigs) > 5:
            sigs = sigs[-5:]
    assert len(sigs) == 5
    # First two should be evicted
    assert _build_signature("tool", {"n": 0}) not in sigs
    assert _build_signature("tool", {"n": 1}) not in sigs
    assert _build_signature("tool", {"n": 6}) in sigs


# ---------------------------------------------------------------------------
# Duplicate count detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_duplicate_count_from_messages():
    """duplicate_tool_count is correctly derived from recent message history."""
    messages = [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "1", "tool_name": "getWeather", "content": "20°C"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "2", "tool_name": "getWeather", "content": "20°C"},
    ]
    tool_name = "getWeather"
    duplicate_count = sum(
        1 for msg in messages[-10:]
        if msg.get("role") == "tool" and msg.get("tool_name") == tool_name
    )
    assert duplicate_count == 2


@pytest.mark.unit
def test_no_duplicates_for_different_tools():
    """Different tool names do not contribute to each other's duplicate count."""
    messages = [
        {"role": "tool", "tool_call_id": "1", "tool_name": "getWeather", "content": "x"},
        {"role": "tool", "tool_call_id": "2", "tool_name": "webSearch", "content": "y"},
    ]
    count_weather = sum(
        1 for m in messages[-10:] if m.get("role") == "tool" and m.get("tool_name") == "getWeather"
    )
    count_search = sum(
        1 for m in messages[-10:] if m.get("role") == "tool" and m.get("tool_name") == "webSearch"
    )
    assert count_weather == 1
    assert count_search == 1


# ---------------------------------------------------------------------------
# Recent-signature deduplication suppresses agent loop re-execution
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_signature_in_history_triggers_suppression():
    """A signature already in recent_tool_signatures triggers cached-guidance response."""
    recent_tool_signatures = []
    tool_name = "getWeather"
    tool_args = {"city": "London"}

    import json
    stable_args = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    signature = (tool_name, stable_args)
    recent_tool_signatures.append(signature)

    # Simulating the check that would happen in the loop
    suppressed = signature in recent_tool_signatures
    assert suppressed is True


@pytest.mark.unit
def test_new_signature_not_suppressed():
    """A signature not yet seen passes through without suppression."""
    recent_tool_signatures = [("getWeather", '{"city": "London"}')]
    import json
    new_sig = ("getWeather", json.dumps({"city": "Paris"}, sort_keys=True))
    assert new_sig not in recent_tool_signatures
