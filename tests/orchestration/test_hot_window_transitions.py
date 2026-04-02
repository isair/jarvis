"""Orchestration tests — listening state machine hot-window transitions (spec 5.5).

Tests that the StateManager correctly handles:
- WAKE_WORD → COLLECTING transitions
- COLLECTING → WAKE_WORD on clear 
- HOT_WINDOW activation and expiry
- Duplicate collection protection
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(**kwargs):
    from jarvis.listening.state_manager import StateManager
    return StateManager(
        hot_window_seconds=kwargs.get("hot_window_seconds", 6.0),
        echo_tolerance=kwargs.get("echo_tolerance", 0.1),
        voice_collect_seconds=kwargs.get("voice_collect_seconds", 0.5),
        max_collect_seconds=kwargs.get("max_collect_seconds", 60.0),
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_initial_state_is_wake_word():
    """StateManager starts in WAKE_WORD mode."""
    from jarvis.listening.state_manager import ListeningState
    sm = _make_manager()
    assert sm.get_state() == ListeningState.WAKE_WORD


@pytest.mark.unit
def test_is_collecting_false_initially():
    """is_collecting() returns False before collection starts."""
    sm = _make_manager()
    assert sm.is_collecting() is False


@pytest.mark.unit
def test_is_hot_window_false_initially():
    """is_hot_window_active() returns False on startup."""
    sm = _make_manager()
    assert sm.is_hot_window_active() is False


# ---------------------------------------------------------------------------
# WAKE_WORD → COLLECTING transition
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_start_collection_enters_collecting_state():
    """start_collection() transitions state to COLLECTING."""
    from jarvis.listening.state_manager import ListeningState, StateManager
    sm = _make_manager()
    sm.start_collection("hello")
    assert sm.get_state() == ListeningState.COLLECTING
    assert sm.is_collecting() is True


@pytest.mark.unit
def test_start_collection_stores_initial_text():
    """start_collection() seeds the pending query with initial text."""
    sm = _make_manager()
    sm.start_collection("set a timer")
    assert sm.get_pending_query() == "set a timer"


@pytest.mark.unit
def test_add_to_collection_appends_text():
    """add_to_collection() appends a word to the pending query."""
    sm = _make_manager()
    sm.start_collection("turn")
    sm.add_to_collection("off the lights")
    assert "turn" in sm.get_pending_query()
    assert "off the lights" in sm.get_pending_query()


# ---------------------------------------------------------------------------
# COLLECTING → WAKE_WORD transition
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clear_collection_returns_query():
    """clear_collection() returns the accumulated query text."""
    sm = _make_manager()
    sm.start_collection("what time is it")
    result = sm.clear_collection()
    assert result == "what time is it"


@pytest.mark.unit
def test_clear_collection_resets_to_wake_word():
    """clear_collection() transitions back to WAKE_WORD state."""
    from jarvis.listening.state_manager import ListeningState
    sm = _make_manager()
    sm.start_collection("query")
    sm.clear_collection()
    assert sm.get_state() == ListeningState.WAKE_WORD


@pytest.mark.unit
def test_add_to_collection_does_nothing_when_not_collecting():
    """add_to_collection() is a no-op when not in COLLECTING state."""
    sm = _make_manager()
    sm.add_to_collection("stray text")
    assert sm.get_pending_query() == ""


# ---------------------------------------------------------------------------
# Collection silence timeout
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_collection_timeout_triggers_on_silence(monkeypatch):
    """check_collection_timeout() returns True after silence exceeds threshold."""
    sm = _make_manager(voice_collect_seconds=0.1)
    sm.start_collection("test")
    # Advance last_voice_time into the past
    sm._last_voice_time = time.time() - 0.5
    assert sm.check_collection_timeout() is True


@pytest.mark.unit
def test_collection_timeout_false_before_silence_threshold():
    """check_collection_timeout() returns False when still within silence window."""
    sm = _make_manager(voice_collect_seconds=30.0)
    sm.start_collection("fast response")
    assert sm.check_collection_timeout() is False


# ---------------------------------------------------------------------------
# HOT_WINDOW state
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hot_window_voice_state_capture():
    """capture_hot_window_state_at_voice_start() captures HOT_WINDOW state."""
    from jarvis.listening.state_manager import ListeningState
    sm = _make_manager()
    # Force state to HOT_WINDOW
    with sm._state_lock:
        sm._state = ListeningState.HOT_WINDOW
        sm._hot_window_start_time = time.time()
    sm.capture_hot_window_state_at_voice_start()
    assert sm.was_hot_window_active_at_voice_start() is True


@pytest.mark.unit
def test_hot_window_voice_state_not_captured_in_wake_word():
    """capture_hot_window_state_at_voice_start() records False when not in HOT_WINDOW."""
    sm = _make_manager()
    sm.capture_hot_window_state_at_voice_start()
    assert sm.was_hot_window_active_at_voice_start() is False


@pytest.mark.unit
def test_hot_window_expiry_returns_to_wake_word(monkeypatch):
    """expire_hot_window() transitions from HOT_WINDOW to WAKE_WORD."""
    from jarvis.listening.state_manager import ListeningState
    sm = _make_manager()
    with sm._state_lock:
        sm._state = ListeningState.HOT_WINDOW
        sm._hot_window_start_time = time.time() - 100.0  # already expired
    sm.expire_hot_window()
    assert sm.get_state() == ListeningState.WAKE_WORD


@pytest.mark.unit
def test_check_hot_window_expiry_returns_true_after_timeout(monkeypatch):
    """check_hot_window_expiry() returns True when the hot window has timed out."""
    from jarvis.listening.state_manager import ListeningState
    sm = _make_manager(hot_window_seconds=1.0)
    with sm._state_lock:
        sm._state = ListeningState.HOT_WINDOW
        sm._hot_window_start_time = time.time() - 5.0  # well past timeout
    expired = sm.check_hot_window_expiry()
    assert expired is True


@pytest.mark.unit
def test_check_hot_window_expiry_false_when_within_window():
    """check_hot_window_expiry() returns False within the hot window period."""
    from jarvis.listening.state_manager import ListeningState, StateManager
    sm = _make_manager(hot_window_seconds=60.0)
    with sm._state_lock:
        sm._state = ListeningState.HOT_WINDOW
        sm._hot_window_start_time = time.time()
    expired = sm.check_hot_window_expiry()
    assert expired is False
