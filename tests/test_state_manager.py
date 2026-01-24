"""
Tests for voice listening state manager.

These tests verify the state transitions, timer-based hot window management,
and query collection behavior.
"""

import time
import threading
import pytest
from unittest.mock import patch, MagicMock

from jarvis.listening.state_manager import StateManager, ListeningState


class TestStateTransitions:
    """Tests for basic state transitions."""

    def test_initial_state_is_wake_word(self):
        """State manager starts in WAKE_WORD state."""
        sm = StateManager()
        assert sm.get_state() == ListeningState.WAKE_WORD

    def test_start_collection_changes_state(self):
        """Starting collection changes state to COLLECTING."""
        sm = StateManager()
        sm.start_collection("hello")
        assert sm.get_state() == ListeningState.COLLECTING

    def test_clear_collection_returns_to_wake_word(self):
        """Clearing collection returns to WAKE_WORD state."""
        sm = StateManager()
        sm.start_collection("hello")
        sm.clear_collection()
        assert sm.get_state() == ListeningState.WAKE_WORD

    def test_is_collecting_helper(self):
        """is_collecting() accurately reflects state."""
        sm = StateManager()
        assert sm.is_collecting() is False
        sm.start_collection("test")
        assert sm.is_collecting() is True
        sm.clear_collection()
        assert sm.is_collecting() is False

    def test_is_hot_window_active_helper(self):
        """is_hot_window_active() accurately reflects state."""
        sm = StateManager()
        assert sm.is_hot_window_active() is False
        # Force hot window state for testing
        sm._state = ListeningState.HOT_WINDOW
        assert sm.is_hot_window_active() is True


class TestQueryCollection:
    """Tests for query collection functionality."""

    def test_start_collection_stores_initial_text(self):
        """Starting collection stores initial text."""
        sm = StateManager()
        sm.start_collection("hello world")
        assert sm.get_pending_query() == "hello world"

    def test_add_to_collection_appends_text(self):
        """Adding to collection appends text."""
        sm = StateManager()
        sm.start_collection("hello")
        sm.add_to_collection("world")
        assert sm.get_pending_query() == "hello world"

    def test_add_to_collection_only_works_when_collecting(self):
        """Adding to collection only works in COLLECTING state."""
        sm = StateManager()
        sm.add_to_collection("ignored")
        assert sm.get_pending_query() == ""

    def test_clear_collection_returns_query(self):
        """Clearing collection returns the accumulated query."""
        sm = StateManager()
        sm.start_collection("hello")
        sm.add_to_collection("world")
        query = sm.clear_collection()
        assert query == "hello world"
        assert sm.get_pending_query() == ""

    def test_silence_timeout_triggers_collection_complete(self):
        """Collection times out after silence period."""
        sm = StateManager(voice_collect_seconds=0.05)  # 50ms timeout
        sm.start_collection("test")

        # Initially no timeout
        assert sm.check_collection_timeout() is False

        # Wait for timeout
        time.sleep(0.06)
        assert sm.check_collection_timeout() is True

    def test_max_duration_timeout(self):
        """Collection times out after max duration."""
        sm = StateManager(max_collect_seconds=0.05)  # 50ms max
        sm.start_collection("test")

        # Keep adding to prevent silence timeout
        for _ in range(3):
            time.sleep(0.02)
            sm.add_to_collection("more")

        assert sm.check_collection_timeout() is True


class TestHotWindowActivation:
    """Tests for hot window activation timer."""

    def test_schedule_hot_window_activation(self):
        """Hot window activates after echo tolerance delay."""
        sm = StateManager(echo_tolerance=0.05, hot_window_seconds=1.0)

        # Patch print to avoid test output
        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Not active immediately
            assert sm.is_hot_window_active() is False

            # Wait for activation
            time.sleep(0.1)
            assert sm.is_hot_window_active() is True

        sm.stop()

    def test_cancel_hot_window_activation(self):
        """Can cancel pending hot window activation."""
        sm = StateManager(echo_tolerance=0.1, hot_window_seconds=1.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Cancel before activation
            time.sleep(0.02)
            sm.cancel_hot_window_activation()

            # Wait past activation time
            time.sleep(0.15)
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_hot_window_not_activated_during_collection(self):
        """Hot window doesn't activate if already collecting."""
        sm = StateManager(echo_tolerance=0.05, hot_window_seconds=1.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Start collection before activation
            time.sleep(0.02)
            sm.start_collection("new query")

            # Wait past activation time
            time.sleep(0.1)

            # Should still be in COLLECTING, not HOT_WINDOW
            assert sm.get_state() == ListeningState.COLLECTING

        sm.stop()


class TestHotWindowExpiry:
    """Tests for hot window expiry timer."""

    def test_hot_window_expires_after_duration(self):
        """Hot window expires after configured duration."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.05)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Wait for activation
            time.sleep(0.04)
            assert sm.is_hot_window_active() is True

            # Wait for expiry
            time.sleep(0.1)
            assert sm.is_hot_window_active() is False
            assert sm.get_state() == ListeningState.WAKE_WORD

        sm.stop()

    def test_manual_expire_hot_window(self):
        """Can manually expire hot window."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=10.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            time.sleep(0.04)
            assert sm.is_hot_window_active() is True

            sm.expire_hot_window()
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_check_hot_window_expiry_fallback(self):
        """check_hot_window_expiry provides synchronous expiry check."""
        sm = StateManager(echo_tolerance=0.0, hot_window_seconds=0.05)

        with patch('builtins.print'):
            # Manually set hot window state
            sm._state = ListeningState.HOT_WINDOW
            sm._hot_window_start_time = time.time()

            # Not expired yet
            assert sm.check_hot_window_expiry() is False

            # Wait for expiry
            time.sleep(0.06)
            assert sm.check_hot_window_expiry() is True
            assert sm.get_state() == ListeningState.WAKE_WORD


class TestHotWindowVoiceState:
    """Tests for hot window voice state capture."""

    def test_capture_hot_window_state_when_active(self):
        """Captures that hot window was active when voice started."""
        sm = StateManager()
        sm._state = ListeningState.HOT_WINDOW

        sm.capture_hot_window_state_at_voice_start()
        assert sm.was_hot_window_active_at_voice_start() is True

    def test_capture_hot_window_state_when_inactive(self):
        """Captures that hot window was NOT active when voice started."""
        sm = StateManager()
        sm._state = ListeningState.WAKE_WORD

        sm.capture_hot_window_state_at_voice_start()
        assert sm.was_hot_window_active_at_voice_start() is False

    def test_clear_hot_window_voice_state(self):
        """Can clear the captured hot window voice state."""
        sm = StateManager()
        sm._state = ListeningState.HOT_WINDOW

        sm.capture_hot_window_state_at_voice_start()
        assert sm.was_hot_window_active_at_voice_start() is True

        sm.clear_hot_window_voice_state()
        assert sm.was_hot_window_active_at_voice_start() is False


class TestStopBehavior:
    """Tests for state manager stop behavior."""

    def test_stop_cancels_all_timers(self):
        """Stopping state manager cancels all pending timers."""
        sm = StateManager(echo_tolerance=1.0, hot_window_seconds=1.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Verify timer is scheduled
            assert sm._hot_window_activation_timer is not None

            sm.stop()

            # Timer should be cancelled
            assert sm._hot_window_activation_timer is None
            assert sm._should_stop is True

    def test_stop_resets_state(self):
        """Stopping state manager resets to WAKE_WORD."""
        sm = StateManager()
        sm._state = ListeningState.HOT_WINDOW

        sm.stop()
        assert sm.get_state() == ListeningState.WAKE_WORD


class TestThreadSafety:
    """Tests for thread safety of state operations."""

    def test_concurrent_state_access(self):
        """State operations are thread-safe."""
        sm = StateManager(voice_collect_seconds=10.0)
        errors = []

        def reader():
            for _ in range(100):
                try:
                    _ = sm.get_state()
                    _ = sm.is_collecting()
                    _ = sm.get_pending_query()
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(100):
                try:
                    if i % 2 == 0:
                        sm.start_collection(f"test {i}")
                    else:
                        sm.clear_collection()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        sm.stop()
