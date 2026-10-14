"""
Tests for voice listening state manager.

These tests verify the state transitions, timer-based hot window management,
and query collection behaviour.
"""

import threading
from types import SimpleNamespace
import pytest
from unittest.mock import patch

from jarvis.listening.state_manager import StateManager, ListeningState

pytestmark = pytest.mark.unit


@pytest.fixture
def collection_clock(monkeypatch):
    """Advance collection deadlines without scheduling or wall-clock sleeps."""
    from jarvis.listening import state_manager

    now = [1000.0]
    monkeypatch.setattr(state_manager, "time", SimpleNamespace(time=lambda: now[0]))

    def advance(seconds):
        now[0] += seconds

    return advance


class TestStateTransitions:
    """Tests for basic state transitions."""

    def test_initial_state_is_wake_word(self):
        """State manager starts in WAKE_WORD state."""
        sm = StateManager()
        assert sm.get_state() == ListeningState.WAKE_WORD

    def test_start_collection_changes_state(self):
        """Starting collection changes state to COLLECTING."""
        sm = StateManager()
        sm.start_collection("hello", language=None)
        assert sm.get_state() == ListeningState.COLLECTING

    def test_clear_collection_returns_to_wake_word(self):
        """Clearing collection returns to WAKE_WORD state."""
        sm = StateManager()
        sm.start_collection("hello", language=None)
        sm.clear_collection()
        assert sm.get_state() == ListeningState.WAKE_WORD

    def test_is_collecting_helper(self):
        """is_collecting() accurately reflects state."""
        sm = StateManager()
        assert sm.is_collecting() is False
        sm.start_collection("test", language=None)
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
        sm.start_collection("hello world", language=None)
        assert sm.get_pending_query() == "hello world"

    def test_add_to_collection_appends_text(self):
        """Adding to collection appends text."""
        sm = StateManager()
        sm.start_collection("hello", language=None)
        sm.add_to_collection("world", language=None)
        assert sm.get_pending_query() == "hello world"

    def test_add_to_collection_only_works_when_collecting(self):
        """Adding to collection only works in COLLECTING state."""
        sm = StateManager()
        sm.add_to_collection("ignored", language=None)
        assert sm.get_pending_query() == ""

    def test_clear_collection_returns_query(self):
        """Clearing collection returns the accumulated query."""
        sm = StateManager()
        sm.start_collection("hello", language=None)
        sm.add_to_collection("world", language=None)
        query = sm.clear_collection()
        assert query.text == "hello world"
        assert query.language is None
        assert sm.get_pending_query() == ""

    def test_silence_timeout_triggers_collection_complete(self, collection_clock):
        """Collection times out after silence period."""
        sm = StateManager(voice_collect_seconds=0.05)  # 50ms timeout
        sm.start_collection("test", language=None)

        # Initially no timeout
        assert sm.check_collection_timeout() is False

        collection_clock(sm.voice_collect_seconds * 1.25)
        assert sm.check_collection_timeout() is True

    def test_max_duration_timeout(self, collection_clock):
        """Collection times out after max duration."""
        sm = StateManager(max_collect_seconds=0.05)  # 50ms max
        sm.start_collection("test", language=None)

        # Keep adding to prevent silence timeout
        for _ in range(3):
            collection_clock(sm.max_collect_seconds / 2)
            sm.add_to_collection("more", language=None)

        assert sm.check_collection_timeout() is True


@pytest.fixture
def hot_clock(monkeypatch):
    """Run real hot-window callbacks against an explicitly advanced clock."""
    from jarvis.listening import state_manager

    clock = SimpleNamespace(now=1000.0)
    pending = []

    class Timer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.cancelled = False

        def start(self):
            self.deadline = clock.now + self.interval
            pending.append(self)

        def cancel(self):
            self.cancelled = True

    def advance(seconds):
        target = clock.now + seconds
        while pending:
            timer = min(pending, key=lambda item: item.deadline)
            if timer.deadline > target:
                break
            pending.remove(timer)
            clock.now = timer.deadline
            if not timer.cancelled:
                timer.callback()
        clock.now = target

    clock.advance = advance
    monkeypatch.setattr(state_manager, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(state_manager, "threading", SimpleNamespace(RLock=threading.RLock, Timer=Timer))
    return clock


class TestHotWindowActivation:
    """Tests for hot window activation timer."""

    def test_schedule_hot_window_activation(self, hot_clock):
        """Hot window activates after echo tolerance delay."""
        sm = StateManager(echo_tolerance=0.05, hot_window_seconds=1.0)

        # Patch print to avoid test output
        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Not active immediately
            assert sm.is_hot_window_active() is False

            # Wait for activation
            hot_clock.advance(0.1)
            assert sm.is_hot_window_active() is True

        sm.stop()

    def test_cancel_hot_window_activation(self, hot_clock):
        """Can cancel pending hot window activation."""
        sm = StateManager(echo_tolerance=0.1, hot_window_seconds=1.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Cancel before activation
            hot_clock.advance(0.02)
            sm.cancel_hot_window_activation()

            # Wait past activation time
            hot_clock.advance(0.15)
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_hot_window_not_activated_during_collection(self, hot_clock):
        """Hot window doesn't activate if already collecting."""
        sm = StateManager(echo_tolerance=0.05, hot_window_seconds=1.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Start collection before activation
            hot_clock.advance(0.02)
            sm.start_collection("new query", language=None)

            # Wait past activation time
            hot_clock.advance(0.1)

            # Should still be in COLLECTING, not HOT_WINDOW
            assert sm.get_state() == ListeningState.COLLECTING

        sm.stop()


class TestHotWindowExpiry:
    """Tests for hot window expiry timer."""

    def test_hot_window_expires_after_duration(self, hot_clock):
        """Hot window expires after configured duration."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.05)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()

            # Wait for activation
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Wait for expiry
            hot_clock.advance(0.1)
            assert sm.is_hot_window_active() is False
            assert sm.get_state() == ListeningState.WAKE_WORD

        sm.stop()

    def test_manual_expire_hot_window(self, hot_clock):
        """Can manually expire hot window."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=10.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            sm.expire_hot_window()
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_reset_hot_window_expiry_extends_timer(self, hot_clock):
        """reset_hot_window_expiry restarts the timer so echo time doesn't eat the window."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.10)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Wait until most of the window has elapsed
            hot_clock.advance(0.07)
            assert sm.is_hot_window_active() is True  # still within 0.10s

            # Reset the timer (simulating echo rejection)
            sm.reset_hot_window_expiry()

            # After the original window would have expired, it should still be active
            hot_clock.advance(0.05)
            assert sm.is_hot_window_active() is True

            # Wait for the full reset window to expire
            hot_clock.advance(0.07)
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_reset_hot_window_expiry_reactivates_expired_window(self, hot_clock):
        """reset_hot_window_expiry reactivates a hot window that expired during echo processing."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.08)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Let the hot window fully expire
            hot_clock.advance(0.12)
            assert sm.get_state() == ListeningState.WAKE_WORD

            # Simulate echo rejection arriving after expiry — should reactivate
            sm.reset_hot_window_expiry()
            assert sm.is_hot_window_active() is True

            # New timer should keep it alive for another full window
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Then expire normally
            hot_clock.advance(0.06)
            assert sm.is_hot_window_active() is False

        sm.stop()

    def test_reset_hot_window_expiry_noop_when_collecting(self, hot_clock):
        """reset_hot_window_expiry does not interfere with COLLECTING state."""
        sm = StateManager()
        sm.start_collection("test query", language=None)
        assert sm.get_state() == ListeningState.COLLECTING

        sm.reset_hot_window_expiry()
        assert sm.get_state() == ListeningState.COLLECTING
        sm.stop()

    def test_check_hot_window_expiry_fallback(self, hot_clock):
        """check_hot_window_expiry provides synchronous expiry check."""
        sm = StateManager(echo_tolerance=0.0, hot_window_seconds=0.05)

        with patch('builtins.print'):
            # Manually set hot window state
            sm._state = ListeningState.HOT_WINDOW
            sm._hot_window_start_time = hot_clock.now

            # Not expired yet
            assert sm.check_hot_window_expiry() is False

            # Wait for expiry
            hot_clock.advance(0.06)
            assert sm.check_hot_window_expiry() is True
            assert sm.get_state() == ListeningState.WAKE_WORD


class TestTimestampBasedHotWindowDetection:
    """Tests for timestamp-based hot window detection.

    Instead of capturing a mutable boolean at VAD onset (which gets cleared
    by timer-based expiry before Whisper finishes), we compare the utterance
    start time against the hot window's time span. This eliminates race
    conditions between the expiry timer and Whisper transcription."""

    @pytest.mark.parametrize('phase,start,end,expected', [
        ('pending', None, None, True),
        ('pending', -1, -0.1, False),
        ('active', None, None, True),
        ('active', -1, -0.1, True),
        ('expired', -1, -0.1, False),
        ('expired', -1, 0.5, True),
        ('expired', 0.5, 2, True),
        ('expired', -1, 0, True),
        ('expired', 1, 2, True),
        ('expired', 1.1, 2, False),
        ('expired', None, None, False),
    ], ids=['pending-no-times', 'pending-before-span', 'active-no-times',
            'active-state-fallback', 'tts-only', 'overlaps-span',
            'transcription-after-expiry', 'ends-at-opening', 'starts-at-expiry',
            'after-expiry', 'expired-no-times'])
    def test_capture_times_govern_delayed_follow_up(self, hot_clock, phase, start, end, expected):
        manager = StateManager(echo_tolerance=0.2, hot_window_seconds=1.0)
        opened = hot_clock.now
        span = manager.echo_tolerance + manager.hot_window_seconds
        try:
            manager.schedule_hot_window_activation()
            if phase == 'active':
                hot_clock.advance(manager.echo_tolerance + manager.hot_window_seconds / 2)
            elif phase == 'expired':
                hot_clock.advance(span + manager.hot_window_seconds)
                assert not manager.is_hot_window_active()
            start_time = 0.0 if start is None else opened + start * span
            end_time = 0.0 if end is None else opened + end * span

            assert manager.was_speech_during_hot_window(start_time, end_time) is expected
        finally:
            manager.stop()

    def test_speech_during_active_window_detected(self, hot_clock):
        """Speech starting while hot window is active returns True."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=3.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Speech starts now, during active window
            speech_start = hot_clock.now
            assert sm.was_speech_during_hot_window(speech_start) is True

        sm.stop()

    def test_speech_before_window_not_detected(self, hot_clock):
        """Speech starting before the hot window span returns False."""
        sm = StateManager(echo_tolerance=0.5, hot_window_seconds=3.0)

        # Speech started before any window was scheduled
        old_time = hot_clock.now - 10.0
        assert sm.was_speech_during_hot_window(old_time) is False
        sm.stop()

    def test_speech_during_pending_activation_detected(self, hot_clock):
        """Speech starting during echo_tolerance delay (pending) returns True."""
        sm = StateManager(echo_tolerance=1.0, hot_window_seconds=3.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            # State is still WAKE_WORD, but activation timer is pending
            assert sm.get_state() == ListeningState.WAKE_WORD

            speech_start = hot_clock.now
            assert sm.was_speech_during_hot_window(speech_start) is True

        sm.stop()

    def test_speech_after_expiry_not_detected(self, hot_clock):
        """Speech starting after hot window expired returns False."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.05)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Wait for expiry
            hot_clock.advance(0.08)
            assert sm.is_hot_window_active() is False

            # Speech starts AFTER expiry
            speech_start = hot_clock.now
            assert sm.was_speech_during_hot_window(speech_start) is False

        sm.stop()

    def test_speech_during_window_detected_after_expiry(self, hot_clock):
        """Speech that STARTED during window is detected even after expiry.

        This is the core fix: Whisper takes time to transcribe, so the
        transcript arrives after the window expired. But the speech started
        during the window, so it should be treated as hot window input.
        """
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.08)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Speech starts during active window
            speech_start = hot_clock.now

            # Window expires while "Whisper is transcribing"
            hot_clock.advance(0.10)
            assert sm.is_hot_window_active() is False

            # Transcript arrives — but speech_start was during the window
            assert sm.was_speech_during_hot_window(speech_start) is True

        sm.stop()

    def test_no_timestamp_falls_back_to_current_state(self, hot_clock):
        """When utterance_start_time is 0, falls back to current state."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=3.0)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.was_speech_during_hot_window(0.0) is True

        sm.stop()

    def test_no_timestamp_after_expiry_returns_false(self, hot_clock):
        """When utterance_start_time is 0 and window expired, returns False."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.05)

        with patch('builtins.print'):
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            hot_clock.advance(0.08)
            assert sm.was_speech_during_hot_window(0.0) is False

        sm.stop()

    def test_new_window_resets_old_span(self, hot_clock):
        """A new hot window span doesn't match speech from before it."""
        sm = StateManager(echo_tolerance=0.02, hot_window_seconds=0.05)

        with patch('builtins.print'):
            # First window
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            hot_clock.advance(0.08)
            assert sm.is_hot_window_active() is False

            # Speech between windows
            between_speech = hot_clock.now

            # Second window
            hot_clock.advance(0.05)
            sm.schedule_hot_window_activation()
            hot_clock.advance(0.04)
            assert sm.is_hot_window_active() is True

            # Wait for second window to expire
            hot_clock.advance(0.08)
            assert sm.is_hot_window_active() is False

            # Speech from between windows should NOT match the second window's span
            assert sm.was_speech_during_hot_window(between_speech) is False

        sm.stop()


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
                        sm.start_collection(f"test {i}", language=None)
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
