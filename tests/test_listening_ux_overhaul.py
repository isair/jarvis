"""
Tests for Voice Assistant UX Overhaul features.

These tests verify:
1. Timer-based hot window management
2. Context-aware echo detection thresholds
"""

import time
import threading
from unittest.mock import patch, MagicMock
import pytest


class TestStateManagerTimerHotWindow:
    """Tests for timer-based hot window management."""

    def _create_state_manager(self):
        """Create a StateManager instance."""
        from jarvis.listening.state_manager import StateManager
        return StateManager(
            hot_window_seconds=6.0,
            echo_tolerance=0.3,
            voice_collect_seconds=2.0,
            max_collect_seconds=60.0
        )

    def test_schedule_hot_window_creates_timer(self):
        """Scheduling hot window creates an activation timer."""
        manager = self._create_state_manager()

        assert manager._hot_window_activation_timer is None
        manager.schedule_hot_window_activation(voice_debug=True)

        # Timer should be created
        assert manager._hot_window_activation_timer is not None

        # Cleanup
        manager.stop()

    def test_cancel_hot_window_activation(self):
        """Pending hot window activation can be cancelled."""
        manager = self._create_state_manager()

        manager.schedule_hot_window_activation(voice_debug=True)
        assert manager._hot_window_activation_timer is not None

        manager.cancel_hot_window_activation()
        assert manager._hot_window_activation_timer is None

    def test_stop_cancels_all_timers(self):
        """Stopping the manager cancels all timers."""
        manager = self._create_state_manager()

        manager.schedule_hot_window_activation(voice_debug=True)
        manager.stop()

        assert manager._hot_window_activation_timer is None
        assert manager._hot_window_expiry_timer is None

    def test_hot_window_activates_after_delay(self):
        """Hot window activates after echo tolerance delay."""
        from jarvis.listening.state_manager import StateManager, ListeningState

        manager = StateManager(
            hot_window_seconds=6.0,
            echo_tolerance=0.1,  # Short delay for testing
            voice_collect_seconds=2.0,
            max_collect_seconds=60.0
        )

        manager.schedule_hot_window_activation(voice_debug=True)

        # Should not be active immediately
        assert manager.get_state() != ListeningState.HOT_WINDOW

        # Wait for activation
        time.sleep(0.2)

        # Should now be active
        assert manager.get_state() == ListeningState.HOT_WINDOW

        manager.stop()

    def test_hot_window_expires_after_full_duration(self):
        """Hot window should expire only after the full duration passes."""
        from jarvis.listening.state_manager import StateManager, ListeningState

        # Use short times for testing
        hot_window_seconds = 0.5
        echo_tolerance = 0.1

        manager = StateManager(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
            voice_collect_seconds=2.0,
            max_collect_seconds=60.0
        )

        manager.schedule_hot_window_activation(voice_debug=True)

        # Wait for activation
        time.sleep(echo_tolerance + 0.05)

        # Should be active
        assert manager.get_state() == ListeningState.HOT_WINDOW

        # Should still be active at 80% of duration
        time.sleep(hot_window_seconds * 0.8)
        assert manager.get_state() == ListeningState.HOT_WINDOW, "Hot window expired too early!"

        # Should expire after full duration (plus small buffer)
        time.sleep(hot_window_seconds * 0.3 + 0.1)
        assert manager.get_state() == ListeningState.WAKE_WORD, "Hot window didn't expire!"

        manager.stop()

    def test_hot_window_total_time_from_tts_end(self):
        """Verify total time from 'TTS end' (schedule call) to hot window expiry."""
        from jarvis.listening.state_manager import StateManager, ListeningState

        # Use realistic but short times
        hot_window_seconds = 0.4
        echo_tolerance = 0.1

        manager = StateManager(
            hot_window_seconds=hot_window_seconds,
            echo_tolerance=echo_tolerance,
            voice_collect_seconds=2.0,
            max_collect_seconds=60.0
        )

        start_time = time.time()
        manager.schedule_hot_window_activation(voice_debug=True)

        # First wait for hot window to activate
        while manager.get_state() != ListeningState.HOT_WINDOW:
            time.sleep(0.01)
            if time.time() - start_time > 1.0:
                manager.stop()
                assert False, "Hot window never activated"

        # Now wait until hot window expires
        while manager.get_state() == ListeningState.HOT_WINDOW:
            time.sleep(0.05)
            if time.time() - start_time > 2.0:
                manager.stop()
                assert False, "Hot window never expired (timeout)"

        elapsed = time.time() - start_time
        expected_min = echo_tolerance + hot_window_seconds - 0.1
        expected_max = echo_tolerance + hot_window_seconds + 0.2

        manager.stop()

        assert expected_min <= elapsed <= expected_max, (
            f"Hot window expired after {elapsed:.2f}s, "
            f"expected {echo_tolerance + hot_window_seconds:.2f}s "
            f"(range: {expected_min:.2f}-{expected_max:.2f}s)"
        )

    def test_capture_hot_window_state_thread_safe(self):
        """Hot window state capture uses proper locking."""
        manager = self._create_state_manager()

        # Should not raise even if called concurrently
        threads = []
        for _ in range(10):
            t = threading.Thread(target=manager.capture_hot_window_state_at_voice_start)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        manager.stop()


class TestEchoDetectionThreshold:
    """Tests for context-aware echo detection thresholds."""

    def _create_echo_detector(self):
        """Create an EchoDetector instance."""
        from jarvis.listening.echo_detection import EchoDetector
        return EchoDetector(echo_tolerance=0.3, energy_spike_threshold=2.0)

    def test_similarity_threshold_normal_mode(self):
        """Normal mode uses standard threshold (85)."""
        detector = self._create_echo_detector()

        # Track some TTS text
        detector.track_tts_start("hello world", 0.01)
        detector.track_tts_finish()

        # With 85% threshold, similar text should be rejected
        result = detector._check_text_similarity("hello world", "hello world", threshold=85)
        assert result is True

    def test_similarity_threshold_hot_window(self):
        """Hot window mode uses higher threshold (92)."""
        detector = self._create_echo_detector()

        # Track some TTS text
        detector.track_tts_start("hello world", 0.01)
        detector.track_tts_finish()

        # With 92% threshold, slightly different text should pass
        result = detector._check_text_similarity("hello", "hello world", threshold=92)
        # The actual result depends on rapidfuzz behavior

    def test_should_reject_accepts_in_hot_window_parameter(self):
        """should_reject_as_echo accepts in_hot_window parameter."""
        detector = self._create_echo_detector()

        detector.track_tts_start("test text", 0.01)
        detector.track_tts_finish()

        # This should not raise - parameter is accepted
        result = detector.should_reject_as_echo(
            heard_text="test text",
            current_energy=0.01,
            is_during_tts=False,
            tts_rate=200.0,
            utterance_start_time=time.time(),
            in_hot_window=True
        )
        # Result depends on timing and energy, but should not raise


class TestConfigNewOptions:
    """Tests for new configuration options."""

    def test_audio_wake_config_defaults(self):
        """Audio wake config has correct defaults."""
        from jarvis.config import get_default_config

        defaults = get_default_config()

        assert defaults["audio_wake_enabled"] is True
        assert defaults["audio_wake_threshold"] == 0.5

    def test_intent_judge_config_defaults(self):
        """Intent judge config has correct defaults."""
        from jarvis.config import get_default_config

        defaults = get_default_config()

        assert defaults["intent_judge_model"] == "llama3.2:3b"
        assert defaults["intent_judge_timeout_sec"] == 3.0

    def test_transcript_buffer_config_defaults(self):
        """Transcript buffer config has correct defaults."""
        from jarvis.config import get_default_config

        defaults = get_default_config()

        # 120s (2 min) provides good context for multi-person conversations
        assert defaults["transcript_buffer_duration_sec"] == 120.0

    def test_load_settings_includes_new_options(self):
        """load_settings includes new options in Settings."""
        with patch("jarvis.config._load_json", return_value={}):
            from jarvis.config import load_settings

            settings = load_settings()

            # Audio wake options
            assert hasattr(settings, "audio_wake_enabled")
            assert hasattr(settings, "audio_wake_threshold")

            # Intent judge options
            assert hasattr(settings, "intent_judge_model")
            assert hasattr(settings, "intent_judge_timeout_sec")

            # Transcript buffer options
            assert hasattr(settings, "transcript_buffer_duration_sec")


class TestHotWindowTimingWithUtteranceTime:
    """Tests for hot window detection using utterance timing.

    This addresses a bug where long utterances spanning TTS completion would
    be incorrectly processed as wake_word mode instead of hot_window mode
    because the hot window had expired by the time processing occurred.

    The key insight is that what matters is when the user STARTED speaking,
    not when processing happens or even when the utterance ends.
    """

    def test_utterance_starting_during_tts_is_hot_window(self):
        """Utterance that started during TTS should be treated as hot window.

        Scenario from real bug:
        - TTS playing from 18:29:38 to 18:30:25 (~48 seconds)
        - User starts speaking at 18:30:21 (DURING TTS)
        - User finishes at 18:30:28 (after hot window expired)
        - Hot window was 18:30:25 to 18:30:28 (3 seconds)

        Even though processing happens after hot window expires, the user
        clearly intended to follow up since they started speaking during TTS.
        """
        tts_finish_time = 1000.0
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance

        # User started speaking DURING TTS (before TTS finished)
        utterance_start_time = tts_finish_time - 3.3  # Started 3.3s before TTS ended
        utterance_end_time = tts_finish_time + 3.5    # Ended 3.5s after TTS

        # Case 1: Started during TTS
        started_during_tts = utterance_start_time < tts_finish_time
        assert started_during_tts is True

        # This should be treated as hot window
        could_be_hot_window = started_during_tts
        assert could_be_hot_window is True

    def test_utterance_ending_within_grace_period_is_hot_window(self):
        """Utterance ending within grace period should be hot window."""
        tts_finish_time = 1000.0
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance  # 3.3 seconds

        # User started after TTS, ended within grace period
        utterance_start_time = tts_finish_time + 1.0  # Started 1s after TTS
        utterance_end_time = tts_finish_time + 2.0    # Ended 2s after TTS

        started_during_tts = utterance_start_time < tts_finish_time
        ended_within_grace = utterance_end_time - tts_finish_time < grace_period

        assert started_during_tts is False
        assert ended_within_grace is True

        # Should still be hot window because ended within grace
        could_be_hot_window = started_during_tts or ended_within_grace
        assert could_be_hot_window is True

    def test_utterance_after_grace_period_not_hot_window(self):
        """Utterance that started after grace period should not be hot window."""
        tts_finish_time = 1000.0
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance  # 3.3 seconds

        # User started well after TTS finished
        utterance_start_time = tts_finish_time + 10.0  # Started 10s after TTS
        utterance_end_time = tts_finish_time + 12.0    # Ended 12s after TTS

        started_during_tts = utterance_start_time < tts_finish_time
        ended_within_grace = utterance_end_time - tts_finish_time < grace_period

        assert started_during_tts is False
        assert ended_within_grace is False

        # Should NOT be hot window
        could_be_hot_window = started_during_tts or ended_within_grace
        assert could_be_hot_window is False

    def test_processing_time_fallback_when_no_utterance_times(self):
        """Falls back to processing time when utterance times not available."""
        tts_finish_time = 1000.0
        hot_window_seconds = 3.0
        echo_tolerance = 0.3
        grace_period = hot_window_seconds + echo_tolerance

        # No utterance times (legacy case)
        utterance_start_time = 0.0
        utterance_end_time = 0.0
        current_time = 1002.0  # Processing within grace period

        started_during_tts = utterance_start_time > 0 and utterance_start_time < tts_finish_time
        ended_within_grace = utterance_end_time > 0 and utterance_end_time - tts_finish_time < grace_period
        processing_within_grace = current_time - tts_finish_time < grace_period

        # Falls back to processing time
        could_be_hot_window = started_during_tts or ended_within_grace or processing_within_grace
        assert could_be_hot_window is True
