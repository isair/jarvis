"""
Tests for wake word and stop command detection.
"""

import pytest

from jarvis.listening.wake_detection import (
    is_wake_word_detected,
    is_stop_command,
)


@pytest.mark.unit
class TestWakeWordDetection:
    """Tests for is_wake_word_detected."""

    def test_exact_match(self):
        assert is_wake_word_detected("hey jarvis", "jarvis", []) is True

    def test_alias_match(self):
        assert is_wake_word_detected("hey computer", "jarvis", ["computer"]) is True

    def test_no_match(self):
        assert is_wake_word_detected("hello world", "jarvis", []) is False

    def test_empty_text(self):
        assert is_wake_word_detected("", "jarvis", []) is False

    def test_fuzzy_match(self):
        """Fuzzy matching catches slight transcription errors."""
        assert is_wake_word_detected("hey jarvas", "jarvis", [], fuzzy_ratio=0.78) is True

    def test_fuzzy_below_threshold(self):
        """Completely different word doesn't fuzzy-match."""
        assert is_wake_word_detected("hey banana", "jarvis", [], fuzzy_ratio=0.78) is False


@pytest.mark.unit
class TestStopCommand:
    """Tests for is_stop_command."""

    def test_exact_stop_command(self):
        assert is_stop_command("stop", ["stop", "quiet"]) is True

    def test_stop_command_in_phrase(self):
        assert is_stop_command("please stop talking", ["stop", "quiet"]) is True

    def test_no_stop_command(self):
        assert is_stop_command("what is the weather", ["stop", "quiet"]) is False

    def test_empty_text(self):
        assert is_stop_command("", ["stop", "quiet"]) is False

    def test_fuzzy_stop_command(self):
        """Short input fuzzy-matches stop commands."""
        assert is_stop_command("stob", ["stop", "quiet"], fuzzy_ratio=0.7) is True
