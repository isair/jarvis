"""
Tests for echo detection module.

These tests verify that TTS echo detection properly identifies
when heard audio is an echo of TTS output vs genuine user speech.
"""

import time
import pytest
from jarvis.listening.echo_detection import EchoDetector


class TestTextNormalization:
    """Tests for text normalization handling TTS/Whisper differences."""

    def test_normalize_celsius_symbol(self):
        """Normalizes 9°C to '9 degrees celsius'."""
        detector = EchoDetector()
        result = detector._normalize_for_comparison("It's 9°C outside")
        assert "9 degrees celsius" in result
        assert "°" not in result

    def test_normalize_fahrenheit_symbol(self):
        """Normalizes 48°F to '48 degrees fahrenheit'."""
        detector = EchoDetector()
        result = detector._normalize_for_comparison("It's 48°F")
        assert "48 degrees fahrenheit" in result

    def test_normalize_generic_degree(self):
        """Normalizes standalone degree symbol."""
        detector = EchoDetector()
        result = detector._normalize_for_comparison("Turn it to 180°")
        assert "180 degrees" in result

    def test_normalize_with_space(self):
        """Handles space between number and degree symbol."""
        detector = EchoDetector()
        result = detector._normalize_for_comparison("It's 9 °C")
        assert "9 degrees celsius" in result

    def test_normalize_removes_parentheses(self):
        """Removes parentheses from text."""
        detector = EchoDetector()
        result = detector._normalize_for_comparison("It's 48°F (9°C)")
        # Should contain both values without parentheses
        assert "(" not in result
        assert ")" not in result
        assert "48 degrees fahrenheit" in result
        assert "9 degrees celsius" in result


class TestTextSimilarity:
    """Tests for text similarity matching."""

    def test_exact_match(self):
        """Detects exact text match."""
        detector = EchoDetector()
        assert detector._check_text_similarity("hello world", "hello world") is True

    def test_case_insensitive_match(self):
        """Detects match regardless of case."""
        detector = EchoDetector()
        assert detector._check_text_similarity("Hello World", "hello world") is True

    def test_partial_match(self):
        """Detects when heard text is substring of TTS."""
        detector = EchoDetector()
        tts = "the weather today is sunny and warm"
        heard = "sunny and warm"
        assert detector._check_text_similarity(heard, tts) is True

    def test_no_match(self):
        """Returns False for unrelated text."""
        detector = EchoDetector()
        assert detector._check_text_similarity("what time is it", "the weather is nice") is False

    def test_degree_symbol_match(self):
        """Matches degree symbol text against Whisper transcription."""
        detector = EchoDetector()
        tts = "It's currently 9°C outside"
        heard = "It's currently 9 degrees celsius outside"
        assert detector._check_text_similarity(heard, tts) is True

    def test_empty_strings(self):
        """Returns False for empty strings."""
        detector = EchoDetector()
        assert detector._check_text_similarity("", "hello") is False
        assert detector._check_text_similarity("hello", "") is False
        assert detector._check_text_similarity("", "") is False

    def test_higher_threshold_in_hot_window(self):
        """Uses higher threshold (92) for hot window to reduce false rejections."""
        detector = EchoDetector()
        # Test that threshold parameter affects matching
        # Use text with typos/variations that won't be exact match
        # "the weether forcast" vs "the weather forecast" scores ~89-92
        tts = "the weather forecast"
        heard = "the weether forcast"  # typos - similar but not exact
        # At low threshold this should match, at threshold above score it should not
        low_threshold = detector._check_text_similarity(heard, tts, threshold=80)
        high_threshold = detector._check_text_similarity(heard, tts, threshold=95)
        # Lower threshold (80) should match text scoring ~92
        assert low_threshold is True
        # Higher threshold (95) should reject text scoring ~92
        assert high_threshold is False


class TestEchoRejection:
    """Tests for the main echo rejection decision logic."""

    def test_no_rejection_without_tts(self):
        """Doesn't reject if no TTS was ever played."""
        detector = EchoDetector()
        assert detector.should_reject_as_echo("hello", current_energy=0.01) is False

    def test_rejects_echo_during_tts(self):
        """Rejects matching text during TTS playback."""
        detector = EchoDetector()
        tts_text = "the weather is nice today"
        detector.track_tts_start(tts_text)

        # Simulate utterance starting right after TTS starts
        utterance_start = time.time()

        result = detector.should_reject_as_echo(
            heard_text="nice today",
            current_energy=0.01,
            is_during_tts=True,
            tts_rate=200.0,
            utterance_start_time=utterance_start
        )
        assert result is True

    def test_accepts_different_text_during_tts(self):
        """Accepts non-matching text during TTS (interruption)."""
        detector = EchoDetector()
        detector.track_tts_start("the weather is nice")

        result = detector.should_reject_as_echo(
            heard_text="stop",
            current_energy=0.05,
            is_during_tts=True,
            tts_rate=200.0,
            utterance_start_time=time.time()
        )
        assert result is False

    def test_rejects_echo_in_cooldown_window(self):
        """Rejects matching text shortly after TTS finishes."""
        detector = EchoDetector()
        tts_text = "hello world"
        detector.track_tts_start(tts_text, baseline_energy=0.01)
        detector.track_tts_finish()

        # Simulate utterance starting immediately after TTS
        utterance_start = time.time()

        result = detector.should_reject_as_echo(
            heard_text="hello world",
            current_energy=0.008,  # Low energy (below baseline * threshold)
            is_during_tts=False,
            utterance_start_time=utterance_start
        )
        assert result is True

    def test_accepts_high_energy_in_cooldown(self):
        """Accepts speech with high energy even in cooldown (real user)."""
        detector = EchoDetector(energy_spike_threshold=2.0)
        detector.track_tts_start("hello world", baseline_energy=0.01)
        detector.track_tts_finish()

        utterance_start = time.time()

        result = detector.should_reject_as_echo(
            heard_text="hello world",
            current_energy=0.05,  # High energy (5x baseline)
            is_during_tts=False,
            utterance_start_time=utterance_start
        )
        assert result is False

    def test_accepts_after_extended_window(self):
        """Accepts speech after extended echo window expires."""
        detector = EchoDetector(echo_tolerance=0.3)
        detector.track_tts_start("hello world")
        detector.track_tts_finish()

        # Simulate utterance starting well after TTS (2 seconds)
        utterance_start = time.time() + 2.0
        detector._last_tts_finish_time = time.time() - 2.0  # TTS finished 2s ago

        result = detector.should_reject_as_echo(
            heard_text="hello world",
            current_energy=0.01,
            is_during_tts=False,
            utterance_start_time=utterance_start
        )
        assert result is False


class TestLeadingEchoCleanup:
    """Tests for cleanup_leading_echo functionality."""

    def test_cleanup_leading_overlap(self):
        """Removes leading words that match end of TTS."""
        detector = EchoDetector()
        detector._last_tts_text = "the weather today is sunny"

        heard = "is sunny what time is it"
        result = detector.cleanup_leading_echo(heard)
        assert result == "what time is it"

    def test_no_cleanup_when_no_overlap(self):
        """Doesn't modify text when there's no overlap."""
        detector = EchoDetector()
        detector._last_tts_text = "the weather is nice"

        heard = "what time is it"
        result = detector.cleanup_leading_echo(heard)
        assert result == heard

    def test_no_cleanup_short_overlap(self):
        """Doesn't cleanup if overlap is only 1 word."""
        detector = EchoDetector()
        detector._last_tts_text = "the weather is nice"

        heard = "nice what time is it"  # Only 1 word overlap
        result = detector.cleanup_leading_echo(heard)
        assert result == heard  # No cleanup for 1-word overlap

    def test_cleanup_requires_remainder(self):
        """Doesn't cleanup if the entire heard text is the echo."""
        detector = EchoDetector()
        detector._last_tts_text = "the weather is nice"

        heard = "is nice"  # Entire text is echo, no remainder
        result = detector.cleanup_leading_echo(heard)
        assert result == heard  # Don't cleanup if nothing remains


class TestHotWindowEchoDetection:
    """Tests for echo detection in hot window mode."""

    def test_higher_threshold_in_hot_window(self):
        """Uses stricter matching in hot window to allow more follow-up speech."""
        detector = EchoDetector()
        detector.track_tts_start("tell me about the weather today")
        detector.track_tts_finish()

        utterance_start = time.time()

        # Text that's somewhat similar but not the same
        result = detector.should_reject_as_echo(
            heard_text="tell me more",
            current_energy=0.01,
            is_during_tts=False,
            utterance_start_time=utterance_start,
            in_hot_window=True  # Hot window mode
        )
        # Should be less likely to reject in hot window due to higher threshold
        # (The actual behavior depends on similarity scores)
        assert result is False  # "tell me more" is different enough
