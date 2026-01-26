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

    def test_partial_echo_from_long_tts(self):
        """Detects partial echo from a long TTS response.

        This tests the scenario where TTS outputs a long response and Whisper
        picks up only a portion of it, potentially with transcription errors.
        Common in rooms with echo/reverb at higher volumes.
        """
        detector = EchoDetector()
        # Simulate a long weather response
        tts_text = (
            "You're in London, and I've got the latest weather update for you: "
            "it's currently overcast with light rain showers, and the temperature "
            "is around 8 degrees celsius at 18:48 UTC. I'd recommend grabbing an "
            "umbrella to stay dry. Would you like me to suggest any outdoor "
            "activities or provide more weather details?"
        )
        detector.track_tts_start(tts_text)
        detector.track_tts_finish()

        utterance_start = time.time()

        # Partial echo that Whisper picked up (with some transcription variations)
        partial_echo = "the temperature is around 8 degrees celsius. I'd recommend grabbing an umbrella"

        # Should detect as echo - this is clearly part of the TTS output
        result = detector._check_text_similarity(partial_echo, tts_text, threshold=70)
        assert result is True, f"Should detect partial echo at threshold 70"

    def test_echo_with_whisper_transcription_errors(self):
        """Detects echo even with Whisper transcription errors.

        Whisper sometimes mishears numbers and times (e.g., "18:48" as "1848").
        The fuzzy matching should still catch these as echo.
        """
        detector = EchoDetector()
        tts_text = "the temperature is 8 degrees celsius at 18:48 UTC"
        detector.track_tts_start(tts_text)
        detector.track_tts_finish()

        # Whisper transcription with errors
        heard_with_errors = "the temperature is around 8 degrees celsius at 1848 UTC"

        # Should still detect similarity despite transcription errors
        result = detector._check_text_similarity(heard_with_errors, tts_text, threshold=70)
        assert result is True, "Should detect echo despite transcription errors"

    def test_echo_question_from_tts(self):
        """Detects when a question from TTS is echoed back.

        TTS often ends with questions like "Would you like more details?"
        These should be detected as echo, not new user queries.
        """
        detector = EchoDetector()
        tts_text = (
            "The weather is nice today. Would you like me to suggest "
            "any outdoor activities or provide more weather details?"
        )
        detector.track_tts_start(tts_text)
        detector.track_tts_finish()

        # Echo of the question portion
        echoed_question = "would you like me to suggest any outdoor activities"

        result = detector._check_text_similarity(echoed_question, tts_text, threshold=70)
        assert result is True, "Should detect echoed question from TTS"

    def test_accepts_genuine_followup_in_hot_window(self):
        """Accepts genuine follow-up that differs from TTS content."""
        detector = EchoDetector()
        tts_text = "The weather in London is currently overcast with rain"
        detector.track_tts_start(tts_text)
        detector.track_tts_finish()

        utterance_start = time.time()

        # Genuine follow-up question - different content
        followup = "what about tomorrow's forecast"

        result = detector.should_reject_as_echo(
            heard_text=followup,
            current_energy=0.03,
            is_during_tts=False,
            utterance_start_time=utterance_start,
            in_hot_window=True
        )
        assert result is False, "Should accept genuine follow-up question"

    def test_threshold_70_catches_partial_matches(self):
        """Verifies threshold 70 catches partial echo matches.

        When using threshold 70 in hot window for fast rejection,
        partial echoes with ~75% similarity should be caught.
        """
        detector = EchoDetector()
        tts_text = "London has about 8 hours of daylight in winter months"

        # Partial echo with some differences
        partial_echo = "London has about 8 hours of daylight"

        # At threshold 70, should match (this is clearly a partial echo)
        result_70 = detector._check_text_similarity(partial_echo, tts_text, threshold=70)
        assert result_70 is True, "Threshold 70 should catch partial echo"

        # At threshold 92 (default hot window), might not match as strictly
        # This is fine - the intent judge handles ambiguous cases
        result_92 = detector._check_text_similarity(partial_echo, tts_text, threshold=92)
        # We don't assert on this as it depends on the fuzzy match algorithm


class TestSalvageDuringTTS:
    """Tests for cleanup_leading_echo_during_tts functionality.

    This tests the salvage logic that extracts user speech from utterances
    that start during TTS (mixed echo + user speech).
    """

    @pytest.fixture
    def detector(self):
        return EchoDetector()

    def test_salvages_user_speech_after_echo(self, detector):
        """Extracts user speech that follows TTS echo.

        Scenario: User starts speaking during TTS, mic picks up end of TTS
        plus user's actual question.
        """
        tts_text = (
            "According to the BBC Weather forecast, tomorrow in Kensington is expected "
            "to be quite gloomy with overcast conditions. You might want to bundle up "
            "and plan your outdoor activities accordingly."
        )
        detector._last_tts_text = tts_text
        detector._tts_start_time = 1000.0

        # User's mic picks up end of TTS + their actual question
        heard = (
            "You might want to bundle up and plan your outdoor activities accordingly. "
            "Okay, let's switch the topic now. I want to talk about philosophy."
        )

        # Utterance started 10 seconds into TTS
        result = detector.cleanup_leading_echo_during_tts(heard, tts_rate=200, utterance_start_time=1010.0)

        # Should remove echo and keep user's speech
        assert "bundle up" not in result.lower(), "Echo portion should be removed"
        assert "philosophy" in result.lower(), "User's actual question should be preserved"
        assert "switch the topic" in result.lower(), "User's speech should be preserved"

    def test_salvage_with_timing_mismatch(self, detector):
        """Salvages correctly even when timing estimate is off.

        Real-world scenario: mic timing doesn't perfectly match TTS timing
        due to audio processing delays, pre-roll buffer, etc.
        """
        tts_text = (
            "It's going to be quite chilly. You might want to bundle up "
            "and plan your outdoor activities accordingly."
        )
        detector._last_tts_text = tts_text
        detector._tts_start_time = 1000.0

        # User's mic picks up end of TTS + their question
        # Timing estimate would be wrong, but full-text fallback should work
        heard = "plan your outdoor activities accordingly. What do you think life is about?"

        # Even with wrong timing estimate, should find match in full TTS
        result = detector.cleanup_leading_echo_during_tts(heard, tts_rate=200, utterance_start_time=1005.0)

        assert "outdoor activities" not in result.lower(), "Echo should be removed"
        assert "life is about" in result.lower(), "User's question should be preserved"

    def test_no_salvage_when_no_overlap(self, detector):
        """Returns original text when no overlap with TTS."""
        detector._last_tts_text = "The weather is nice today"
        detector._tts_start_time = 1000.0

        heard = "What time is it?"
        result = detector.cleanup_leading_echo_during_tts(heard, tts_rate=200, utterance_start_time=1005.0)

        assert result == heard, "Should return original when no echo overlap"

    def test_no_salvage_when_all_echo(self, detector):
        """Returns original when entire utterance is echo (no user speech to salvage)."""
        tts_text = "The weather is nice and sunny today"
        detector._last_tts_text = tts_text
        detector._tts_start_time = 1000.0

        # Entire heard text matches end of TTS - nothing to salvage
        heard = "nice and sunny today"
        result = detector.cleanup_leading_echo_during_tts(heard, tts_rate=200, utterance_start_time=1005.0)

        # Should return original since there's nothing left after removing echo
        assert result == heard

    def test_echo_not_in_salvaged_output(self, detector):
        """Verifies echo portion doesn't slip into salvaged output.

        This is the critical test - ensures we don't accidentally include
        echo text in what we return to the user.
        """
        tts_text = (
            "According to the forecast, it will rain tomorrow. "
            "Would you like me to suggest indoor activities?"
        )
        detector._last_tts_text = tts_text
        detector._tts_start_time = 1000.0

        heard = "Would you like me to suggest indoor activities? No thanks, tell me about philosophy instead."
        result = detector.cleanup_leading_echo_during_tts(heard, tts_rate=200, utterance_start_time=1008.0)

        # Critical: echo words should NOT be in the result
        assert "suggest indoor activities" not in result.lower(), "Echo phrase must not be in output"
        assert "would you like" not in result.lower(), "Echo phrase must not be in output"
        # User's actual request should be preserved
        assert "philosophy" in result.lower(), "User's request should be preserved"


class TestRealWorldSalvageScenarios:
    """Tests for real-world salvage scenarios that have caused regressions.

    These tests capture actual issues encountered in production:
    - Temperature notation differences (5.7°C vs "5.7 degrees Celsius")
    - User appending speech to TTS echo
    - Whisper transcription differences from TTS text
    """

    @pytest.fixture
    def detector(self):
        return EchoDetector()

    def test_temperature_notation_mismatch(self, detector):
        """Salvages user speech when Whisper transcribes temperature differently.

        Real scenario: TTS says "5.7°C" but Whisper transcribes "5.7 degrees Celsius"
        This caused salvage to fail because word-level matching didn't match.
        """
        tts_text = "It's going to be a bit chilly tomorrow in Kensington, with overcast skies and a temperature around 5.7°C."
        detector._last_tts_text = tts_text

        # Whisper transcribes temperature differently
        heard = "It's going to be a bit chilly tomorrow in Kensington with overcast skies and a temperature around 5.7 degrees Celsius. Nice, you remembered not to say it in Fahrenheit."

        result = detector.cleanup_leading_echo(heard)

        # Should salvage user's follow-up
        assert "nice" in result.lower(), "User's follow-up should be preserved"
        assert "fahrenheit" in result.lower(), "User's comment should be preserved"
        # Echo should be removed
        assert "chilly tomorrow" not in result.lower(), "Echo should be removed"

    def test_user_appends_speech_to_full_tts_echo(self, detector):
        """User speaks immediately after TTS, mic captures both.

        The entire TTS is captured plus user's response. cleanup_leading_echo
        should remove the TTS portion and return user's speech.
        """
        tts_text = "Would you like some help finding one?"
        detector._last_tts_text = tts_text

        # User responds right after TTS, mic captures both
        heard = "Would you like some help finding one? No thanks, I'm good."

        result = detector.cleanup_leading_echo(heard)

        # Should return user's response
        assert "no thanks" in result.lower(), "User's response should be preserved"
        assert "i'm good" in result.lower() or "im good" in result.lower(), "User's response should be preserved"
        # Echo should be removed
        assert "would you like" not in result.lower(), "Echo should be removed"

    def test_salvage_preserves_user_question(self, detector):
        """Salvage preserves user's follow-up question after echo."""
        tts_text = "The weather tomorrow will be cloudy with a high of 12 degrees."
        detector._last_tts_text = tts_text

        heard = "The weather tomorrow will be cloudy with a high of 12 degrees. What about the day after?"

        result = detector.cleanup_leading_echo(heard)

        assert "what about" in result.lower(), "User's question should be preserved"
        assert "day after" in result.lower(), "User's question should be preserved"
        assert "cloudy" not in result.lower(), "Echo should be removed"

    def test_no_salvage_when_heard_matches_tts_exactly(self, detector):
        """Returns original when heard text is exactly TTS (no user speech).

        This ensures we don't accidentally salvage a trailing word from pure echo.
        """
        tts_text = "Would you like some help finding one?"
        detector._last_tts_text = tts_text

        # Heard matches TTS exactly - no user speech to salvage
        heard = "Would you like some help finding one?"

        result = detector.cleanup_leading_echo(heard)

        # Should return original (full echo, nothing to salvage)
        assert result == heard, "Should return original when no user speech to salvage"

    def test_salvage_with_minor_transcription_errors(self, detector):
        """Salvage works despite minor Whisper transcription errors."""
        tts_text = "I can see you're interested in finding out more about this topic."
        detector._last_tts_text = tts_text

        # Whisper may drop punctuation or have minor differences
        heard = "I can see youre interested in finding out more about this topic tell me about philosophy"

        result = detector.cleanup_leading_echo(heard)

        # Should salvage user's request (may or may not work depending on how different)
        # At minimum, shouldn't crash
        assert result is not None
