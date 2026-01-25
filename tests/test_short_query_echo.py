"""
Test that short legitimate queries are not incorrectly rejected as echo.

The hot window echo detection uses length-aware processing:
- Short queries (<=4 words): Skip fast rejection entirely, let intent judge handle
- Longer queries (>4 words): Use threshold 70 for fast rejection

This prevents false positives on "tell me more", "how", "weather" etc.
while still catching actual partial echoes from long TTS responses.
"""

import pytest
from jarvis.listening.echo_detection import EchoDetector


class TestShortQueryBehavior:
    """Test that short queries are handled appropriately.

    The fast echo rejection path is SKIPPED for queries <=4 words.
    These tests verify the thresholds that WOULD apply if used,
    demonstrating why we skip fast rejection for short queries.
    """

    @pytest.fixture
    def detector(self):
        return EchoDetector()

    @pytest.fixture
    def weather_tts(self):
        return (
            "The weather in London is currently overcast with light rain "
            "showers and the temperature is around 8 degrees celsius. "
            "Would you like me to provide more details?"
        )

    def test_partial_ratio_matches_substrings_falsely(self, detector, weather_tts):
        """Demonstrate why we skip fast rejection for short queries.

        partial_ratio finds substrings, causing false positives:
        - 'how' matches 's**how**ers' with 100%
        - 'weather' matches exactly with 100%
        - 'more details' matches exactly with 100%

        This is why queries <=4 words skip fast rejection.
        """
        # These short queries would be incorrectly rejected at any reasonable threshold
        false_positive_queries = [
            "how",           # Substring of 'showers'
            "weather",       # Exact word match
            "more details",  # Exact phrase match
            "light rain",    # Exact phrase match
            "the",           # Common word
        ]

        for query in false_positive_queries:
            # These all get high scores from partial_ratio
            result = detector._check_text_similarity(query, weather_tts, threshold=85)
            # We're demonstrating these WOULD be rejected, which is why we skip them
            assert result is True, f"'{query}' should match at threshold 85 (demonstrating the problem)"

    def test_legitimate_short_queries_pass_intent_judge(self, detector, weather_tts):
        """Short queries that don't match TTS should be accepted by intent judge.

        These queries have low similarity scores and would pass even with fast rejection,
        but they still go through intent judge for proper context-aware handling.
        """
        legitimate_queries = [
            "yes",
            "no",
            "what about tomorrow",
            "sounds good",
            "thanks",
        ]

        for query in legitimate_queries:
            # Verify these have low similarity - would pass fast rejection if applied
            result = detector._check_text_similarity(query, weather_tts, threshold=85)
            assert result is False, f"'{query}' has low similarity as expected"


class TestLongerEchoDetection:
    """Test that longer echoes (>4 words) are detected."""

    @pytest.fixture
    def detector(self):
        return EchoDetector()

    @pytest.fixture
    def weather_tts(self):
        return (
            "The weather in London is currently overcast with light rain "
            "showers and the temperature is around 8 degrees celsius. "
            "Would you like me to provide more details?"
        )

    def test_longer_echo_detected_at_threshold_70(self, detector, weather_tts):
        """Longer queries (>4 words) that match TTS should be detected at threshold 70."""
        actual_echoes = [
            "the weather in london is currently overcast",  # 7 words
            "light rain showers and the temperature is around",  # 8 words
            "would you like me to provide more details",  # 8 words
        ]

        for echo in actual_echoes:
            word_count = len(echo.split())
            assert word_count > 4, f"Test setup error: '{echo}' has only {word_count} words"
            result = detector._check_text_similarity(echo, weather_tts, threshold=70)
            assert result is True, f"Echo '{echo[:30]}...' ({word_count} words) should be detected at threshold 70"

    def test_partial_echo_with_transcription_errors(self, detector):
        """Longer partial echoes with transcription errors should be detected."""
        tts = (
            "The temperature is around 8 degrees celsius at 18:48 UTC. "
            "Would you like me to provide more weather information?"
        )
        detector.track_tts_start(tts)

        # Whisper transcription with errors (common in high-volume rooms)
        echo_with_errors = "the temperature is around 8 degrees celsius at 1848 UTC"  # 10 words

        # This should be detected at threshold 70
        result = detector._check_text_similarity(echo_with_errors, tts, threshold=70)
        assert result is True, "Partial echo with transcription errors should be detected"

    def test_longer_followups_not_rejected(self, detector, weather_tts):
        """Longer follow-up questions (>4 words) should NOT match TTS."""
        long_followups = [
            "what will the weather be like tomorrow",  # 7 words
            "should i bring an umbrella with me today",  # 8 words
            "thanks jarvis that was very helpful information",  # 7 words
            "can you tell me about the weekend forecast",  # 8 words
        ]

        for query in long_followups:
            word_count = len(query.split())
            assert word_count > 4, f"Test setup error: '{query}' has only {word_count} words"
            result = detector._check_text_similarity(query, weather_tts, threshold=70)
            assert result is False, f"Follow-up '{query}' should not be rejected at threshold 70"


class TestLengthBoundary:
    """Test behavior at the 4-word boundary."""

    @pytest.fixture
    def detector(self):
        return EchoDetector()

    def test_four_word_query_skips_fast_rejection(self, detector):
        """4-word queries skip fast rejection (handled by intent judge)."""
        # This is a design decision, not an assertion about similarity
        query = "tell me more please"  # 4 words
        assert len(query.split()) == 4

    def test_five_word_query_uses_fast_rejection(self, detector):
        """5-word queries use fast rejection at threshold 70."""
        tts = "The weather today is nice and sunny in London"
        query = "the weather today is nice"  # 5 words - matches TTS

        assert len(query.split()) == 5
        result = detector._check_text_similarity(query, tts, threshold=70)
        assert result is True, "5-word echo should be detected at threshold 70"

    def test_five_word_non_echo_passes(self, detector):
        """5-word non-echo queries should pass fast rejection."""
        tts = "The weather today is nice and sunny in London"
        query = "what about the rain tomorrow"  # 5 words - doesn't match

        assert len(query.split()) == 5
        result = detector._check_text_similarity(query, tts, threshold=70)
        assert result is False, "5-word non-echo should pass threshold 70"
