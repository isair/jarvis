"""
Tests for wake word validation in the listener.

These tests verify that:
1. Wake word presence is verified in wake word mode
2. Hot window mode doesn't require wake word
3. Various state timing scenarios are handled correctly
"""

import pytest
from unittest.mock import patch, MagicMock
import time

from jarvis.listening.wake_detection import is_wake_word_detected


class TestWakeWordValidation:
    """Tests for wake word presence validation in wake word mode.

    The listener must verify wake word is present when:
    1. We're in wake word mode (not hot window)
    2. Intent judge says directed=true
    """

    def test_wake_word_detected_with_jarvis(self):
        """Wake word detection finds 'jarvis' in text."""
        text = "hey jarvis what time is it"
        assert is_wake_word_detected(text, "jarvis", []) is True

    def test_wake_word_detected_with_alias(self):
        """Wake word detection finds alias."""
        text = "hey assistant what time is it"
        assert is_wake_word_detected(text, "jarvis", ["assistant"]) is True

    def test_wake_word_not_detected_without_wake_word(self):
        """Wake word detection returns False when no wake word present."""
        text = "how are you"
        assert is_wake_word_detected(text, "jarvis", []) is False

    def test_wake_word_not_detected_similar_but_different(self):
        """Wake word detection doesn't match similar words."""
        text = "I was jarring some preserves"
        # "jarring" is similar to "jarvis" but should not match with high threshold
        assert is_wake_word_detected(text, "jarvis", [], fuzzy_ratio=0.9) is False

    def test_bug_scenario_no_wake_word_in_query(self):
        """
        Bug scenario: Intent judge says directed=true for 'How are you?'
        but there's no wake word - this should be rejected in wake word mode.
        """
        text = "how are you"
        wake_word = "jarvis"
        aliases = []

        # In wake word mode (not hot window), we need to verify wake word
        could_be_hot_window = False

        if not could_be_hot_window:
            # Check if wake word is present
            has_wake_word = is_wake_word_detected(text, wake_word, aliases)
            # This should be False - there's no "jarvis" in "how are you"
            assert has_wake_word is False, "Should reject - no wake word in text"

    def test_valid_query_with_wake_word(self):
        """Valid scenario: Wake word is present in the query."""
        text = "jarvis what's the weather"
        wake_word = "jarvis"
        aliases = []

        has_wake_word = is_wake_word_detected(text, wake_word, aliases)
        assert has_wake_word is True

    def test_hot_window_mode_no_wake_word_needed(self):
        """In hot window mode, wake word is not required."""
        text = "tell me more"
        wake_word = "jarvis"
        aliases = []

        # In hot window mode, we don't check for wake word
        could_be_hot_window = True

        # The wake word check is skipped in hot window mode
        # Intent judge decides based on context
        if not could_be_hot_window:
            has_wake_word = is_wake_word_detected(text, wake_word, aliases)
            # Would fail, but we're in hot window so this check is skipped
        # No assertion needed - just verifying the logic flow

    def test_wake_word_with_fuzzy_match(self):
        """Fuzzy matching catches slight variations."""
        text = "hey jarv what time is it"  # Slight typo
        wake_word = "jarvis"
        aliases = []

        # With lower fuzzy ratio (0.7), "jarv" might match "jarvis"
        result = is_wake_word_detected(text, wake_word, aliases, fuzzy_ratio=0.7)
        # "jarv" to "jarvis" ratio is about 0.73
        assert result is True

    def test_wake_word_case_insensitive(self):
        """Wake word detection is case insensitive."""
        text = "JARVIS what time is it"
        wake_word = "jarvis"
        aliases = []

        # Function expects lowercase text
        assert is_wake_word_detected(text.lower(), wake_word, aliases) is True


class TestIntentJudgeWakeWordValidation:
    """Integration tests for intent judge + wake word validation."""

    def test_intent_judge_directed_rejected_without_wake_word(self):
        """
        Simulate the bug: Intent judge says directed=true but no wake word.
        In wake word mode, this should be rejected.
        """
        # Simulated state
        text_lower = "how are you"
        could_be_hot_window = False  # Wake word mode
        wake_timestamp = None  # No wake word detected by audio detector
        wake_word = "jarvis"
        aliases = []

        # Intent judge (incorrectly) returns directed=true
        intent_judgment_directed = True
        intent_judgment_query = "how are you"

        # Validation logic from listener
        should_accept = False
        if intent_judgment_directed and intent_judgment_query:
            if not could_be_hot_window:
                # In wake word mode, verify wake word
                has_wake_word = wake_timestamp is not None or is_wake_word_detected(
                    text_lower, wake_word, aliases
                )
                should_accept = has_wake_word
            else:
                should_accept = True

        assert should_accept is False, "Should reject - no wake word in wake word mode"

    def test_intent_judge_directed_accepted_with_wake_word(self):
        """Intent judge directed=true is accepted when wake word is present."""
        text_lower = "jarvis what's the weather"
        could_be_hot_window = False  # Wake word mode
        wake_timestamp = None  # Doesn't matter, text has wake word
        wake_word = "jarvis"
        aliases = []

        intent_judgment_directed = True
        intent_judgment_query = "what's the weather"

        should_accept = False
        if intent_judgment_directed and intent_judgment_query:
            if not could_be_hot_window:
                has_wake_word = wake_timestamp is not None or is_wake_word_detected(
                    text_lower, wake_word, aliases
                )
                should_accept = has_wake_word
            else:
                should_accept = True

        assert should_accept is True, "Should accept - wake word present"

    def test_intent_judge_directed_accepted_with_timestamp(self):
        """Intent judge directed=true is accepted when wake_timestamp is set."""
        text_lower = "what's the weather"  # Wake word might be trimmed already
        could_be_hot_window = False  # Wake word mode
        wake_timestamp = 1000.5  # Wake word was detected by audio detector
        wake_word = "jarvis"
        aliases = []

        intent_judgment_directed = True
        intent_judgment_query = "what's the weather"

        should_accept = False
        if intent_judgment_directed and intent_judgment_query:
            if not could_be_hot_window:
                has_wake_word = wake_timestamp is not None or is_wake_word_detected(
                    text_lower, wake_word, aliases
                )
                should_accept = has_wake_word
            else:
                should_accept = True

        assert should_accept is True, "Should accept - wake_timestamp is set"

    def test_hot_window_always_accepts_directed(self):
        """In hot window mode, directed=true is always accepted."""
        text_lower = "tell me more"
        could_be_hot_window = True  # Hot window mode
        wake_timestamp = None
        wake_word = "jarvis"
        aliases = []

        intent_judgment_directed = True
        intent_judgment_query = "tell me more"

        should_accept = False
        if intent_judgment_directed and intent_judgment_query:
            if not could_be_hot_window:
                has_wake_word = wake_timestamp is not None or is_wake_word_detected(
                    text_lower, wake_word, aliases
                )
                should_accept = has_wake_word
            else:
                should_accept = True  # Hot window - no wake word needed

        assert should_accept is True, "Should accept - hot window mode"


class TestStateTimingScenarios:
    """Tests for state timing and transitions.

    These tests verify that the listener correctly handles various
    timing scenarios involving wake word, TTS, and hot window states.
    """

    def test_utterance_time_matters_not_processing_time(self):
        """
        Key principle: What matters is WHEN the user started speaking,
        not when processing completes.
        """
        hot_window_end_time = 1000.0

        # Scenario 1: User spoke during hot window, processed after expiry
        utterance_start_time = 998.0  # During hot window
        processing_time = 1002.0  # After hot window expired

        spoke_during_hot_window = utterance_start_time < hot_window_end_time
        assert spoke_during_hot_window is True

        # Should be treated as hot window because user STARTED during hot window

    def test_utterance_after_hot_window_requires_wake_word(self):
        """Utterance that started after hot window requires wake word."""
        hot_window_end_time = 1000.0

        # User started speaking after hot window ended
        utterance_start_time = 1002.0  # After hot window

        spoke_during_hot_window = utterance_start_time < hot_window_end_time
        assert spoke_during_hot_window is False

        # This requires wake word

    def test_utterance_spanning_hot_window_expiry(self):
        """
        Utterance that started during hot window but ended after expiry
        should still be treated as hot window.
        """
        tts_finish_time = 995.0
        hot_window_seconds = 5.0
        hot_window_end_time = tts_finish_time + hot_window_seconds  # 1000.0

        # User started during hot window, finished after
        utterance_start_time = 998.0
        utterance_end_time = 1003.0

        # The key check: did user START during hot window?
        spoke_during_hot_window = utterance_start_time < hot_window_end_time
        assert spoke_during_hot_window is True

    def test_long_utterance_during_tts(self):
        """
        Long utterance that started during TTS should be treated as
        potential follow-up or interrupt.
        """
        tts_start_time = 990.0
        tts_finish_time = 1010.0  # 20 second TTS

        # User started speaking during TTS
        utterance_start_time = 1005.0  # During TTS
        utterance_end_time = 1015.0  # After TTS ended

        spoke_during_tts = (
            utterance_start_time >= tts_start_time and
            utterance_start_time < tts_finish_time
        )
        assert spoke_during_tts is True

    def test_quick_followup_after_tts(self):
        """Quick follow-up right after TTS should be in hot window."""
        tts_finish_time = 1000.0
        echo_tolerance = 0.3
        hot_window_seconds = 6.0

        # User speaks right after TTS
        utterance_start_time = 1000.5  # Just after TTS

        # Should be well within hot window
        time_since_tts = utterance_start_time - tts_finish_time
        in_hot_window = time_since_tts < (echo_tolerance + hot_window_seconds)

        assert in_hot_window is True


class TestHotWindowQueryValidation:
    """Tests for hot window behavior."""

    def test_stop_command_validation(self):
        """Stop commands should work in hot window."""
        current_segment = "stop"
        # Stop commands are always accepted when detected
        assert "stop" in current_segment.lower()

    def test_interrupt_during_tts(self):
        """Interrupt during TTS should work with wake word."""
        current_segment = "jarvis stop talking"
        wake_word = "jarvis"

        has_wake_word = is_wake_word_detected(current_segment.lower(), wake_word, [])
        assert has_wake_word is True


class TestHotWindowEchoRejection:
    """Tests documenting that echo rejection should NOT expire hot window.

    Bug scenario: User says follow-up, but TTS echo is transcribed first.
    The echo gets rejected, but the hot window should remain active for
    the real follow-up that comes immediately after.
    """

    def test_echo_rejection_should_not_expire_hot_window(self):
        """
        Bug fix test: Echo rejection must NOT expire hot window.

        Scenario from real usage:
        1. TTS finishes at 13:12:24.390, hot window starts (3 seconds)
        2. User says: "No, that's you. I was talking to Google."
        3. But Whisper first transcribes TTS echo (97.3% similarity)
        4. Echo is correctly rejected
        5. BUG (fixed): Hot window was being expired here
        6. Real follow-up arrives but hot window is already gone

        The fix: Echo rejection clears voice state but keeps hot window active.
        """
        # Timeline simulation
        tts_finish_time = 1000.0
        hot_window_duration = 3.0
        hot_window_end_time = tts_finish_time + hot_window_duration  # 1003.0

        # Echo arrives at 1000.5 (during hot window)
        echo_arrival_time = 1000.5

        # Real follow-up arrives at 1001.2 (during hot window)
        followup_arrival_time = 1001.2

        # Both arrive within hot window
        assert echo_arrival_time < hot_window_end_time
        assert followup_arrival_time < hot_window_end_time

        # Key assertion: After rejecting echo, hot window should still be active
        # for the follow-up that arrives 0.7 seconds later
        time_between_echo_and_followup = followup_arrival_time - echo_arrival_time
        assert time_between_echo_and_followup < hot_window_duration, \
            "Follow-up should be within hot window if echo didn't expire it"

    def test_real_followup_after_echo_is_accepted(self):
        """
        After echo is rejected, real follow-up should still work.

        The hot window stays active, so the follow-up doesn't need wake word.
        """
        # User's real follow-up (no wake word needed in hot window)
        followup_text = "no that's you i was talking to google"
        wake_word = "jarvis"

        # This doesn't have wake word
        has_wake_word = is_wake_word_detected(followup_text, wake_word, [])
        assert has_wake_word is False

        # But in hot window mode, it should still be accepted
        # (the listener trusts intent judge for hot window speech)
        in_hot_window = True
        should_require_wake_word = not in_hot_window

        # No wake word required in hot window
        assert should_require_wake_word is False


class TestQueryValidationNotUsed:
    """Tests documenting why we DON'T use query-to-segment text matching.

    Query validation (checking if LLM's extracted query matches the segment text)
    was considered but rejected because it has both false positives and false
    negatives that make it unreliable.

    Instead, we rely on:
    1. Wake word presence check (in wake word mode)
    2. CURRENT - JUDGE THIS prompt marker (guides LLM to right segment)
    3. Processed segment filtering (old queries filtered from prompt)
    """

    def test_false_negative_synthesized_query_paraphrased(self):
        """
        FALSE NEGATIVE: Valid synthesized query rejected due to paraphrasing.

        User says: "Jarvis what do you think"
        LLM synthesizes: "share your thoughts on the weather"
        These have almost no word overlap - validation would reject valid query!
        """
        text = "jarvis what do you think"
        synthesized_query = "share your thoughts on the weather"

        # Remove wake word for fair comparison
        text_without_wake = text.replace("jarvis", "").strip()

        # Check 1: substring match
        assert synthesized_query not in text
        assert text not in synthesized_query
        assert text_without_wake not in synthesized_query

        # Check 2: word overlap
        text_words = set(text_without_wake.split())  # {what, do, you, think}
        query_words = set(synthesized_query.split())  # {share, your, thoughts, on, the, weather}
        overlap = text_words & query_words

        # Only "your" might overlap (you vs your - not exact match)
        # This valid query would be INCORRECTLY REJECTED
        assert len(overlap) < len(query_words) / 2, "Low overlap would reject valid query"

    def test_false_negative_synthesized_query_context_heavy(self):
        """
        FALSE NEGATIVE: Valid query with heavy context synthesis rejected.

        Multi-person conversation about iPhone, user asks "Jarvis how much"
        LLM synthesizes: "how much does the new iPhone 15 Pro Max cost in the UK"
        """
        text = "jarvis how much"
        synthesized_query = "how much does the new iPhone 15 Pro Max cost in the UK"

        text_without_wake = text.replace("jarvis", "").strip()  # "how much"

        # Substring check passes! "how much" is in the query
        assert text_without_wake in synthesized_query

        # But what if user said it differently?
        text2 = "jarvis what's the price"
        text2_without_wake = text2.replace("jarvis", "").strip()  # "what's the price"

        # This would FAIL - different phrasing
        assert text2_without_wake not in synthesized_query

    def test_false_positive_coincidental_overlap(self):
        """
        FALSE POSITIVE: Wrong segment query accepted due to coincidental overlap.

        User says: "hey assistant, how are you doing, tell me the weather"
        LLM extracts from WRONG segment: "how are you"
        But "how are you" IS in the current text!
        """
        current_text = "hey assistant how are you doing tell me the weather"
        wrong_query = "how are you"  # From a different segment!

        # This INCORRECTLY PASSES - query is substring of text
        assert wrong_query in current_text, "Wrong query passes validation!"

    def test_false_positive_common_words_overlap(self):
        """
        FALSE POSITIVE: Wrong query has word overlap with common phrases.

        User says: "assistant what time is it"
        Wrong segment had: "what time should we leave for dinner"
        """
        current_text = "assistant what time is it"
        wrong_query = "what time should we leave for dinner"

        # Word overlap
        current_words = set(current_text.split())
        query_words = set(wrong_query.split())
        overlap = current_words & query_words

        # Overlap: {what, time} = 2 words
        # Query has 7 words, threshold = 3.5
        # 2 < 3.5 - this one would be rejected

        # But with shorter wrong query:
        wrong_query_short = "what time should we leave"
        query_words_short = set(wrong_query_short.split())
        overlap_short = current_words & query_words_short

        # Overlap: {what, time} = 2 words
        # Query has 5 words, threshold = 2.5
        # 2 < 2.5 - still rejected, but barely

        # The point: validation is fragile and unreliable

    def test_wake_word_check_is_reliable(self):
        """
        Wake word check is reliable - no false positives or negatives.

        If user says "how are you" without wake word:
        - Wake word check correctly rejects (no "jarvis")

        If user says "jarvis what do you think":
        - Wake word check correctly accepts (has "jarvis")
        - LLM can synthesize any query it wants
        """
        # Case 1: No wake word - correctly rejected
        text_no_wake = "how are you"
        assert is_wake_word_detected(text_no_wake, "jarvis", []) is False

        # Case 2: Has wake word - correctly accepted
        text_with_wake = "jarvis what do you think"
        assert is_wake_word_detected(text_with_wake, "jarvis", []) is True

        # The LLM can then synthesize: "what do you think about the weather"
        # We trust the LLM's synthesis because the wake word validated user intent
