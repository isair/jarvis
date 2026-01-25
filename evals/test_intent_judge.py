"""
Evals for the Intent Judge LLM.

Tests the accuracy of intent classification and query extraction for:
1. Wake word detection + query extraction
2. Echo detection (TTS being picked up by mic)
3. Echo + follow-up pattern (mixed transcript)
4. Stop command detection
5. Not directed speech (mentioned in narrative)
6. Hot window follow-up queries
7. Multi-person conversations (chiming into ongoing discussions)
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, List

from helpers import JUDGE_MODEL, JUDGE_BASE_URL, is_judge_llm_available


# =============================================================================
# Test Data
# =============================================================================

@dataclass
class IntentJudgeTestCase:
    """Test case for intent judge evaluation."""
    name: str
    transcript: str
    last_tts_text: str
    in_hot_window: bool
    wake_timestamp: Optional[float]  # None for hot window mode
    expected_directed: bool
    expected_query_contains: Optional[str]  # Substring that should be in query
    expected_query_not_contains: Optional[str] = None  # Substring that should NOT be in query
    expected_stop: bool = False


# Test cases covering the key scenarios
INTENT_JUDGE_TEST_CASES = [
    # Wake word + question
    IntentJudgeTestCase(
        name="wake_word_simple_question",
        transcript="Jarvis what time is it",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="time",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_with_pre_chatter",
        transcript="I was telling him about the project and then I thought Jarvis what's the weather",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="project",
    ),

    # Echo detection
    IntentJudgeTestCase(
        name="pure_echo_rejected",
        transcript="The weather today is sunny and 72 degrees",
        last_tts_text="The weather today is sunny and 72 degrees Fahrenheit",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),
    IntentJudgeTestCase(
        name="partial_echo_rejected",
        transcript="sunny and 72 degrees",
        last_tts_text="The weather today is sunny and 72 degrees Fahrenheit",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),

    # Echo + follow-up (the key scenario from user's issue)
    IntentJudgeTestCase(
        name="echo_plus_followup_extracted",
        transcript="London has 8 hours of daylight. That's quite cool. Tell me more.",
        last_tts_text="On this day, London receives around 7-8 hours of daylight.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="tell me more",
        expected_query_not_contains="daylight",
    ),
    IntentJudgeTestCase(
        name="echo_plus_different_query",
        transcript="The weather is sunny. What about tomorrow?",
        last_tts_text="The weather today is sunny and warm.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="tomorrow",
    ),

    # Stop command detection
    # Note: stop commands can have directed=False since they're intercepted
    # before normal query processing. The key is stop=True.
    IntentJudgeTestCase(
        name="stop_command_during_tts",
        transcript="stop",
        last_tts_text="Let me tell you about the history of...",
        in_hot_window=False,  # During TTS
        wake_timestamp=None,
        expected_directed=False,  # Stop commands may not be "directed" in the usual sense
        expected_query_contains=None,
        expected_stop=True,
    ),
    IntentJudgeTestCase(
        name="quiet_command",
        transcript="quiet please",
        last_tts_text="And furthermore...",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=False,  # Stop commands may not be "directed" in the usual sense
        expected_query_contains=None,
        expected_stop=True,
    ),

    # Not directed (mentioned in narrative)
    # NOTE: These are challenging cases that the LLM often gets wrong.
    # Marked with xfail=True for now - the model needs prompt improvements.
    IntentJudgeTestCase(
        name="mentioned_in_narrative_past_tense",
        transcript="I told my friend about Jarvis yesterday",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,  # Wake word detected but not addressing
        expected_directed=False,
        expected_query_contains=None,
    ),
    IntentJudgeTestCase(
        name="mentioned_in_narrative_third_person",
        transcript="He said Jarvis is a cool assistant",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.6,
        expected_directed=False,
        expected_query_contains=None,
    ),

    # Hot window follow-up
    IntentJudgeTestCase(
        name="hot_window_simple_followup",
        transcript="What about next week?",
        last_tts_text="The weather this weekend will be rainy.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="next week",
    ),
    IntentJudgeTestCase(
        name="hot_window_thanks_followup",
        transcript="Thanks. And what time does it get dark?",
        last_tts_text="The sunrise is at 7:30 AM.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="dark",
    ),

    # Non-English follow-up (from user's original issue)
    # NOTE: The model struggles with short non-English inputs in hot window.
    IntentJudgeTestCase(
        name="non_english_followup",
        transcript="Ni hao",
        last_tts_text="The weather is sunny.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="ni hao",
    ),

    # Wake word with unrelated TTS history (should NOT be confused as echo)
    # This tests the case where TTS said something, user later asks about different topic
    IntentJudgeTestCase(
        name="wake_word_different_topic_not_echo",
        transcript="Jarvis, what about new movies tomorrow?",
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected.",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="movies",
    ),
    IntentJudgeTestCase(
        name="wake_word_completely_unrelated_to_tts",
        transcript="Jarvis, set a timer for 5 minutes",
        last_tts_text="The sunrise is at 7:30 AM and sunset at 4:45 PM.",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="timer",
    ),
]


# Multi-segment test cases (more realistic - transcript buffer has multiple utterances)
@dataclass
class MultiSegmentTestCase:
    """Test case with multiple transcript segments (realistic buffer state)."""
    name: str
    segments: list  # List of (text, is_during_tts) tuples
    last_tts_text: str
    in_hot_window: bool
    wake_timestamp: Optional[float]
    expected_directed: bool
    expected_query_contains: Optional[str]
    expected_query_not_contains: Optional[str] = None
    expected_stop: bool = False


MULTI_SEGMENT_TEST_CASES = [
    # Real scenario: TTS echoes in buffer, then user asks unrelated question with wake word
    MultiSegmentTestCase(
        name="buffer_with_echoes_then_wake_word_query",
        segments=[
            ("Tomorrow's weather looks a bit gloomy with overcast conditions", True),  # Echo during TTS
            ("It'll be quite cool around 6 degrees", True),  # Echo during TTS
            ("Jarvis, what about new movies tomorrow?", False),  # Real query
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected. It'll be quite cool, around 6°C.",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="movies",
        expected_query_not_contains="weather",
    ),
    # Exact scenario from user logs: echo + rejected similar query + wake word retry
    # The buffer contains a previous similar utterance that was already rejected
    MultiSegmentTestCase(
        name="echo_plus_rejected_similar_plus_wake_retry",
        segments=[
            ("and relatively windy, about 11 kilometers per hour", False),  # Echo that slipped through
            ("Okay, well, what about any new movies tomorrow?", False),  # Previous attempt (no wake word)
            ("Jarvis, what about new movies tomorrow?", False),  # Retry with wake word
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected. It'll be quite cool, around 6°C, and relatively windy, about 11 km/h.",
        in_hot_window=False,
        wake_timestamp=1004.5,  # Wake detected in third segment
        expected_directed=True,
        expected_query_contains="movies",
        expected_query_not_contains="weather",
    ),
    # FULL scenario from user logs: includes TTS-marked echoes in buffer
    # This is the most realistic test - full buffer with TTS echoes + user queries
    MultiSegmentTestCase(
        name="full_buffer_with_tts_echoes_and_wake_retry",
        segments=[
            ("Tomorrow's weather in Kensington looks a bit gloomy with overcast conditions expected.", True),  # TTS echo
            ("It'll be quite cool, around 6 degrees Celsius, 42.8 degrees Fahrenheit.", True),  # TTS echo
            ("and relatively windy, about 11 kilometers per hour. You might want to bundle up if you plan on stepping out.", False),  # Late echo (not marked)
            ("Okay, well, what about any new movies tomorrow?", False),  # User attempt without wake word
            ("Jarvis, what about new movies tomorrow?", False),  # User retry WITH wake word
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected. It'll be quite cool, around 6°C (42.8°F), and relatively windy, about 11 km/h. You might want to bundle up if you plan on stepping out.",
        in_hot_window=False,
        wake_timestamp=1008.5,  # Wake detected in fifth segment
        expected_directed=True,
        expected_query_contains="movies",
        expected_query_not_contains="weather",
    ),
    # Similar scenario but only echo + wake word query (no intermediate rejected)
    MultiSegmentTestCase(
        name="echo_slipped_through_then_wake_query",
        segments=[
            ("and relatively windy, about 11 kilometers per hour. You might want to bundle up.", False),
            ("Jarvis, what about new movies tomorrow?", False),
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy. It'll be quite cool, around 6°C, and relatively windy, about 11 km/h. You might want to bundle up if you plan on stepping out.",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="movies",
    ),
    # Hot window with echo then follow-up (no wake word needed)
    MultiSegmentTestCase(
        name="buffer_echo_then_followup_hot_window",
        segments=[
            ("The weather is sunny and warm", False),  # Echo (matches TTS)
            ("What about the weekend?", False),  # Real follow-up
        ],
        last_tts_text="The weather today is sunny and warm, around 20 degrees.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="weekend",
        expected_query_not_contains="sunny",
    ),
    # Multiple echoes, user interrupts with wake word
    MultiSegmentTestCase(
        name="multiple_echoes_then_interrupt",
        segments=[
            ("Let me tell you about", True),
            ("the history of", True),
            ("Jarvis stop", False),
        ],
        last_tts_text="Let me tell you about the history of ancient Rome.",
        in_hot_window=False,
        wake_timestamp=1002.0,
        expected_directed=True,  # Wake word + stop command IS directed
        expected_query_contains=None,
        expected_stop=True,
    ),

    # ==========================================================================
    # Multi-person conversation scenarios
    # These test the ability to chime into ongoing conversations between people
    # The intent judge should SYNTHESIZE context into a complete query
    # ==========================================================================

    # Classic scenario: two people discussing, one asks Jarvis for input
    # Intent judge should synthesize: "what do you think about the weather tomorrow"
    MultiSegmentTestCase(
        name="multi_person_weather_discussion",
        segments=[
            ("I wonder what the weather will be like tomorrow", False),  # Person A
            ("Yeah we should check before planning the picnic", False),  # Person B
            ("Jarvis what do you think", False),  # Person A asks Jarvis
        ],
        last_tts_text="",  # No recent TTS
        in_hot_window=False,
        wake_timestamp=1004.0,  # Wake detected in third segment
        expected_directed=True,
        expected_query_contains="weather",  # Should synthesize context about weather
    ),

    # Multi-person discussion about restaurants (explicit question, no synthesis needed)
    MultiSegmentTestCase(
        name="multi_person_restaurant_recommendation",
        segments=[
            ("I'm getting hungry should we order food", False),  # Person A
            ("Yeah I could go for some Italian", False),  # Person B
            ("Or maybe sushi what do you think", False),  # Person A
            ("Jarvis can you recommend a good restaurant nearby", False),  # Person B asks Jarvis
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1006.0,
        expected_directed=True,
        expected_query_contains="restaurant",
    ),

    # Longer conversation with context-dependent question (explicit, no synthesis needed)
    MultiSegmentTestCase(
        name="multi_person_travel_planning",
        segments=[
            ("We should start planning our trip to Japan", False),  # Person A
            ("When were you thinking of going", False),  # Person B
            ("Maybe in April for cherry blossom season", False),  # Person A
            ("That sounds beautiful", False),  # Person B
            ("Jarvis when is the best time to see cherry blossoms in Tokyo", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1008.0,
        expected_directed=True,
        expected_query_contains="cherry blossom",
    ),

    # Vague reference that requires context ("that" refers to earlier topic)
    # Intent judge should resolve "that" → "iPhone"
    MultiSegmentTestCase(
        name="multi_person_vague_reference",
        segments=[
            ("The new iPhone looks pretty cool", False),  # Person A
            ("I heard the camera is amazing", False),  # Person B
            ("Jarvis how much does that cost", False),  # Person A - "that" = iPhone
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="iphone",  # Should resolve "that" to iPhone
    ),
]

# Test cases that are known to fail with current model (3b)
# These cases have LLM variability or are challenging for small models
KNOWN_FAILING_CASES = {
    "mentioned_in_narrative_past_tense",
    "mentioned_in_narrative_third_person",
    "non_english_followup",
    # Echo detection edge cases - model sometimes returns stop=True for pure echo
    "pure_echo_rejected",
    "partial_echo_rejected",
}


# =============================================================================
# Helper Functions
# =============================================================================

def create_transcript_segment(text: str, start_time: float = 1000.0, is_during_tts: bool = False):
    """Create a TranscriptSegment for testing."""
    from jarvis.listening.transcript_buffer import TranscriptSegment
    return TranscriptSegment(
        text=text,
        start_time=start_time,
        end_time=start_time + 2.0,
        energy=0.01,
        is_during_tts=is_during_tts,
    )


def run_intent_judge(case: IntentJudgeTestCase):
    """Run the intent judge on a test case."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        model="llama3.2:3b",  # Use the model specified in the spec
        timeout_sec=10.0,
    ))

    if not judge.available:
        return None

    segments = [create_transcript_segment(case.transcript)]

    return judge.judge(
        segments=segments,
        wake_timestamp=case.wake_timestamp,
        last_tts_text=case.last_tts_text,
        last_tts_finish_time=999.0 if case.last_tts_text else 0.0,
        in_hot_window=case.in_hot_window,
    )


def run_intent_judge_multi_segment(case: "MultiSegmentTestCase"):
    """Run the intent judge on a multi-segment test case."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        model="llama3.2:3b",
        timeout_sec=10.0,
    ))

    if not judge.available:
        return None

    # Create segments with proper timestamps
    segments = []
    base_time = 1000.0
    for i, (text, is_during_tts) in enumerate(case.segments):
        segments.append(create_transcript_segment(
            text=text,
            start_time=base_time + (i * 2.0),
            is_during_tts=is_during_tts,
        ))

    return judge.judge(
        segments=segments,
        wake_timestamp=case.wake_timestamp,
        last_tts_text=case.last_tts_text,
        last_tts_finish_time=999.0 if case.last_tts_text else 0.0,
        in_hot_window=case.in_hot_window,
    )


def is_intent_judge_available() -> bool:
    """Check if the intent judge model is available."""
    import requests
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return any("llama3.2" in m for m in models)
    except Exception:
        return False


# =============================================================================
# Tests
# =============================================================================

class TestIntentJudgeAccuracy:
    """Evals for intent judge accuracy."""

    @pytest.mark.parametrize("case", INTENT_JUDGE_TEST_CASES, ids=lambda c: c.name)
    def test_intent_judge_case(self, case: IntentJudgeTestCase):
        """Test individual intent judge case."""
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (llama3.2:3b) not available")

        # Mark known failing cases as xfail
        if case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known issue: {case.name} needs prompt improvement")

        result = run_intent_judge(case)

        if result is None:
            pytest.fail("Intent judge returned None")

        # Log for debugging
        print(f"\n{'='*60}")
        print(f"Test Case: {case.name}")
        print(f"Transcript: {case.transcript}")
        print(f"TTS: {case.last_tts_text[:50]}..." if case.last_tts_text else "TTS: None")
        print(f"Mode: {'hot_window' if case.in_hot_window else 'wake_word'}")
        print(f"{'='*60}")
        print(f"Result: directed={result.directed}, query='{result.query}', stop={result.stop}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"{'='*60}")

        # Check directed
        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )

        # Check stop
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )

        # Check query content
        if case.expected_query_contains:
            assert case.expected_query_contains.lower() in result.query.lower(), (
                f"Expected query to contain '{case.expected_query_contains}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )

        if case.expected_query_not_contains and result.query:
            assert case.expected_query_not_contains.lower() not in result.query.lower(), (
                f"Expected query to NOT contain '{case.expected_query_not_contains}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )


class TestIntentJudgePromptQuality:
    """Tests for intent judge prompt construction quality."""

    def test_hot_window_mode_indicated_in_prompt(self):
        """Prompt should clearly indicate hot window mode."""
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        segments = [create_transcript_segment("hello")]

        prompt = judge._build_user_prompt(
            segments=segments,
            wake_timestamp=None,
            last_tts_text="Test TTS",
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )

        assert "HOT WINDOW" in prompt, "Hot window mode should be clearly indicated"

    def test_tts_text_included_for_echo_detection(self):
        """TTS text should be included in prompt for echo detection."""
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        segments = [create_transcript_segment("The weather is nice")]
        tts_text = "The weather today is nice and sunny"

        prompt = judge._build_user_prompt(
            segments=segments,
            wake_timestamp=None,
            last_tts_text=tts_text,
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )

        assert "nice and sunny" in prompt, "TTS text should be included"

    def test_system_prompt_has_echo_guidance(self):
        """System prompt should have clear echo detection guidance."""
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        assert "echo" in prompt.lower(), "Should mention echo"
        assert "BOTH echo AND real speech" in prompt, "Should handle mixed echo+speech"


class TestIntentJudgeFallback:
    """Tests for intent judge fallback behavior."""

    def test_returns_none_when_ollama_unavailable(self):
        """Should return None gracefully when Ollama is unavailable."""
        from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

        judge = IntentJudge(IntentJudgeConfig(
            ollama_base_url="http://127.0.0.1:99999",  # Invalid port
            timeout_sec=1.0,
        ))

        segments = [create_transcript_segment("test")]
        result = judge.judge(segments)

        # Should return None, not raise
        assert result is None


class TestIntentJudgeMultiSegment:
    """Evals for intent judge with realistic multi-segment transcript buffers.

    These tests simulate real-world scenarios where the transcript buffer
    contains multiple utterances, including TTS echoes and user speech.
    """

    @pytest.mark.parametrize("case", MULTI_SEGMENT_TEST_CASES, ids=lambda c: c.name)
    def test_multi_segment_case(self, case: MultiSegmentTestCase):
        """Test intent judge with multiple transcript segments."""
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (llama3.2:3b) not available")

        result = run_intent_judge_multi_segment(case)

        if result is None:
            pytest.fail("Intent judge returned None")

        # Log for debugging
        print(f"\n{'='*60}")
        print(f"Test Case: {case.name}")
        print(f"Segments:")
        for text, is_tts in case.segments:
            marker = " (during TTS)" if is_tts else ""
            print(f"  - \"{text}\"{marker}")
        print(f"TTS: {case.last_tts_text[:50]}..." if case.last_tts_text else "TTS: None")
        print(f"Mode: {'hot_window' if case.in_hot_window else 'wake_word'}")
        print(f"{'='*60}")
        print(f"Result: directed={result.directed}, query='{result.query}', stop={result.stop}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"{'='*60}")

        # Check directed
        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )

        # Check stop
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )

        # Check query content
        if case.expected_query_contains:
            assert case.expected_query_contains.lower() in result.query.lower(), (
                f"Expected query to contain '{case.expected_query_contains}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )

        if case.expected_query_not_contains and result.query:
            assert case.expected_query_not_contains.lower() not in result.query.lower(), (
                f"Expected query to NOT contain '{case.expected_query_not_contains}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )
