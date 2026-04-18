"""
Evals for the Intent Judge LLM.

Deduplicated suite: 22 cases covering all behaviour axes from the original 59.
See PR description / commit message for the dedup rationale.
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
    wake_timestamp: Optional[float]
    expected_directed: bool
    expected_query_contains: Optional[str]
    expected_query_not_contains: Optional[str] = None
    expected_stop: bool = False


# Single-segment cases - one per distinct behaviour axis.
INTENT_JUDGE_TEST_CASES = [
    # Wake word + simple question (canonical directed+extract)
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
    # Wake word + command/imperative addressed to the assistant (not a question)
    IntentJudgeTestCase(
        name="wake_word_command_timer",
        transcript="Jarvis set a timer for 5 minutes",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="timer",
        expected_query_not_contains="jarvis",
    ),
    # Wake word + statement/command to remember something
    IntentJudgeTestCase(
        name="wake_word_statement_remember",
        transcript="Jarvis remind me to call mum at 5pm",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="mum",
    ),
    # Same-segment context synthesis (distinct from simple wake+Q)
    IntentJudgeTestCase(
        name="context_synthesis_weather_opinion",
        transcript="I think the weather is great today in London. What do you think, Jarvis?",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=True,
        expected_query_contains="weather",
    ),
    # Echo + user follow-up in hot window
    IntentJudgeTestCase(
        name="echo_plus_followup_extracted",
        transcript="London has 8 hours of daylight. That's quite cool. Tell me more.",
        last_tts_text="On this day, London receives around 7-8 hours of daylight.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="more",
    ),
    # Stop command during TTS
    IntentJudgeTestCase(
        name="stop_command_during_tts",
        transcript="stop",
        last_tts_text="Let me tell you about the history of...",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains=None,
        expected_stop=True,
    ),
    # No wake word, not hot window -> not directed
    IntentJudgeTestCase(
        name="no_wake_word_casual_speech",
        transcript="I think the weather is nice today",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Wake word only mentioned in narrative -> not directed
    IntentJudgeTestCase(
        name="mentioned_in_narrative_past_tense",
        transcript="I told my friend about Jarvis yesterday",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Hot window simple follow-up
    IntentJudgeTestCase(
        name="hot_window_simple_followup",
        transcript="What about next week?",
        last_tts_text="The weather this weekend will be rainy.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="next week",
    ),
]


@dataclass
class MultiSegmentTestCase:
    """Test case with multiple transcript segments (realistic buffer state)."""
    name: str
    segments: list
    last_tts_text: str
    in_hot_window: bool
    wake_timestamp: Optional[float]
    expected_directed: bool
    expected_query_contains: Optional[str]
    expected_query_not_contains: Optional[str] = None
    expected_stop: bool = False


MULTI_SEGMENT_TEST_CASES = [
    # Real-logs scenario: echo + rejected similar + wake retry
    MultiSegmentTestCase(
        name="echo_plus_rejected_similar_plus_wake_retry",
        segments=[
            ("and relatively windy, about 11 kilometers per hour", False),
            ("Okay, well, what about any new movies tomorrow?", False),
            ("Jarvis, what about new movies tomorrow?", False),
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected. It'll be quite cool, around 6°C, and relatively windy, about 11 km/h.",
        in_hot_window=False,
        wake_timestamp=1004.5,
        expected_directed=True,
        expected_query_contains="movies",
        expected_query_not_contains="weather",
    ),
    # Hot window with echo in buffer + user follow-up
    MultiSegmentTestCase(
        name="buffer_echo_then_followup_hot_window",
        segments=[
            ("The weather is sunny and warm", False),
            ("What about the weekend?", False),
        ],
        last_tts_text="The weather today is sunny and warm, around 20 degrees.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="weekend",
        expected_query_not_contains="sunny",
    ),
    # Stop command with TTS echoes in buffer
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
        expected_directed=True,
        expected_query_contains=None,
        expected_stop=True,
    ),
    # No wake word in multi-segment buffer
    MultiSegmentTestCase(
        name="no_wake_word_in_buffer",
        segments=[
            ("How are you?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Context synthesis with prior ambient speech that must be filtered
    MultiSegmentTestCase(
        name="context_synthesis_with_prior_ambient",
        segments=[
            ("Did you see the game last night?", False),
            ("Yeah it was amazing", False),
            ("The food here is excellent. Jarvis, what's the best dish to order?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="dish",
        expected_query_not_contains="game",
    ),
    # Multi-person conversation: context synthesis across speakers without explicit pronoun
    MultiSegmentTestCase(
        name="multi_person_weather_discussion",
        segments=[
            ("I wonder what the weather will be like tomorrow", False),
            ("Yeah we should check before planning the picnic", False),
            ("Jarvis what do you think", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="weather",
    ),
    # Multi-person + vague reference ("that" = iPhone from earlier segment)
    MultiSegmentTestCase(
        name="multi_person_vague_reference",
        segments=[
            ("The new iPhone looks pretty cool", False),
            ("I heard the camera is amazing", False),
            ("Jarvis how much does that cost", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="iphone",
    ),
    # User statement follow-up in hot window (not an echo of TTS question)
    MultiSegmentTestCase(
        name="user_followup_statement_after_question_nihilism",
        segments=[
            ("Some people find that appealing", True),
            ("While others see it as a bleak outlook", True),
            ("What are your thoughts on nihilism", True),
            ("I think it's way more ridiculous than absurdism. Absurdism is the way to go.", False),
        ],
        last_tts_text="Nihilism is an interesting philosophical position. Some people find it appealing, while others see it as a bleak outlook. What are your thoughts on nihilism?",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="absurdism",
        expected_query_not_contains="what are your thoughts",
    ),
    # Cross-segment vague reference ("that" -> dinosaurs)
    MultiSegmentTestCase(
        name="cross_segment_dinosaur_opinion",
        segments=[
            ("I think dinosaurs are cool", False),
            ("What do you think about that Jarvis", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="dinosaur",
    ),
    # Imperative resolution: "answer that" -> re-issue prior question
    MultiSegmentTestCase(
        name="cross_segment_answer_that_weather",
        segments=[
            ("Sorry, how's the weather today?", False),
            ("Jarvis, answer that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="answer that",
    ),
    # Imperative resolution with unrelated noise between Q and imperative
    MultiSegmentTestCase(
        name="cross_segment_answer_that_with_noise",
        segments=[
            ("How tall is Mount Everest", False),
            ("Charlie sands to that", False),
            ("Jarvis answer that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.5,
        expected_directed=True,
        expected_query_contains="everest",
        expected_query_not_contains="answer that",
    ),
    # Whisper tense variant of imperative ("answered that")
    MultiSegmentTestCase(
        name="cross_segment_answered_that_whisper_variant",
        segments=[
            ("Sorry, how's the weather today?", False),
            ("Jarvis answered that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="answered that",
    ),
    # Multi-word imperative variant
    MultiSegmentTestCase(
        name="cross_segment_go_ahead_and_answer",
        segments=[
            ("What's the capital of Portugal", False),
            ("Jarvis go ahead and answer", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="portugal",
        expected_query_not_contains="go ahead and answer",
    ),
    # Imperative superseded by new explicit question in same segment
    MultiSegmentTestCase(
        name="cross_segment_imperative_superseded_by_new_question",
        segments=[
            ("How's the weather today?", False),
            ("Jarvis, answer that — actually, what time is it?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="time",
        expected_query_not_contains="weather",
    ),
    # Cross-segment follow-up in hot window (topic extension)
    MultiSegmentTestCase(
        name="cross_segment_hot_window_followup",
        segments=[
            ("The capital of France is Paris", True),
            ("What about Germany", False),
        ],
        last_tts_text="The capital of France is Paris, known as the City of Light.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="germany",
    ),
]


# Cases known to fail with the small model on the current prompt.
# Kept empty during baseline runs so we observe true pass/fail.
KNOWN_FAILING_CASES: set = set()


# =============================================================================
# Helper Functions
# =============================================================================

def create_transcript_segment(
    text: str,
    start_time: float = 1000.0,
    is_during_tts: bool = False,
    processed: bool = False,
):
    """Create a TranscriptSegment for testing."""
    from jarvis.listening.transcript_buffer import TranscriptSegment
    return TranscriptSegment(
        text=text,
        start_time=start_time,
        end_time=start_time + 2.0,
        energy=0.01,
        is_during_tts=is_during_tts,
        processed=processed,
    )


def run_intent_judge(case: IntentJudgeTestCase):
    """Run the intent judge on a test case."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        model="gemma4:e2b",
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
        current_text=case.transcript,
    )


def run_intent_judge_multi_segment(case: "MultiSegmentTestCase"):
    """Run the intent judge on a multi-segment test case."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        model="gemma4:e2b",
        timeout_sec=10.0,
    ))

    if not judge.available:
        return None

    segments = []
    base_time = 1000.0
    for i, (text, is_during_tts) in enumerate(case.segments):
        segments.append(create_transcript_segment(
            text=text,
            start_time=base_time + (i * 2.0),
            is_during_tts=is_during_tts,
        ))

    current_text = ""
    for text, is_during_tts in reversed(case.segments):
        if not is_during_tts:
            current_text = text
            break

    return judge.judge(
        segments=segments,
        wake_timestamp=case.wake_timestamp,
        last_tts_text=case.last_tts_text,
        last_tts_finish_time=999.0 if case.last_tts_text else 0.0,
        in_hot_window=case.in_hot_window,
        current_text=current_text,
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
        return any("gemma4" in m for m in models)
    except Exception:
        return False


def _skip_if_not_intent_judge_phase():
    """Intent judge tests are fixed to gemma4:e2b and would run twice under the
    multi-model eval matrix. Skip during the large-model phase to keep runtime
    down; they still run once during the small-model (gemma4) phase."""
    if "gemma4" not in JUDGE_MODEL:
        pytest.skip(f"Intent judge tests only run in the gemma4 phase (current: {JUDGE_MODEL})")


# =============================================================================
# Tests
# =============================================================================

class TestIntentJudgeAccuracy:
    """Evals for intent judge accuracy."""

    @pytest.mark.parametrize("case", INTENT_JUDGE_TEST_CASES, ids=lambda c: c.name)
    def test_intent_judge_case(self, case: IntentJudgeTestCase):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        if case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known issue: {case.name} needs prompt improvement")

        result = run_intent_judge(case)

        if result is None:
            pytest.fail("Intent judge returned None")

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

        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )
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

        assert "HOT WINDOW" in prompt

    def test_tts_text_included_for_echo_detection(self):
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

        assert "nice and sunny" in prompt

    def test_system_prompt_has_echo_guidance(self):
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        assert "echo" in prompt.lower()
        assert "(during TTS)" in prompt


class TestIntentJudgeFallback:
    """Tests for intent judge fallback behaviour."""

    def test_returns_none_when_ollama_unavailable(self):
        from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

        judge = IntentJudge(IntentJudgeConfig(
            ollama_base_url="http://127.0.0.1:99999",
            timeout_sec=1.0,
        ))

        segments = [create_transcript_segment("test")]
        result = judge.judge(segments)

        assert result is None


class TestIntentJudgeMultiSegment:
    """Evals for intent judge with realistic multi-segment transcript buffers."""

    @pytest.mark.parametrize("case", MULTI_SEGMENT_TEST_CASES, ids=lambda c: c.name)
    def test_multi_segment_case(self, case: MultiSegmentTestCase):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        if case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known issue: {case.name} needs prompt improvement")

        result = run_intent_judge_multi_segment(case)

        if result is None:
            pytest.fail("Intent judge returned None")

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

        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )
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


class TestProcessedSegmentFiltering:
    """Tests for processed segment filtering in intent judge."""

    def test_processed_segment_not_reextracted(self):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

        judge = IntentJudge(IntentJudgeConfig(
            assistant_name="Jarvis",
            model="gemma4:e2b",
            timeout_sec=10.0,
        ))

        segments = [
            create_transcript_segment(
                text="Jarvis what's the weather in London",
                start_time=1000.0,
                processed=True,
            ),
            create_transcript_segment(
                text="Jarvis tell me a random topic",
                start_time=1010.0,
                processed=False,
            ),
        ]

        result = judge.judge(
            segments=segments,
            wake_timestamp=1010.0,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
            current_text="Jarvis tell me a random topic",
        )

        assert result is not None
        assert result.directed is True
        assert "random" in result.query.lower() or "topic" in result.query.lower(), (
            f"Expected query about 'random topic', got '{result.query}'."
        )
        assert "weather" not in result.query.lower(), (
            f"Query contains 'weather' from processed segment: '{result.query}'"
        )

        print(f"\n✅ Correctly extracted new query: '{result.query}'")
