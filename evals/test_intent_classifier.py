"""Classifier-only intent evals using the listening judge's full input contract.

Set EVAL_INTENT_CLASSIFIER_URL to opt into a local model. An unavailable model,
malformed answer or abstention fails the case. There is no generative fallback.
"""

import os
import time

import pytest

from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig
from jarvis.listening.transcript_buffer import TranscriptSegment

from evals.intent_classifier import classify_intent
from evals.test_intent_judge import (
    INTENT_JUDGE_TEST_CASES,
    MULTI_SEGMENT_TEST_CASES,
    MultiSegmentTestCase,
)


def speech_case(name, text, *, directed=True, stop=False, echo=False, hot=False, tts=""):
    return MultiSegmentTestCase(
        name=name, segments=[(text, echo)], last_tts_text=tts,
        in_hot_window=hot, wake_timestamp=None if hot or echo else 1000.8,
        expected_directed=directed,
        expected_stop=stop,
    )


ADDITIONAL_CASES = [
    speech_case("spanish_question", "Jarvis qué tiempo hace en Madrid"),
    speech_case("turkish_statement", "Jarvis bugün kendimi yorgun hissediyorum"),
    speech_case("japanese_question", "Jarvis、今何時ですか"),
    speech_case("french_stop", "Jarvis arrête de parler", stop=True),
    speech_case("spanish_stop", "Jarvis deja de hablar", stop=True),
    speech_case("turkish_stop", "Jarvis konuşmayı bırak", stop=True),
    speech_case("french_narrative", "Hier j'ai parlé de Jarvis à mon ami", directed=False),
    speech_case("spanish_hot_followup", "Y mañana", hot=True, tts="Hoy hace sol"),
    speech_case("stop_word_definition", "Jarvis what does the word stop mean"),
    speech_case("bus_stop_question", "Jarvis where is the nearest bus stop"),
    speech_case("quoted_stop", "Jarvis my colleague said stop talking, what should I do"),
    speech_case("only_tts_echo", "The weather is sunny", echo=True, directed=False,
                tts="The weather is sunny"),
]
CASES = INTENT_JUDGE_TEST_CASES + MULTI_SEGMENT_TEST_CASES + ADDITIONAL_CASES


@pytest.fixture(scope="module")
def local_classifier_url():
    url = os.environ.get("EVAL_INTENT_CLASSIFIER_URL")
    if not url:
        pytest.skip("Set EVAL_INTENT_CLASSIFIER_URL to qualify a local classifier")
    return url


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_intent_classification(case, local_classifier_url, record_property):
    if hasattr(case, "segments"):
        entries = case.segments
    else:
        entries = [(case.transcript, False)]
    segments = [
        TranscriptSegment(text, 1000 + i * 2, 1002 + i * 2, is_during_tts=echo)
        for i, (text, echo) in enumerate(entries)
    ]
    current_text = next((text for text, echo in reversed(entries) if not echo), entries[-1][0])
    judge = IntentJudge(IntentJudgeConfig(aliases=getattr(case, "aliases", None) or []))
    state = judge._build_user_prompt(
        segments, case.wake_timestamp, case.last_tts_text,
        999.0 if case.last_tts_text else 0.0, case.in_hot_window, current_text,
    )
    started = time.monotonic()
    decision = classify_intent(
        state, base_url=local_classifier_url,
        model=os.environ.get("EVAL_INTENT_CLASSIFIER_MODEL", "multilingual"),
        assistant_name=judge.config.assistant_name,
        threshold=float(os.environ.get("EVAL_INTENT_CLASSIFIER_THRESHOLD", "0.9")),
    )
    record_property("classifier_latency_ms", round((time.monotonic() - started) * 1000, 1))
    assert decision is not None, "Classifier abstained; this is not a successful judgement"
    assert decision.directed == case.expected_directed
    assert decision.stop == case.expected_stop
