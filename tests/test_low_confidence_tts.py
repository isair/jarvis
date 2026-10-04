"""Spoken feedback behaviour for fully rejected low-confidence utterances."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jarvis.listening import listener as listener_module


pytestmark = pytest.mark.unit


class FakeTTS:
    """Minimal TTS stand-in with the guard surface the listener relies on."""

    def __init__(self, speaking=False, enabled=True):
        self.enabled = enabled
        self._speaking = speaking
        self.spoken = []

    def is_speaking(self):
        return self._speaking

    def speak(self, text, completion_callback=None, duration_callback=None):
        self.spoken.append(text)


@pytest.fixture
def make_listener(monkeypatch):
    monkeypatch.setattr(listener_module, "create_intent_judge", lambda cfg: None)
    monkeypatch.setattr(listener_module, "np", np)

    def make(tts=None, **cfg_overrides):
        cfg = SimpleNamespace(
            vad_enabled=False,
            voice_debug=False,
            whisper_min_confidence=0.5,
            whisper_no_speech_threshold=0.6,
            whisper_min_audio_duration=0.1,
            **cfg_overrides,
        )
        return listener_module.VoiceListener(None, cfg, tts, None)

    return make


def make_result(listener, text="", events=None, **overrides):
    if events is None:
        events = (listener_module.LowConfidenceEvent(0.2, "mumbled words"),)
    fields = dict(
        text=text,
        language="en",
        low_confidence_events=events,
        start_time=0.0,
        end_time=1.0,
        energy=0.01,
        dictation_generation=listener._dictation_generation,
        captured_during_tts=False,
        captured_tts_start_time=0.0,
    )
    fields.update(overrides)
    return listener_module._TranscriptionResult(**fields)


def test_speaks_configured_phrase_for_fully_rejected_utterance(make_listener):
    tts = FakeTTS()
    listener = make_listener(
        tts, low_confidence_feedback_phrase="Please say that again"
    )
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == ["Please say that again"]


def test_speaks_default_phrase_when_phrase_not_configured(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == [listener_module.DEFAULT_LOW_CONFIDENCE_FEEDBACK_PHRASE]


def test_empty_configured_phrase_disables_feedback(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts, low_confidence_feedback_phrase="")
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == []


def test_no_feedback_when_tts_already_speaking(make_listener):
    tts = FakeTTS(speaking=True)
    listener = make_listener(tts)
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == []


def test_no_feedback_when_tts_disabled(make_listener):
    tts = FakeTTS(enabled=False)
    listener = make_listener(tts)
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == []


def test_no_feedback_without_rejected_segments(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    listener._handle_transcription_result(make_result(listener, events=()))
    assert tts.spoken == []


def test_no_feedback_for_audio_captured_during_tts(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    result = make_result(listener, captured_during_tts=True)
    listener._handle_transcription_result(result)
    assert tts.spoken == []


def test_no_feedback_during_dictation(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    listener._dictation_is_active = True
    listener._handle_transcription_result(make_result(listener))
    assert tts.spoken == []


def test_feedback_spoken_once_for_multiple_rejected_segments(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    events = (
        listener_module.LowConfidenceEvent(0.2, "first"),
        listener_module.LowConfidenceEvent(0.1, "second"),
        listener_module.LowConfidenceEvent(0.3, "third"),
    )
    listener._handle_transcription_result(make_result(listener, events=events))
    assert len(tts.spoken) == 1


def test_no_feedback_when_part_of_the_utterance_was_accepted(make_listener):
    tts = FakeTTS()
    listener = make_listener(tts)
    listener._process_transcript = Mock()
    result = make_result(listener, text="turn on the lights")
    listener._handle_transcription_result(result)
    assert tts.spoken == []
