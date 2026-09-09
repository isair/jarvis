"""Low-confidence events expose rejected speech without changing voice behaviour."""

from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest

from jarvis.listening import listener as listener_module


pytestmark = pytest.mark.unit


@pytest.fixture
def make_listener(monkeypatch):
    monkeypatch.setattr(listener_module, "create_intent_judge", lambda cfg: None)
    monkeypatch.setattr(listener_module, "np", np)

    def make(**kwargs):
        cfg = SimpleNamespace(
            vad_enabled=False,
            voice_debug=False,
            whisper_min_confidence=0.5,
            whisper_no_speech_threshold=0.6,
            whisper_min_audio_duration=0.1,
        )
        return listener_module.VoiceListener(None, cfg, None, None, **kwargs)

    return make


@pytest.fixture(params=["faster-whisper", "mlx"])
def transcribe(request, monkeypatch):
    """Run the real utterance pipeline with deterministic speech-model output."""
    def run(listener, segments):
        listener._whisper_backend = request.param
        if request.param == "mlx":
            model = SimpleNamespace(transcribe=lambda *args, **kwargs: {
                "segments": segments, "language": "en",
            })
            monkeypatch.setattr(listener_module, "mlx_whisper", model, raising=False)
        else:
            listener.model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
                iter(SimpleNamespace(**seg) for seg in segments),
                SimpleNamespace(language="en"),
            ))
        listener._utterance_frames = [np.zeros(listener._samplerate, dtype=np.float32)]
        listener.echo_detector._utterance_start_time = time.time() - 1
        listener._process_transcript = Mock()
        listener._finalize_utterance()
        return [call.args[0] for call in listener._process_transcript.call_args_list]

    return run


def segment(text, confidence, no_speech_prob=0):
    return {"text": text, "avg_logprob": confidence - 1,
            "no_speech_prob": no_speech_prob}


def test_rejections_expose_raw_transcript_confidence_and_reason(make_listener, transcribe):
    events = []
    listener = make_listener(on_low_confidence=events.append)
    threshold = listener.cfg.whisper_min_confidence
    rejected = [
        segment("  Could you repeat that? " * 4, threshold / 2),
        segment(" unclear speech ", threshold / 10),
    ]

    assert transcribe(listener, rejected) == []
    assert len(events) == len(rejected)
    for event, original in zip(events, rejected):
        assert isinstance(event, listener_module.LowConfidenceEvent)
        assert event.transcript == original["text"]
        assert event.confidence == pytest.approx(original["avg_logprob"] + 1)
        assert event.reason == "low_confidence"


def test_accepted_segments_and_no_speech_rejections_do_not_emit(make_listener, transcribe):
    events = []
    listener = make_listener(on_low_confidence=events.append)
    threshold = listener.cfg.whisper_min_confidence
    segments = [
        segment("hello", threshold),
        segment("world", (threshold + 1) / 2),
        segment("background noise", threshold / 2,
                listener.cfg.whisper_no_speech_threshold),
    ]

    assert transcribe(listener, segments) == ["hello world"]
    assert events == []


def test_mixed_utterance_preserves_accepted_speech(make_listener, transcribe):
    events = []
    listener = make_listener(on_low_confidence=events.append)
    threshold = listener.cfg.whisper_min_confidence

    assert transcribe(listener, [
        segment("hello", threshold),
        segment("unclear", threshold / 2),
        segment("world", threshold),
    ]) == ["hello world"]
    assert [event.transcript for event in events] == ["unclear"]


def test_callback_is_optional(make_listener, transcribe):
    listener = make_listener()
    threshold = listener.cfg.whisper_min_confidence

    assert transcribe(listener, [
        segment("unclear", threshold / 2),
        segment("hello", threshold),
    ]) == ["hello"]


def test_callback_failure_does_not_discard_accepted_speech(make_listener, transcribe):
    events = []

    def failing_callback(event):
        events.append(event)
        raise RuntimeError("Consumer failed")

    listener = make_listener(on_low_confidence=failing_callback)
    threshold = listener.cfg.whisper_min_confidence

    assert transcribe(listener, [
        segment("unclear", threshold / 2),
        segment("hello", threshold),
        segment("also unclear", threshold / 10),
    ]) == ["hello"]
    assert [event.transcript for event in events] == ["unclear", "also unclear"]


def test_faster_whisper_confidence_fallback(make_listener):
    events = []
    listener = make_listener(on_low_confidence=events.append)
    listener.cfg.whisper_no_speech_threshold = 0.9
    rejected = SimpleNamespace(text="unclear", no_speech_prob=0.75)
    unknown = SimpleNamespace(text="no confidence metadata")

    assert listener._filter_noisy_segments([rejected, unknown]) == [unknown]
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(1 - rejected.no_speech_prob)
    assert events[0].transcript == rejected.text
