"""Exercise acoustic interruption through queued, synthetic microphone blocks."""

import json
import queue
from unittest.mock import Mock

import numpy as np
import pytest

from jarvis.config import load_settings
from jarvis.listening.listener import VoiceListener
from jarvis.output.tts import SpeechRequest
from tests.test_shush_detector import _make_shush

pytestmark = pytest.mark.unit


@pytest.fixture
def capture(tmp_path, monkeypatch):
    listeners = []
    monkeypatch.setattr("jarvis.listening.listener.create_intent_judge", lambda settings: None)

    def open_capture(*, stream_rate=16000, **settings):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "sample_rate": 16000,
            "tune_enabled": False,
            "vad_enabled": False,
            **settings,
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(config))
        stopped = []
        current = None
        tts = Mock(spec=["is_speaking", "set_playback_callbacks", "interrupt"])
        tts.is_speaking.side_effect = lambda: current is not None

        def stop(request):
            nonlocal current
            if current is request:
                stopped.append(request.text)
                listener.on_playback_finished(request, "interrupted")
                current = None

        tts.interrupt.side_effect = lambda: stop(current)
        listener = VoiceListener(None, load_settings(), tts, None)
        listeners.append(listener)
        listener._stream_samplerate = stream_rate
        listener._frame_samples = stream_rate // 50

        def play(text):
            nonlocal current
            if current is not None:
                current._finish("completed")
                listener.on_playback_finished(current, "completed")
            current = SpeechRequest(text, _interrupt_callback=stop)
            assert current._claim()
            listener.on_playback_started(current, 2.0)

        class CapturedQueue(queue.Queue):
            def get(self, block=True, timeout=None):
                try:
                    return super().get(block=False)
                except queue.Empty:
                    listener._should_stop = True
                    raise

        listener._audio_q = CapturedQueue()

        def feed(frame, count=1):
            chunk = np.tile(frame, count).reshape(-1, 1)
            listener._on_audio(chunk, len(chunk), None, None)

        def process():
            listener._consume_audio_frames(20)
            return stopped

        return play, feed, process

    yield open_capture
    for listener in listeners:
        listener.stop()


def test_acoustic_interrupt_can_be_disabled_through_config(capture):
    play, feed, process = capture(shush_detection_enabled=False)
    play("answer")
    feed(_make_shush(), 15)
    assert process() == []


def test_capture_before_playback_cannot_stop_the_reply(capture):
    play, feed, process = capture()
    feed(_make_shush(), 15)
    play("later answer")
    assert process() == []


def test_capture_from_finished_playback_cannot_stop_replacement(capture):
    play, feed, process = capture(shush_consecutive_frames=3)
    play("first answer")
    feed(_make_shush(), 2)
    play("replacement")
    feed(_make_shush(), 1)
    assert process() == []


def test_a_partial_frame_from_before_playback_is_ineligible(capture):
    play, feed, process = capture(shush_consecutive_frames=1)
    frame = _make_shush()
    feed(frame[:160])
    play("later answer")
    feed(frame[160:])
    assert process() == []


@pytest.mark.parametrize("frequency", [3000, 3017])
def test_sustained_tones_do_not_interrupt(capture, frequency):
    play, feed, process = capture()
    play("answer")
    phase = 2 * np.pi * frequency * np.arange(320 * 15) / 16000
    feed((0.05 * np.sin(phase)).astype(np.float32))
    assert process() == []


@pytest.mark.parametrize("stream_rate", [16000, 48000])
def test_acoustic_interrupt_targets_current_reply_at_capture_rate(capture, stream_rate):
    play, feed, process = capture(stream_rate=stream_rate, shush_consecutive_frames=3)
    play("answer")
    feed(_make_shush(stream_rate // 50, sample_rate=stream_rate), 3)
    assert process() == ["answer"]


def test_configured_energy_floor_is_applied(capture):
    play, feed, process = capture(shush_energy_floor=0.2)
    play("answer")
    feed(_make_shush(), 15)
    assert process() == []
