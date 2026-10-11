"""Captured dictation retains its own sample rate across overlapping sessions."""

import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from jarvis.dictation import dictation_engine as dictation

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def synthetic_hotkey(monkeypatch):
    monkeypatch.setattr(dictation, 'parse_hotkey', lambda hotkey: (frozenset(), None))


@pytest.mark.parametrize('recorded_rate,next_rate', [(48000, 16000), (16000, 48000), (44100, 48000)])
def test_released_audio_uses_its_recorded_rate(recorded_rate, next_rate, monkeypatch):
    closing = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    captured = []

    class Stream:
        def stop(self):
            closing.set()
            assert release.wait(5), 'Test did not release stream teardown'

        def close(self):
            pass

    engine = dictation.DictationEngine(whisper_model_ref=lambda: object(), whisper_backend_ref=lambda: 'faster-whisper', mlx_repo_ref=lambda: None, history=Mock(), on_dictation_end=finished.set)
    engine._recording = True
    engine._stream = Stream()
    engine._stream_sample_rate = recorded_rate
    engine._record_start_time = time.time()
    engine._audio_frames = [np.full(recorded_rate, 0.25, dtype=np.float32)]
    monkeypatch.setattr(dictation, '_play_beep', lambda *args: None)
    monkeypatch.setattr(engine, '_transcribe', lambda audio: captured.append(audio.copy()) or '')
    try:
        engine._stop_recording()
        assert closing.wait(5), 'The released recording did not reach teardown'
        # A later capture negotiates a different device format while this job waits.
        engine._stream_sample_rate = next_rate
    finally:
        release.set()
        assert finished.wait(5), 'The released recording did not finish'

    assert len(captured) == 1
    assert len(captured[0]) == engine._target_sample_rate
    np.testing.assert_allclose(captured[0], 0.25)


def test_superseded_stream_cannot_replace_active_capture_rate(monkeypatch):
    class Stream:
        closed = False

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            self.closed = True

    previous = Stream()
    active = Stream()
    engine = dictation.DictationEngine(whisper_model_ref=lambda: object(), whisper_backend_ref=lambda: 'faster-whisper', mlx_repo_ref=lambda: None, history=Mock())
    engine._recording = True
    engine._session = 1
    active_rate = engine._target_sample_rate

    def open_stream(*args, **kwargs):
        engine._session = 2
        engine._stream = active
        engine._stream_sample_rate = active_rate
        return previous, active_rate * 3, 1

    monkeypatch.setattr(dictation, '_play_beep', lambda *args: None)
    monkeypatch.setattr(dictation, 'resolve_input_device', lambda *args: {'device': 0})
    monkeypatch.setattr(dictation, 'sd', Mock())
    dictation.sd.query_devices.return_value = {'default_samplerate': active_rate}
    monkeypatch.setattr(dictation, 'open_input_stream', open_stream)
    engine._begin_recording(1)

    assert engine._stream is active
    assert engine._stream_sample_rate == active_rate
    assert previous.closed
    assert not active.closed


def test_release_during_stream_start_retains_negotiated_rate(monkeypatch):
    finished = threading.Event()
    captured = []
    engine = dictation.DictationEngine(
        whisper_model_ref=lambda: object(), whisper_backend_ref=lambda: 'faster-whisper',
        mlx_repo_ref=lambda: None, history=Mock(), on_dictation_end=finished.set,
    )
    engine._recording = True
    engine._session = 1
    rate = engine._target_sample_rate * 3

    class Stream:
        closed = False

        def start(self):
            audio = np.full(rate, 0.25, dtype=np.float32)
            engine._audio_callback(engine._session, audio, len(audio), None, None)
            engine._stop_recording()

        def stop(self):
            pass

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setattr(dictation, '_play_beep', lambda *args: None)
    monkeypatch.setattr(dictation, 'resolve_input_device', lambda *args: {'device': 0})
    monkeypatch.setattr(dictation, 'sd', Mock())
    dictation.sd.query_devices.return_value = {'default_samplerate': rate}
    monkeypatch.setattr(dictation, 'open_input_stream', lambda *args, **kwargs: (stream, rate, 1))
    monkeypatch.setattr(engine, '_transcribe', lambda audio: captured.append(audio.copy()) or '')
    engine._begin_recording(1)
    assert finished.wait(5)
    assert stream.closed
    assert len(captured) == 1
    assert len(captured[0]) == engine._target_sample_rate
    np.testing.assert_allclose(captured[0], 0.25)
