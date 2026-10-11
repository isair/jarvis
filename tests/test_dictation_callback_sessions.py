"""Audio callbacks belong to the recording that opened their stream."""

from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from jarvis.dictation import dictation_engine as dictation

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(dictation, 'parse_hotkey', lambda hotkey: (frozenset(), None))
    monkeypatch.setattr(dictation, '_play_beep', lambda *args: None)
    monkeypatch.setattr(dictation, 'resolve_input_device', lambda *args: {'device': 0})
    monkeypatch.setattr(dictation, 'sd', Mock())
    callbacks = []
    transcribed = []
    finished = threading.Event()

    def open_stream(*args, **kwargs):
        callbacks.append(kwargs['callback'])
        return Mock(), args[1], 1

    monkeypatch.setattr(dictation, 'open_input_stream', open_stream)
    engine = dictation.DictationEngine(lambda: object(), lambda: 'faster-whisper', lambda: None,
                                       history=Mock(), on_dictation_end=finished.set)
    dictation.sd.query_devices.return_value = {'default_samplerate': engine._target_sample_rate}
    monkeypatch.setattr(engine, '_transcribe', lambda audio: transcribed.append(audio.copy()) or '')

    def begin():
        engine._recording = True
        engine._session += 1
        engine._begin_recording(engine._session)
        return callbacks[-1]

    def finish():
        finished.clear()
        engine._stop_recording()
        assert finished.wait(5), 'Released recording did not finish'

    yield engine, begin, finish, transcribed
    engine._stop_recording(discard=True)


@pytest.mark.parametrize('channels', [1, 2])
def test_old_stream_does_not_enter_the_new_recording(capture, channels):
    engine, begin, finish, transcribed = capture
    old_callback = begin()
    engine._stop_recording(discard=True)
    active_callback = begin()
    current = np.full((engine._target_sample_rate // 2, channels), 0.75, dtype=np.float32)
    active_callback(current, len(current), None, None)
    old_callback(np.zeros_like(current), len(current), None, None)

    finish()
    assert len(transcribed) == 1
    assert len(transcribed[0]) == len(current)
    np.testing.assert_allclose(transcribed[0], 0.75)


@pytest.mark.parametrize('next_action', ['release', 'replacement'])
def test_session_change_during_audio_copy_discards_the_old_frame(capture, monkeypatch, next_action):
    engine, begin, finish, transcribed = capture
    callback = begin()
    copying = threading.Event()
    release_copy = threading.Event()
    finalising = threading.Event()
    release_finalise = threading.Event()
    completed = threading.Event()
    original_mono = dictation.mono_capture
    original_finalise = engine._finalise_and_transcribe

    def slow_copy(indata):
        copying.set()
        assert release_copy.wait(5), 'Test did not release audio conversion'
        return original_mono(indata)

    def delayed_finalise(*args):
        finalising.set()
        assert release_finalise.wait(5), 'Test did not release finalisation'
        try:
            original_finalise(*args)
        finally:
            completed.set()

    monkeypatch.setattr(dictation, 'mono_capture', slow_copy)
    monkeypatch.setattr(engine, '_finalise_and_transcribe', delayed_finalise)
    with ThreadPoolExecutor(max_workers=1) as pool:
        audio = np.zeros(engine._target_sample_rate // 2, dtype=np.float32)
        result = pool.submit(callback, audio, len(audio), None, None)
        try:
            assert copying.wait(5)
            engine._stop_recording()
            assert finalising.wait(5)
            if next_action == 'replacement':
                begin()
            release_copy.set()
            result.result(timeout=5)
        finally:
            release_copy.set()
            release_finalise.set()
        assert completed.wait(5)

    assert transcribed == [], 'Released snapshots must not acquire late audio'


def test_superseded_start_worker_preserves_new_session_audio(capture, monkeypatch):
    engine, begin, finish, transcribed = capture
    replaced = False

    def log(message, *args):
        nonlocal replaced
        if message == 'dictation recording started' and not replaced:
            replaced = True
            engine._stop_recording(discard=True)
            active_callback = begin()
            audio = np.full(engine._target_sample_rate // 2, 0.75, dtype=np.float32)
            active_callback(audio, len(audio), None, None)

    monkeypatch.setattr(dictation, 'debug_log', log)
    begin()
    finish()
    assert len(transcribed) == 1
    np.testing.assert_allclose(transcribed[0], 0.75)


@pytest.mark.parametrize('phase', ['teardown', 'transcription'])
@pytest.mark.parametrize('failure', [False, True])
def test_old_completion_preserves_new_recording_presence(capture, monkeypatch, phase, failure):
    engine, begin, _finish, _transcribed = capture
    state = []
    engine._on_dictation_start = lambda: state.append('recording')
    engine._on_dictation_processing_start = lambda: state.append('processing')
    engine._on_dictation_end = lambda: state.append('idle')
    blocked, release, completed = threading.Event(), threading.Event(), threading.Event()
    pasted = []
    monkeypatch.setattr(dictation, '_clipboard_paste', pasted.append)
    old_callback = begin()
    audio = np.full(engine._target_sample_rate // 2, 0.25, dtype=np.float32)
    old_callback(audio, len(audio), None, None)
    old_stream = engine._stream
    close = dictation._close_stream
    finalise = engine._finalise_and_transcribe

    def wait():
        blocked.set()
        assert release.wait(5), 'Test did not release the old recording'
        if failure:
            raise RuntimeError('Synthetic dictation failure')

    def close_stream(stream):
        if phase == 'teardown' and stream is old_stream:
            wait()
        close(stream)

    def transcribe(audio):
        if phase == 'transcription':
            wait()
        return 'old words'

    def finalise_and_signal(*args):
        try:
            finalise(*args)
        finally:
            completed.set()

    monkeypatch.setattr(dictation, '_close_stream', close_stream)
    monkeypatch.setattr(engine, '_transcribe', transcribe)
    monkeypatch.setattr(engine, '_finalise_and_transcribe', finalise_and_signal)
    try:
        engine._stop_recording()
        assert blocked.wait(5), 'Old recording did not reach the controlled boundary'
        begin()
        assert state[-1] == 'recording'
    finally:
        release.set()
        assert completed.wait(5), 'Old recording did not finish'
    assert state[-1] == 'recording', state
    assert pasted == ([] if failure else ['old words'])
    engine._stop_recording(discard=True)
    assert state[-1] == 'idle'


def test_unready_replacement_releases_the_previous_capture_pause(capture, monkeypatch):
    engine, begin, _finish, _transcribed = capture
    state = []
    engine._on_dictation_start = lambda: state.append('recording')
    engine._on_dictation_end = lambda: state.append('idle')
    begin()
    # A replacement can be reserved while the previous release is processing.
    engine._recording = False
    engine._session += 1
    monkeypatch.setattr(engine, '_whisper_model_ref', lambda: None)
    engine._recording = True
    engine._begin_recording(engine._session)
    assert state[-1] == 'idle', state


def test_released_start_worker_cannot_pause_capture(capture, monkeypatch):
    engine, begin, _finish, _transcribed = capture
    state = []
    engine._on_dictation_start = lambda: state.append('recording')
    engine._on_dictation_end = lambda: state.append('idle')

    def release_at_notification(message, *args):
        if message == 'dictation recording started':
            engine._stop_recording(discard=True)

    monkeypatch.setattr(dictation, 'debug_log', release_at_notification)
    begin()
    assert state == ['idle']
