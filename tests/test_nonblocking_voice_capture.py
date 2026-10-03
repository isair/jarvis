"""Microphone frames keep reaching speech detection during slow language work."""
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.config import load_settings
from jarvis.listening.listener import VoiceListener
import jarvis.listening.listener as capture

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_listener(monkeypatch):
    cfg = replace(load_settings(), voice_debug=False, vad_enabled=False,
                  whisper_backend='faster-whisper', whisper_device='cpu')
    monkeypatch.setattr(capture, 'create_intent_judge', lambda _cfg: None)
    monkeypatch.setattr(capture, 'FASTER_WHISPER_AVAILABLE', True)
    monkeypatch.setattr(capture, '_load_faster_whisper_model', lambda *a, **kw: MagicMock())
    monkeypatch.setattr(VoiceListener, '_decode_faster_whisper', lambda *a: ([], None))
    monkeypatch.setattr(VoiceListener, '_start_llm_warmup', lambda _self: [])
    monkeypatch.setattr('desktop_app.face_widget.get_jarvis_state', lambda: MagicMock())
    obj = VoiceListener(MagicMock(), cfg, None, None)

    class Stream:
        active = True
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    def devices(device=None, **kwargs):
        info = {'index': 0, 'name': 'Synthetic input', 'max_input_channels': 1}
        return info if device is not None or kwargs else [info]
    monkeypatch.setattr(capture, 'sd', SimpleNamespace(
        query_devices=devices, InputStream=lambda **kw: Stream(),
    ))
    yield obj
    obj.stop()


@pytest.mark.parametrize('blocked_stage', ['intent', 'reply'])
def test_audio_reaches_vad_while_language_processing_is_blocked(synthetic_listener, blocked_stage):
    obj = synthetic_listener
    entered = threading.Event()
    release = threading.Event()
    processed = threading.Event()
    errors = []

    def block(*args):
        entered.set()
        if not release.wait(3):
            raise AssertionError('Synthetic language work was not released')

    if blocked_stage == 'intent':
        obj._transcription_results_q.put(object())
        obj._handle_transcription_result = block
        obj._check_query_timeout = lambda: None
    else:
        obj._handle_transcription_result = lambda result: None
        obj._check_query_timeout = block
    def detect(frame):
        processed.set()
        return False
    obj._is_speech_frame = detect
    def run():
        try:
            obj.run()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2), 'Synthetic listener did not reach the slow language stage'
        frames = obj.cfg.sample_rate * obj.cfg.vad_frame_ms // 1000
        obj._on_audio(np.ones((frames, 1), dtype=np.float32) * .1, frames, None, None)
        assert processed.wait(.5), 'Audio stalled behind intent judging or reply generation'
    finally:
        obj.stop()
        release.set()
        thread.join(2)
    assert not thread.is_alive(), 'Synthetic listener did not stop'
    assert not errors


@pytest.mark.parametrize('invalidate', ['shutdown', 'dictation'])
def test_full_transcript_queue_cancels_pending_delivery(synthetic_listener, invalidate):
    import queue
    obj = synthetic_listener
    delivering = threading.Event()
    class ResultQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            if self.full():
                delivering.set()
            return super().put(item, *args, **kwargs)
    obj._transcription_results_q = ResultQueue(maxsize=1)
    obj._transcription_results_q.put(object())
    def transcribe(audio):
        return 'Jarvis hello', 'en', ()
    obj._transcribe_audio = transcribe
    job = SimpleNamespace(audio=np.ones(obj.cfg.sample_rate), start_time=1., end_time=2.,
                          energy=.1, dictation_generation=0, captured_during_tts=False,
                          captured_tts_start_time=0.)
    obj._transcription_jobs_q.put(job)
    obj._transcription_jobs_q.put(None)
    worker = threading.Thread(target=obj._run_transcription_worker)
    worker.start()
    try:
        assert delivering.wait(1)
        if invalidate == 'shutdown':
            obj.stop()
        else:
            obj._dictation_active = True
            obj._dictation_active = False
        worker.join(.4)
        assert not worker.is_alive(), 'Invalidated transcript stayed blocked behind a full result queue'
    finally:
        while not obj._transcription_results_q.empty():
            obj._transcription_results_q.get_nowait()
        worker.join(1)
    assert not worker.is_alive()


@pytest.mark.parametrize('invalidate', ['shutdown', 'dictation'])
def test_invalidated_intent_result_does_not_start_query(synthetic_listener, invalidate):
    from jarvis.listening.intent_judge import IntentJudgment
    obj = synthetic_listener
    def judge(**kwargs):
        if invalidate == 'shutdown':
            obj.stop()
        else:
            obj._dictation_active = True
            obj._dictation_active = False
        return IntentJudgment(directed=True, query='weather', stop=False,
                              confidence='high', reasoning='Addressed to Jarvis')
    obj._intent_judge = SimpleNamespace(available=True, judge=judge)
    obj._process_transcript('Jarvis weather', captured_during_tts=False,
                            captured_tts_start_time=0.)
    assert not obj.state_manager.get_pending_query(), 'Invalidated intent was accepted as a new query'


@pytest.mark.parametrize('invalidate', ['shutdown', 'dictation'])
@pytest.mark.parametrize('reply_error', [False, True])
def test_invalidated_reply_does_not_speak(synthetic_listener, monkeypatch, invalidate, reply_error):
    obj = synthetic_listener
    spoken = []
    obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: False,
                              speak=lambda text, **kwargs: spoken.append(text))
    def reply(*args, **kwargs):
        if invalidate == 'shutdown':
            obj.stop()
        else:
            obj._dictation_active = True
            obj._dictation_active = False
        if reply_error:
            raise RuntimeError('Synthetic reply failure')
        return 'Synthetic reply'
    monkeypatch.setattr('jarvis.reply.engine.run_reply_engine', reply)
    obj._dispatch_query('weather')
    assert spoken == [], 'Invalidated voice reply reached speech output'


def test_brief_dictation_pause_discards_captured_audio_without_waiting_for_worker(synthetic_listener):
    obj = synthetic_listener
    old = np.full(obj.cfg.sample_rate, .1, dtype=np.float32)
    fresh = np.full(obj.cfg.sample_rate, .2, dtype=np.float32)
    obj._utterance_frames = [old]
    obj.is_speech_active = True
    obj._audio_q.put(old[:, None])
    obj._dictation_active = True
    obj._dictation_active = False
    decoded = []
    def transcribe(audio):
        decoded.append(audio.copy())
        return '', None, ()
    obj._transcribe_audio = transcribe
    obj._utterance_frames.append(fresh)
    obj._finalize_utterance()
    obj._transcription_jobs_q.put(None)
    obj._run_transcription_worker()
    assert len(decoded) == 1
    np.testing.assert_array_equal(decoded[0], fresh)
    assert obj._audio_q.empty(), 'Audio captured before dictation remained queued'


@pytest.mark.parametrize('blocked_stage', ['intent', 'reply'])
def test_completed_audio_is_decoded_fifo_during_language_work(synthetic_listener, blocked_stage):
    obj = synthetic_listener
    obj.cfg = replace(obj.cfg, max_utterance_ms=400, whisper_min_audio_duration=.1)
    entered, release, decoded = threading.Event(), threading.Event(), threading.Event()
    samples = []
    def block(*args):
        entered.set()
        assert release.wait(3)
    if blocked_stage == 'intent':
        obj._transcription_results_q.put(object())
        obj._handle_transcription_result = block
        obj._check_query_timeout = lambda: None
    else:
        obj._handle_transcription_result = lambda result: None
        obj._check_query_timeout = block
    obj._is_speech_frame = lambda frame: True
    def transcribe(audio):
        samples.append(audio.copy())
        if len(samples) == 2:
            decoded.set()
        return '', None, ()
    obj._transcribe_audio = transcribe
    thread = threading.Thread(target=obj.run)
    thread.start()
    try:
        assert entered.wait(2)
        size = obj.cfg.sample_rate * obj.cfg.max_utterance_ms // 1000
        for value in (.1, .2):
            obj._on_audio(np.full((size, 1), value, dtype=np.float32), size, None, None)
        assert decoded.wait(1), 'Completed speech waited for language work'
        assert len(samples) == 2
        for audio, value in zip(samples, (.1, .2)):
            np.testing.assert_array_equal(audio, np.full(size, value, dtype=np.float32))
    finally:
        obj.stop()
        release.set()
        thread.join(2)
    assert not thread.is_alive()


def test_reset_discards_remaining_frames_in_an_already_dequeued_batch(synthetic_listener):
    obj = synthetic_listener
    obj.cfg = replace(obj.cfg, max_utterance_ms=40, whisper_min_audio_duration=.01)
    obj._frame_samples = obj.cfg.sample_rate * obj.cfg.vad_frame_ms // 1000
    reached, resume, decoded = threading.Event(), threading.Event(), threading.Event()
    old = np.full(obj._frame_samples, .1, dtype=np.float32)
    fresh = np.full(obj._frame_samples, .2, dtype=np.float32)
    first_batch = True
    def frames(buf):
        nonlocal first_batch
        if first_batch:
            first_batch = False
            yield old
            reached.set()
            assert resume.wait(2)
            yield old
        else:
            yield fresh
            yield fresh
    obj._audio_frames = frames
    obj._is_speech_frame = lambda frame: True
    samples = []
    def transcribe(audio):
        samples.append(audio.copy())
        decoded.set()
        return '', None, ()
    obj._transcribe_audio = transcribe
    obj._start_transcription_worker()
    obj._start_audio_worker(obj.cfg.vad_frame_ms)
    try:
        obj._audio_q.put(old[:, None])
        assert reached.wait(1)
        obj._clear_audio_buffers()
        obj._audio_q.put(fresh[:, None])
        resume.set()
        assert decoded.wait(1)
        assert len(samples) == 1
        np.testing.assert_array_equal(samples[0], np.concatenate([fresh, fresh]))
    finally:
        resume.set()
        obj.stop()
        obj._finish_audio_worker()
        obj._finish_transcription_worker()


def test_failed_frame_processing_stops_listener_and_reports_recovery(synthetic_listener, capsys):
    obj = synthetic_listener
    def fail(frame):
        raise RuntimeError('Synthetic VAD failure')
    obj._is_speech_frame = fail
    obj._frame_samples = obj.cfg.sample_rate * obj.cfg.vad_frame_ms // 1000
    obj._audio_q.put(np.ones((obj._frame_samples, 1), dtype=np.float32))
    obj._run_audio_worker(obj.cfg.vad_frame_ms)
    assert obj._should_stop
    assert 'Restart Jarvis to resume listening' in capsys.readouterr().out
