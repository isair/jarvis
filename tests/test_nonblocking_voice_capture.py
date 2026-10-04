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
                            captured_tts_start_time=0., generation=obj._dictation_generation)
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
    obj._on_audio(old[:, None], len(old), None, None)
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
        obj._on_audio(old[:, None], len(old), None, None)
        assert reached.wait(1)
        obj._clear_audio_buffers()
        obj._on_audio(fresh[:, None], len(fresh), None, None)
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
    obj._on_audio(np.ones((obj._frame_samples, 1), dtype=np.float32), obj._frame_samples, None, None)
    obj._run_audio_worker(obj.cfg.vad_frame_ms)
    assert obj._should_stop
    assert 'Restart Jarvis to resume listening' in capsys.readouterr().out


def test_pause_after_dequeue_rejects_audio_captured_before_pause(synthetic_listener):
    import queue
    obj = synthetic_listener
    obj._frame_samples = obj.cfg.sample_rate * obj.cfg.vad_frame_ms // 1000
    class PauseOnDequeue(queue.Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                obj.stop()
                raise queue.Empty
            item = super().get(*args, **kwargs)
            obj._dictation_active = True
            obj._dictation_active = False
            return item
    obj._audio_q = PauseOnDequeue()
    detected = []
    obj._is_speech_frame = lambda frame: detected.append(frame.copy()) or True
    obj._on_audio(np.full((obj._frame_samples, 1), .125, dtype=np.float32),
                  obj._frame_samples, None, None)
    obj._consume_audio_frames(obj.cfg.vad_frame_ms)
    assert detected == [], 'Pre-pause audio acquired the post-pause generation'


def test_discarded_reply_stops_thinking_audio(synthetic_listener, monkeypatch):
    obj = synthetic_listener
    playing = [True]
    obj._tune_player = SimpleNamespace(stop_tune=lambda: playing.__setitem__(0, False),
                                      is_playing=lambda: playing[0])
    def reply(*args, **kwargs):
        obj._dictation_active = True
        obj._dictation_active = False
        return 'Stale reply'
    monkeypatch.setattr('jarvis.reply.engine.run_reply_engine', reply)
    obj._dispatch_query('weather')
    assert not playing[0], 'Suppressed reply left thinking audio playing'


@pytest.mark.parametrize('invalidate', ['shutdown', 'dictation'])
def test_pause_during_thinking_tune_teardown_cancels_tts(synthetic_listener, monkeypatch, invalidate):
    obj = synthetic_listener
    spoken = []
    obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: False,
                              speak=lambda text, **kw: spoken.append(text))
    def teardown():
        if invalidate == 'shutdown':
            obj.stop()
        else:
            obj._dictation_active = True
            obj._dictation_active = False
    obj._tune_player = SimpleNamespace(stop_tune=teardown)
    if invalidate == 'shutdown':
        # Simulate shutdown during the blocking player join without recursive teardown.
        obj._tune_player.stop_tune = lambda: setattr(obj, '_should_stop', True)
    monkeypatch.setattr('jarvis.reply.engine.run_reply_engine', lambda *a, **kw: 'Stale reply')
    obj._dispatch_query('weather')
    assert spoken == [], 'Invalidation during tune teardown still queued speech'


def test_callback_copy_cannot_relabel_audio_after_pause(synthetic_listener):
    obj = synthetic_listener
    obj._frame_samples = obj.cfg.sample_rate * obj.cfg.vad_frame_ms // 1000
    class PausingInput:
        def copy(self):
            obj._dictation_active = True
            obj._dictation_active = False
            return np.full((obj._frame_samples, 1), .125, dtype=np.float32)
    obj._on_audio(PausingInput(), obj._frame_samples, None, None)
    samples = []
    obj._is_speech_frame = lambda frame: samples.append(frame.copy()) or True
    class StopWhenEmpty:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def get(self, *args, **kwargs):
            if self.wrapped.empty():
                obj.stop()
            return self.wrapped.get(*args, **kwargs)
    obj._audio_q = StopWhenEmpty(obj._audio_q)
    obj._consume_audio_frames(obj.cfg.vad_frame_ms)
    assert not samples, 'Pre-pause callback samples entered speech detection'


def test_transcript_keeps_capture_generation_across_buffer_storage(synthetic_listener):
    from jarvis.listening.listener import _TranscriptionResult
    from jarvis.listening.intent_judge import IntentJudgment
    obj = synthetic_listener
    def add(**kwargs):
        obj._dictation_active = True
        obj._dictation_active = False
    obj._transcript_buffer.add = add
    obj._intent_judge = SimpleNamespace(available=True, judge=lambda **kw:
        IntentJudgment(directed=True, query='weather', stop=False,
                      confidence='high', reasoning='Addressed to Jarvis'))
    result = _TranscriptionResult('Jarvis weather', 'en', (), 1., 2., .1,
                                   obj._dictation_generation, False, 0.)
    obj._handle_transcription_result(result)
    assert not obj.state_manager.get_pending_query(), 'Old transcript adopted a new voice generation'


@pytest.mark.parametrize('text', [
    "I'm talking to Jairus not you.",
    "I'm talking to Jarvis, not you.",
    "I told my friend about Jarvis yesterday.",
    "Jarvis'le konuşuyorum, seninle değil.",
    "Estoy hablando con Jarvis, no contigo.",
])
def test_confident_mention_rejection_does_not_start_query(synthetic_listener, text):
    from jarvis.listening.intent_judge import IntentJudgment
    obj = synthetic_listener
    obj._intent_judge = SimpleNamespace(available=True, judge=lambda **kwargs:
        IntentJudgment(directed=False, query='', stop=False, confidence='high',
                       reasoning='Assistant mentioned while addressing another person'))
    obj._process_transcript(text, captured_during_tts=False,
                            captured_tts_start_time=0., generation=obj._dictation_generation)
    assert not obj.state_manager.get_pending_query(), 'Mention reached reply collection'


@pytest.mark.parametrize('confidence,directed', [('high', True), ('low', False)])
def test_direct_address_and_inconclusive_fallback_remain_usable(synthetic_listener, confidence, directed):
    from jarvis.listening.intent_judge import IntentJudgment
    obj = synthetic_listener
    obj._intent_judge = SimpleNamespace(available=True, judge=lambda **kwargs:
        IntentJudgment(directed=directed, query='I am tired' if directed else '',
                       stop=False, confidence=confidence, reasoning='Address or uncertainty'))
    obj._process_transcript('Jarvis I am tired', captured_during_tts=False,
                            captured_tts_start_time=0., generation=obj._dictation_generation)
    assert obj.state_manager.get_pending_query(), 'Direct address or fail-open fallback was lost'


@pytest.mark.parametrize('text,accepted', [
    ("Doesn't matter if her oppresses A, first joins", False),
    ('the first person joins the game', False),
    ('Jairus what time is it', True),
    ('Jarvas what time is it', True),
])
def test_whole_alias_gate_keeps_ambient_words_out_of_reply_collection(synthetic_listener, text, accepted):
    from jarvis.listening.intent_judge import IntentJudgment
    obj = synthetic_listener
    obj._intent_judge = SimpleNamespace(available=True, judge=lambda **kwargs:
        IntentJudgment(directed=True, query='what time is it', stop=False,
                       confidence='high', reasoning='Synthetic directed decision'))
    obj._process_transcript(text, captured_during_tts=False,
                            captured_tts_start_time=0., generation=obj._dictation_generation)
    assert bool(obj.state_manager.get_pending_query()) is accepted
