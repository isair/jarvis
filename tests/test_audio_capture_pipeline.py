"""Native-rate microphone frames reach VAD without silent rejection or loss."""

from collections import deque
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener
import jarvis.listening.listener as capture
import jarvis.utils.audio_capture as audio_capture

pytestmark = pytest.mark.unit


def listener(rate=48000):
    obj = VoiceListener.__new__(VoiceListener)
    obj.cfg = SimpleNamespace(voice_min_energy=0.0045, voice_debug=False)
    obj._samplerate = 16000
    obj._stream_samplerate = rate
    obj._frame_samples = rate * 20 // 1000
    obj._pending_audio = None
    obj._recent_audio_energy = deque(maxlen=20)
    obj._vad_error_logged = False
    obj._vad = None
    obj._should_stop = False
    obj._capture_pause_reasons = frozenset()
    obj._capture_pause_reset_pending = False
    obj._capture_generation = 0
    obj._transcription_worker_thread = None
    import threading
    obj._audio_worker_thread = None
    obj._audio_state_lock = threading.RLock()
    obj._audio_generation = 0
    obj._pre_roll = deque()
    obj._slow_transcription_streak = 0
    obj._slow_transcription_warned = False
    obj._whisper_model_name = "small"
    obj._callback_count = 0
    obj._audio_q = queue.Queue(maxsize=2)
    obj._reset_audio_health(now=0)
    return obj


@pytest.mark.parametrize('rate', [16000, 44100, 48000])
def test_native_frames_use_supported_vad_format(rate):
    obj = listener(rate)

    class StrictVad:
        def is_speech(self, pcm, sample_rate):
            assert sample_rate == 16000
            assert len(pcm) == 640  # 20 ms, mono int16 at 16 kHz
            return True

    obj._vad = StrictVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples, dtype=np.float32) * .1)


def test_partial_callback_frames_are_not_discarded():
    obj = listener(44100)
    audio = np.linspace(-.1, .1, obj._frame_samples * 3 + 17, dtype=np.float32)
    output = []
    for block in np.array_split(audio, 13):
        output.extend(obj._audio_frames(block[:, None]))
    np.testing.assert_array_equal(np.concatenate(output), audio[:-17])
    np.testing.assert_array_equal(obj._pending_audio, audio[-17:])


def test_vad_failure_warns_once_and_uses_energy_gate(capsys):
    obj = listener()
    class BrokenVad:
        def is_speech(self, *args):
            raise ValueError('invalid frame')
    obj._vad = BrokenVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples) * .1)
    assert not obj._is_speech_frame(np.zeros(obj._frame_samples))
    assert capsys.readouterr().out.count('Speech detection failed') == 1


def test_capture_health_distinguishes_missing_callbacks_and_silent_samples(capsys):
    obj = listener()
    obj._check_audio_health(now=6)
    assert 'No microphone callbacks' in capsys.readouterr().out
    obj._callback_count = 1
    obj._last_audio_callback = 11
    obj._audio_frames(np.zeros((obj._frame_samples, 1), dtype=np.float32))
    obj._check_audio_health(now=12)
    assert 'silent samples' in capsys.readouterr().out


def test_callback_status_and_queue_overflow_are_visible(capsys):
    obj = listener()
    for _ in range(3):
        obj._on_audio(np.ones((960, 1), dtype=np.float32), 960, None, 'input overflow')
    obj._check_audio_health(now=6)
    output = capsys.readouterr().out
    assert 'input overflow' in output
    assert '1' in output and 'dropped' in output


def test_dictation_pause_does_not_report_capture_failure(capsys):
    obj = listener()
    obj.set_capture_paused('dictation', True)
    obj._check_audio_health(now=30)
    assert not capsys.readouterr().out


def test_stalled_capture_warns_once_then_reports_recovery(capsys):
    obj = listener()
    obj._last_audio_callback = 2
    obj._check_audio_health(now=8)
    obj._check_audio_health(now=14)
    assert capsys.readouterr().out.count('No microphone callbacks') == 1
    obj._last_audio_callback = 19
    obj._audio_frames(np.ones((obj._frame_samples, 1), dtype=np.float32) * .1)
    obj._check_audio_health(now=20)
    assert 'arriving again' in capsys.readouterr().out


def test_callback_exception_is_not_silenced(capsys):
    obj = listener()
    class BrokenInput:
        def copy(self):
            raise RuntimeError('capture buffer failed')
    obj._on_audio(BrokenInput(), 960, None, None)
    obj._check_audio_health(now=6)
    assert 'capture buffer failed' in capsys.readouterr().out


@pytest.mark.parametrize('rate,channels', [(16000, 2), (48000, 2), (44100, 4), (48000, 1)])
def test_capture_negotiates_format_on_selected_device(monkeypatch, rate, channels):
    opened = []
    stream = object()

    def open_stream(**kwargs):
        assert kwargs['device'] == 7
        opened.append((kwargs['samplerate'], kwargs['channels']))
        if kwargs['channels'] != channels:
            raise RuntimeError('Invalid number of channels', -9998)
        if kwargs['samplerate'] != rate:
            raise RuntimeError('Invalid sample rate', -9997)
        assert kwargs['blocksize'] == rate * 20 // 1000
        return stream

    monkeypatch.setattr(capture.sd, 'InputStream', open_stream)
    def device_info(device=None, **kwargs):
        assert device == 7
        return {'max_input_channels': channels, 'default_samplerate': rate}
    monkeypatch.setattr(capture.sd, 'query_devices', device_info)
    result, actual_rate, actual_channels = audio_capture.open_input_stream(
        capture.sd, 16000, 20, {'device': 7}, callback=lambda *args: None,
    )
    assert result is stream
    assert (actual_rate, actual_channels) == (rate, channels)
    assert len(opened) == len(set(opened)) <= 6


@pytest.mark.parametrize('failure', ['Microphone access denied', 'Device unavailable', 'Device busy'])
def test_capture_does_not_retry_non_format_errors(monkeypatch, failure):
    def open_stream(**kwargs):
        raise RuntimeError(failure)
    def unexpected_query(*args, **kwargs):
        pytest.fail('must not negotiate a permission or device availability failure')
    monkeypatch.setattr(capture.sd, 'InputStream', open_stream)
    monkeypatch.setattr(capture.sd, 'query_devices', unexpected_query)
    with pytest.raises(RuntimeError, match=failure):
        audio_capture.open_input_stream(capture.sd, 16000, 20, {})


def test_multichannel_capture_preserves_signal_outside_first_channel():
    obj = listener(48000)
    source = np.zeros((obj._frame_samples * 2 + 13, 2), dtype=np.float32)
    source[:, 1] = np.linspace(-.5, .5, len(source))
    frames = []
    for block in np.array_split(source, 7):
        frames.extend(obj._audio_frames(block))
    output = np.concatenate(frames + [obj._pending_audio])
    np.testing.assert_allclose(output, source.mean(axis=1))
    assert np.max(np.abs(output)) > 0


def test_unsupported_device_exhausts_bounded_formats(monkeypatch):
    attempts = []
    def reject(**kwargs):
        attempts.append((kwargs['samplerate'], kwargs['channels']))
        raise RuntimeError('Invalid number of channels', -9998)
    monkeypatch.setattr(capture.sd, 'InputStream', reject)
    def default_input(**kwargs):
        assert kwargs == {'kind': 'input'}
        return {'max_input_channels': 4, 'default_samplerate': 48000}
    monkeypatch.setattr(capture.sd, 'query_devices', default_input)
    with pytest.raises(RuntimeError, match='Invalid number of channels'):
        audio_capture.open_input_stream(capture.sd, 16000, 20, {})
    assert len(attempts) == len(set(attempts)) == 6


def test_access_failure_during_negotiation_stops_retries(monkeypatch):
    attempts = []
    def reject(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError('Invalid number of channels', -9998)
        if len(attempts) > 2:
            pytest.fail('must stop after an access failure')
        raise RuntimeError('Access denied')
    monkeypatch.setattr(capture.sd, 'InputStream', reject)
    monkeypatch.setattr(capture.sd, 'query_devices', lambda **kwargs: {
        'max_input_channels': 2, 'default_samplerate': 48000,
    })
    with pytest.raises(RuntimeError, match='Access denied'):
        audio_capture.open_input_stream(capture.sd, 16000, 20, {})


@pytest.mark.parametrize('serialise', [False, True])
def test_stream_open_lock_policy_includes_retries(monkeypatch, serialise):
    locked = []
    class Lock:
        def __enter__(self):
            locked.append(True)
        def __exit__(self, *args):
            locked.pop()
    monkeypatch.setattr(audio_capture, 'portaudio_lock', Lock())
    def open_stream(**kwargs):
        assert bool(locked) == serialise
        if kwargs['channels'] == 1:
            raise RuntimeError('Invalid number of channels', -9998)
        return 'stream'
    monkeypatch.setattr(capture.sd, 'InputStream', open_stream)
    monkeypatch.setattr(capture.sd, 'query_devices', lambda **kwargs: {
        'max_input_channels': 2, 'default_samplerate': 16000,
    })
    assert audio_capture.open_input_stream(capture.sd, 16000, 20, {}, serialise=serialise)[0] == 'stream'
    assert not locked


def test_missing_named_input_does_not_record_another_microphone(monkeypatch, capsys):
    obj = listener()
    obj.cfg.voice_device = 'Disconnected Headset'
    monkeypatch.setattr(capture.sd, 'query_devices', lambda: [
        {'name': 'Built-in Microphone', 'max_input_channels': 1},
    ])
    monkeypatch.setattr(capture.sd, 'InputStream', lambda **kwargs: pytest.fail('unexpected capture'))
    obj.run()
    assert 'Selected microphone not found' in capsys.readouterr().out


def test_capture_reports_configured_format_error_after_all_retries(monkeypatch):
    def reject(**kwargs):
        raise RuntimeError(f"Unsupported {kwargs['samplerate']} Hz / {kwargs['channels']} channels", -9998)

    monkeypatch.setattr(capture.sd, 'InputStream', reject)
    monkeypatch.setattr(capture.sd, 'query_devices', lambda **kwargs: {
        'max_input_channels': 2, 'default_samplerate': 48000,
    })

    with pytest.raises(RuntimeError, match='Unsupported 16000 Hz / 1 channels'):
        audio_capture.open_input_stream(capture.sd, 16000, 20, {})


def test_capture_keeps_original_error_when_device_query_fails(monkeypatch):
    original = RuntimeError('Invalid number of channels', -9998)
    monkeypatch.setattr(capture.sd, 'InputStream', lambda **kwargs: (_ for _ in ()).throw(original))
    monkeypatch.setattr(capture.sd, 'query_devices', lambda **kwargs: (_ for _ in ()).throw(OSError('device disconnected')))

    with pytest.raises(RuntimeError) as raised:
        audio_capture.open_input_stream(capture.sd, 16000, 20, {})
    assert raised.value is original


def test_named_input_skips_devices_with_missing_names(monkeypatch):
    monkeypatch.setattr(capture.sd, 'query_devices', lambda: [
        {'name': None, 'max_input_channels': 1},
        {'name': 'Headset microphone', 'max_input_channels': 1},
    ])
    assert audio_capture.resolve_input_device(capture.sd, 'Headset') == {'device': 1}


def test_numeric_input_index_zero_is_selected():
    assert audio_capture.resolve_input_device(capture.sd, 0) == {'device': 0}


def test_utterance_finalisation_queues_transcription_without_blocking():
    from types import SimpleNamespace

    obj = listener()
    obj._utterance_frames = [np.ones(8000, dtype=np.float32)]
    obj._samplerate = obj._stream_samplerate = 16000
    obj._transcription_jobs_q = queue.Queue(maxsize=8)
    obj._transcription_results_q = queue.Queue()
    obj.is_speech_active = True
    obj._silence_frames = 0
    obj.echo_detector = SimpleNamespace(_utterance_start_time=1.0, _tts_start_time=0,
                                        _last_tts_finish_time=0, echo_tolerance=0.3)
    obj.tts = None
    obj._capture_generation = 0
    obj.cfg = SimpleNamespace(voice_debug=False, whisper_min_audio_duration=0.3)
    obj._calculate_audio_energy = lambda frames: 0.1

    obj._finalize_utterance()

    assert not obj.is_speech_active
    assert obj._utterance_frames == []
    assert obj._transcription_jobs_q.qsize() == 1
    assert obj._transcription_results_q.empty()
    job = obj._transcription_jobs_q.get_nowait()
    assert job.audio.shape == (8000,)
    assert job.start_time == 1.0
    assert job.energy == 0.1


def test_transcription_worker_preserves_utterance_order():
    import threading

    obj = listener()
    obj._transcription_jobs_q = queue.Queue()
    obj._transcription_results_q = queue.Queue()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    started = []

    def transcribe(audio):
        started.append(audio)
        if audio == "first":
            first_started.set()
            release_first.wait(timeout=2)
        else:
            second_started.set()
        return str(audio), "en", ()

    obj._transcribe_audio = transcribe
    obj._transcription_jobs_q.put(SimpleNamespace(audio="first", start_time=1.0, end_time=2.0, energy=0.1, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))
    obj._transcription_jobs_q.put(SimpleNamespace(audio="second", start_time=2.0, end_time=3.0, energy=0.2, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))
    obj._transcription_jobs_q.put(None)

    worker = threading.Thread(target=obj._run_transcription_worker)
    worker.start()
    try:
        assert first_started.wait(timeout=1)
        assert not second_started.wait(timeout=0.05)
    finally:
        release_first.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert started == ["first", "second"]
    results = [obj._transcription_results_q.get_nowait() for _ in range(2)]
    assert [result.text for result in results] == ["first", "second"]


def test_utterance_finalisation_continues_while_whisper_is_busy():
    import threading
    from types import SimpleNamespace

    obj = listener()
    obj._transcription_jobs_q = queue.Queue(maxsize=8)
    obj._transcription_results_q = queue.Queue()
    whisper_started = threading.Event()
    release_whisper = threading.Event()

    def transcribe(audio):
        if isinstance(audio, str):
            whisper_started.set()
            release_whisper.wait(timeout=2)
        return "recognised", "en", ()

    obj._transcribe_audio = transcribe
    obj._transcription_jobs_q.put(
        SimpleNamespace(audio="slow", start_time=1.0, end_time=2.0, energy=0.1, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0)
    )
    worker = threading.Thread(target=obj._run_transcription_worker)
    worker.start()

    try:
        assert whisper_started.wait(timeout=1)
        obj._utterance_frames = [np.ones(8000, dtype=np.float32)]
        obj._samplerate = obj._stream_samplerate = 16000
        obj.is_speech_active = True
        obj._silence_frames = 0
        obj.echo_detector = SimpleNamespace(_utterance_start_time=1.0, _tts_start_time=0,
                                            _last_tts_finish_time=0, echo_tolerance=0.3)
        obj.cfg = SimpleNamespace(voice_debug=False, whisper_min_audio_duration=0.3)
        obj.tts = None
        obj._calculate_audio_energy = lambda frames: 0.1

        obj._finalize_utterance()

        assert obj._transcription_jobs_q.qsize() == 1
        assert worker.is_alive()
    finally:
        release_whisper.set()
        obj._transcription_jobs_q.put(None)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert [obj._transcription_results_q.get_nowait().text for _ in range(2)] == [
        "recognised",
        "recognised",
    ]


def test_transcription_result_keeps_language_and_utterance_context():
    from types import SimpleNamespace
    from unittest.mock import Mock

    obj = listener()
    obj._last_detected_language = None
    obj._first_utterance = True
    obj.tts = None
    obj.echo_detector = SimpleNamespace(_tts_start_time=0, _last_tts_finish_time=0, echo_tolerance=0.3)
    obj._transcript_buffer = Mock()
    obj._process_transcript = Mock()
    obj._is_repetitive_hallucination = lambda text: False
    result = SimpleNamespace(
        text="hello there",
        language="en",
        low_confidence_events=(),
        start_time=10.0,
        end_time=11.0,
        energy=0.2,
        capture_generation=0,
        captured_during_tts=False,
        captured_tts_start_time=0,
    )

    obj._handle_transcription_result(result)

    assert obj._last_detected_language == "en"
    obj._transcript_buffer.add.assert_called_once_with(
        text="hello there",
        start_time=10.0,
        end_time=11.0,
        energy=0.2,
        is_during_tts=False,
    )
    obj._process_transcript.assert_called_once_with(
        "hello there", 0.2, 10.0, 11.0,
        generation=result.capture_generation,
        captured_during_tts=False, captured_tts_start_time=0,
    )


def test_shutdown_discards_full_backlog_without_waiting_for_whisper():
    import threading
    from unittest.mock import Mock

    obj = listener()
    obj._transcription_jobs_q = queue.Queue(maxsize=1)
    obj._transcription_results_q = queue.Queue()
    obj._handle_transcription_result = Mock()
    started = threading.Event()
    release = threading.Event()

    def transcribe(_audio):
        started.set()
        release.wait(timeout=3)
        return "late answer", "en", ()

    obj._transcribe_audio = transcribe
    job = SimpleNamespace(audio="active", start_time=1.0, end_time=2.0, energy=0.1, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0)
    obj._transcription_jobs_q.put(job)
    obj._start_transcription_worker()
    assert started.wait(timeout=1)
    obj._transcription_jobs_q.put(SimpleNamespace(audio="queued", start_time=2.0, end_time=3.0, energy=0.1, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))
    obj._should_stop = True
    finished = threading.Event()

    def finish():
        obj._finish_transcription_worker()
        finished.set()

    shutdown = threading.Thread(target=finish, daemon=True)
    shutdown.start()
    try:
        assert finished.wait(timeout=1), "shutdown blocked on queued or active Whisper work"
        assert obj._transcription_jobs_q.qsize() <= 1
        obj._handle_transcription_result.assert_not_called()
    finally:
        release.set()
        shutdown.join(timeout=3)


def test_transcription_result_during_dictation_is_not_dispatched():
    from unittest.mock import Mock

    obj = listener()
    obj.set_capture_paused('dictation', True)
    obj._last_detected_language = "fr"
    obj._first_utterance = True
    obj.tts = None
    obj.echo_detector = SimpleNamespace(_tts_start_time=0, _last_tts_finish_time=0, echo_tolerance=0.3)
    obj._transcript_buffer = Mock()
    obj._process_transcript = Mock()
    obj._is_repetitive_hallucination = lambda text: False
    result = SimpleNamespace(text="jarvis do this", language="en", low_confidence_events=(), start_time=10.0, end_time=11.0, energy=0.2, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0)

    obj._handle_transcription_result(result)

    assert obj._last_detected_language == "fr"
    obj._transcript_buffer.add.assert_not_called()
    obj._process_transcript.assert_not_called()


def test_transcription_result_without_detected_language_clears_previous_locale():
    from unittest.mock import Mock

    obj = listener()
    obj._last_detected_language = "fr"
    obj._first_utterance = True
    obj.tts = None
    obj.echo_detector = SimpleNamespace(_tts_start_time=0, _last_tts_finish_time=0, echo_tolerance=0.3)
    obj._transcript_buffer = Mock()
    obj._process_transcript = Mock()
    obj._is_repetitive_hallucination = lambda text: False

    obj._handle_transcription_result(SimpleNamespace(text="hello", language=None, low_confidence_events=(), start_time=10.0, end_time=11.0, energy=0.2, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))

    assert obj._last_detected_language is None


def test_echo_flag_uses_capture_interval_when_tts_starts_during_whisper():
    from unittest.mock import Mock

    obj = listener()
    obj._last_detected_language = None
    obj._first_utterance = True
    obj.tts = SimpleNamespace(is_speaking=lambda: True)
    obj.echo_detector = SimpleNamespace(_tts_start_time=12.0, _last_tts_finish_time=0, echo_tolerance=0.3)
    obj._transcript_buffer = Mock()
    obj._process_transcript = Mock()
    obj._is_repetitive_hallucination = lambda text: False

    obj._handle_transcription_result(SimpleNamespace(text="jarvis hello", language="en", low_confidence_events=(), start_time=10.0, end_time=11.0, energy=0.2, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))

    assert obj._transcript_buffer.add.call_args.kwargs["is_during_tts"] is False


def test_queued_job_keeps_tts_context_from_audio_capture():
    from unittest.mock import Mock

    obj = listener()
    obj._utterance_frames = [np.ones(8000, dtype=np.float32)]
    obj._samplerate = obj._stream_samplerate = 16000
    obj._transcription_jobs_q = queue.Queue(maxsize=8)
    obj._transcription_results_q = queue.Queue()
    obj.is_speech_active = True
    obj._silence_frames = 0
    obj.echo_detector = SimpleNamespace(_utterance_start_time=10.0, _tts_start_time=0,
                                        _last_tts_finish_time=0, echo_tolerance=0.3)
    obj.tts = SimpleNamespace(is_speaking=Mock(return_value=False))
    obj.cfg = SimpleNamespace(voice_debug=False, whisper_min_audio_duration=0.3)
    obj._calculate_audio_energy = lambda frames: 0.1

    obj._finalize_utterance()
    job = obj._transcription_jobs_q.get_nowait()
    obj.echo_detector._tts_start_time = 12.0
    obj.tts.is_speaking.return_value = True
    obj._transcribe_audio = lambda audio: ("stop", "en", ())
    obj._transcription_jobs_q.put(job)
    obj._transcription_jobs_q.put(None)
    obj._run_transcription_worker()
    result = obj._transcription_results_q.get_nowait()

    assert result.captured_during_tts is False
    assert result.captured_tts_start_time == 0


def test_run_cleans_up_worker_after_listener_exception():
    import threading
    from unittest.mock import Mock

    obj = listener()
    obj._transcription_jobs_q = queue.Queue(maxsize=1)
    obj._transcription_results_q = queue.Queue()
    obj._handle_transcription_result = Mock()
    obj._transcribe_audio = lambda audio: ("hello", "en", ())
    obj._start_transcription_worker()
    worker = obj._transcription_worker_thread
    obj._run = lambda: (_ for _ in ()).throw(RuntimeError("capture failed"))

    with pytest.raises(RuntimeError, match="capture failed"):
        obj.run()

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert obj._should_stop


def test_dictation_session_invalidates_decode_even_after_listener_resumes():
    import threading
    from unittest.mock import Mock

    obj = listener()
    obj._transcription_jobs_q = queue.Queue(maxsize=1)
    obj._transcription_results_q = queue.Queue()
    started = threading.Event()
    release = threading.Event()
    obj._handle_transcription_result = Mock()

    def transcribe(_audio):
        started.set()
        release.wait(timeout=2)
        return "jarvis do this", "en", ()

    obj._transcribe_audio = transcribe
    obj._start_transcription_worker()
    worker = obj._transcription_worker_thread
    obj._transcription_jobs_q.put(SimpleNamespace(audio="before pause", start_time=1.0, end_time=2.0, energy=0.1, capture_generation=0, captured_during_tts=False, captured_tts_start_time=0))
    assert started.wait(timeout=1)
    obj.set_capture_paused('dictation', True)
    obj.set_capture_paused('dictation', False)
    release.set()
    obj._transcription_jobs_q.put(None)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert obj._transcription_results_q.empty()


@pytest.mark.parametrize(
    "captured_during_tts,captured_tts_start_time,should_interrupt",
    [(False, 0.0, False), (True, 12.0, True), (True, 9.0, False)],
)
def test_delayed_stop_only_interrupts_tts_that_overlapped_capture(
    captured_during_tts, captured_tts_start_time, should_interrupt,
):
    from unittest.mock import Mock

    obj = listener()
    obj.cfg.wake_word = "jarvis"
    obj.cfg.wake_aliases = []
    obj.cfg.tune_enabled = False
    obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: True, interrupt=Mock())
    obj.echo_detector = SimpleNamespace(_tts_start_time=12.0, _last_tts_finish_time=0,
                                        _last_tts_text="", echo_tolerance=0.3)
    obj.state_manager = Mock()
    obj.state_manager.was_speech_during_hot_window.return_value = False
    obj.state_manager.is_collecting.return_value = False
    obj._intent_judge = None
    obj._transcript_buffer = Mock()
    obj._stop_thinking_tune = Mock()
    obj._is_thinking_tune_active = lambda: False
    obj._start_thinking_tune = Mock()
    obj._set_face_state_listening = Mock()
    obj._clear_audio_buffers = Mock()
    obj._wake_timestamp = None

    start_time, end_time = ((12.1, 12.5) if captured_tts_start_time == 12.0 and captured_during_tts
                            else (10.0, 11.0))
    obj._process_transcript("stop", 0.2, start_time, end_time,
                            captured_during_tts=captured_during_tts,
                            captured_tts_start_time=captured_tts_start_time, generation=obj._capture_generation)

    assert obj.tts.interrupt.called is should_interrupt


def test_delayed_utterance_does_not_inherit_later_tts_text_for_intent():
    from unittest.mock import Mock

    obj = listener()
    obj.cfg.wake_word = "jarvis"
    obj.cfg.wake_aliases = []
    obj.cfg.tune_enabled = False
    obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: False)
    obj.echo_detector = SimpleNamespace(_tts_start_time=12.0, _last_tts_finish_time=13.0,
                                        _last_tts_text="future reply", echo_tolerance=0.3)
    obj.state_manager = Mock()
    obj.state_manager.was_speech_during_hot_window.return_value = False
    obj.state_manager.is_collecting.return_value = False
    obj._intent_judge = Mock(available=True)
    obj._intent_judge.judge.return_value = None
    obj._intent_judge.last_failure_reason = "unavailable"
    obj._transcript_buffer = Mock()
    obj._buffer_duration = 120
    obj._stop_thinking_tune = Mock()
    obj._is_thinking_tune_active = lambda: False
    obj._start_thinking_tune = Mock()
    obj._set_face_state_listening = Mock()
    obj._clear_audio_buffers = Mock()
    obj._wake_timestamp = None

    obj._process_transcript("jarvis weather", 0.2, 10.0, 11.0,
                            captured_during_tts=False, captured_tts_start_time=0, generation=obj._capture_generation)

    assert obj._intent_judge.judge.call_args.kwargs["last_tts_text"] == ""


@pytest.mark.parametrize('configured', [None, '', 'default', 'system'])
def test_default_input_is_pinned_across_capture_phases(configured):
    default = {'index': 3, 'max_input_channels': 1}
    opened = []

    def query_devices(*, kind):
        assert kind == 'input'
        return default.copy()

    def open_stream(**kwargs):
        opened.append(kwargs['device'])
        return object()

    audio = SimpleNamespace(query_devices=query_devices, InputStream=open_stream)
    selected = audio_capture.resolve_input_device(audio, configured)
    audio_capture.open_input_stream(audio, 16000, 20, selected)
    default['index'] = 7
    audio_capture.open_input_stream(audio, 16000, 20, selected)
    assert opened == [3, 3]


def test_missing_default_input_is_actionable_without_selecting_another_device():
    def missing_default(*, kind):
        raise RuntimeError('Error querying device -1')

    audio = SimpleNamespace(query_devices=missing_default)
    with pytest.raises(ValueError, match='default microphone.*Settings'):
        audio_capture.resolve_input_device(audio, None, [
            {'index': 2, 'name': 'Other microphone', 'max_input_channels': 1},
        ])


def test_output_only_default_is_not_accepted_as_a_microphone():
    audio = SimpleNamespace(query_devices=lambda **kwargs: {
        'index': 2, 'max_input_channels': 0,
    })
    with pytest.raises(ValueError, match='default microphone.*Settings'):
        audio_capture.resolve_input_device(audio, None)


def test_listener_reports_missing_default_before_loading_whisper(monkeypatch, capsys):
    obj = listener()
    obj.cfg.voice_device = None
    obj._warm_up_models = lambda: pytest.fail('unexpected model warm-up')

    def devices(*args, **kwargs):
        if kwargs.get('kind') == 'input':
            raise RuntimeError('Error querying device -1')
        return [{'index': 0, 'name': 'Available mic', 'max_input_channels': 1}]

    monkeypatch.setattr(capture.sd, 'query_devices', devices)
    monkeypatch.setattr(capture.sd, 'InputStream', lambda **kwargs: pytest.fail('unexpected capture'))
    monkeypatch.setattr(capture, 'WhisperModel', lambda *args, **kwargs: pytest.fail('unexpected model load'))
    obj.run()
    assert 'default microphone' in capsys.readouterr().out

@pytest.fixture(autouse=True)
def cached_whisper_files(monkeypatch):
    """Model-loading tests use synthetic local files and never access the Hub."""
    monkeypatch.setattr("jarvis.listening.model_download.prepare_faster_whisper_model",
                        lambda name: name)


@pytest.mark.parametrize('frame_ms,tts_speaking,limit_frames,silent_endpoint', [
    (10, False, 7, False), (20, False, 11, False),
    (30, False, 5, False), (20, True, 6, False),
    (20, False, 1, False), (20, False, 11, True),
])
def test_speech_reaches_whisper_at_configured_boundary(
    monkeypatch, frame_ms, tts_speaking, limit_frames, silent_endpoint,
):
    from unittest.mock import MagicMock
    from jarvis.config import load_settings

    from dataclasses import replace
    cfg = replace(
        load_settings(), voice_debug=False, vad_enabled=False,
        vad_frame_ms=frame_ms, vad_pre_roll_ms=0,
        max_utterance_ms=limit_frames * frame_ms,
        tts_max_utterance_ms=(limit_frames if tts_speaking else 2) * frame_ms,
        whisper_backend='faster-whisper', whisper_device='cpu',
        whisper_min_audio_duration=0, endpoint_silence_ms=3 * frame_ms,
    )
    monkeypatch.setattr(capture, 'create_intent_judge', lambda _cfg: None)
    monkeypatch.setattr(capture, 'FASTER_WHISPER_AVAILABLE', True)
    monkeypatch.setattr(capture, '_load_faster_whisper_model', lambda *a, **kw: MagicMock())
    monkeypatch.setattr(VoiceListener, '_decode_faster_whisper', lambda *a: ([], None))
    monkeypatch.setattr(VoiceListener, '_start_llm_warmup', lambda _self: [])
    monkeypatch.setattr(VoiceListener, '_start_transcription_worker', lambda _self: None)
    monkeypatch.setattr('desktop_app.face_widget.get_jarvis_state', lambda: MagicMock())
    obj = VoiceListener(MagicMock(), cfg, SimpleNamespace(is_speaking=lambda: tts_speaking), None)
    frame_samples = cfg.sample_rate * frame_ms // 1000
    speech_frames = limit_frames - 2 if silent_endpoint else limit_frames * 3
    speech = np.linspace(.05, .2, frame_samples * speech_frames, dtype=np.float32)
    audio = (np.concatenate([speech, np.zeros(frame_samples * 3, dtype=np.float32)])
             if silent_endpoint else speech)

    class Feed(queue.Queue):
        def get(self, *args, **kwargs):
            if self.empty():
                obj._should_stop = True
                raise queue.Empty
            return super().get(*args, **kwargs)

    obj._audio_q = Feed()

    class Stream:
        active = True
        def start(self):
            obj._on_audio(audio[:, None], len(audio), None, None)
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
    obj.run()

    jobs = list(obj._transcription_jobs_q.queue)
    expected_count = 1 if silent_endpoint else 3
    expected_frames = speech_frames if silent_endpoint else limit_frames
    assert len(jobs) == expected_count, 'Speech must be submitted at its capture boundary'
    assert all(len(job.audio) == frame_samples * expected_frames for job in jobs)
    np.testing.assert_array_equal(np.concatenate([job.audio for job in jobs]), speech)


@pytest.mark.parametrize('reset', ['dictation_cycle', 'query_boundary', 'callback_status'])
def test_callback_interleaved_with_reset_does_not_relabel_old_audio_as_fresh(monkeypatch, reset):
    """A state transition during admission cannot revive pre-transition speech."""
    obj = listener(16000)
    paused = VoiceListener.capture_paused
    first_admission = True

    def interrupted_pause_read(instance):
        nonlocal first_admission
        active = paused.fget(instance)
        if first_admission and reset != 'callback_status':
            first_admission = False
            if reset == 'dictation_cycle':
                instance.set_capture_paused('dictation', True)
                instance.set_capture_paused('dictation', False)
            else:
                instance._clear_audio_buffers()
        return active

    class ResetDuringStatus:
        def __str__(self):
            obj._clear_audio_buffers()
            return 'input overflow'

    status = ResetDuringStatus() if reset == 'callback_status' else None
    with monkeypatch.context() as scoped:
        scoped.setattr(VoiceListener, 'capture_paused', property(interrupted_pause_read))
        obj._on_audio(np.ones((320, 1), dtype=np.float32) * .1, 320, None, status)

    obj._on_audio(np.ones((320, 1), dtype=np.float32) * .2, 320, None, None)
    admitted = []
    obj._is_speech_frame = lambda frame: admitted.append(float(np.mean(frame))) or False
    obj._check_audio_health = lambda: None

    queued = obj._audio_q
    class DrainThenStop:
        def get(self, **kwargs):
            if queued.empty():
                obj._should_stop = True
                raise queue.Empty
            return queued.get_nowait()
    obj._audio_q = DrainThenStop()
    obj._consume_audio_frames(20)

    assert admitted == pytest.approx([.2])
