"""Echo timing and follow-up ownership follow actual local speech playback."""
import pytest
from jarvis import activity
from jarvis.output import tts as speech
from jarvis.listening.listener import VoiceListener

pytestmark = pytest.mark.unit


@pytest.fixture(params=[speech.PiperTTS, speech.ChatterboxTTS], ids=['piper', 'chatterbox'])
def playback(request, make_speech_backend, mock_config, monkeypatch):
    engine = make_speech_backend(request.param)
    mock_config.tune_enabled = False
    mock_config.hot_window_enabled = True
    monkeypatch.setattr('jarvis.listening.listener.create_intent_judge', lambda cfg: None)
    listener = VoiceListener(None, mock_config, engine, None)
    yield listener, engine
    listener.stop()


def test_all_played_speech_uses_preprocessed_text_and_exact_audio_duration(playback):
    listener, engine = playback
    raw = '**A local announcement**'
    request = engine.speak(raw)
    assert listener.echo_detector._last_tts_text == ''
    assert listener.echo_detector._tts_start_time == 0
    engine._run()
    assert listener.echo_detector._last_tts_text == request.text.lower()
    assert listener.echo_detector._tts_exact_duration == 1.0
    assert listener.echo_detector._last_tts_finish_time >= listener.echo_detector._tts_start_time > 0
    assert not listener.state_manager.is_hot_window_active()


def test_cancelled_unplayed_speech_cannot_replace_echo_history(playback):
    listener, engine = playback
    engine.speak('first')
    cancelled = engine.speak('second')
    cancelled.cancel()
    engine._run()
    assert listener.echo_detector._last_tts_text == 'first'
    assert listener.echo_detector._tts_exact_duration == 1.0


@pytest.mark.parametrize('result', ['answer', 'error', 'empty', 'disabled', 'failed-synthesis'])
def test_voice_dispatch_retains_thinking_until_delivery_or_terminal_failure(playback, monkeypatch, result):
    listener, engine = playback
    from jarvis.reply import engine as reply
    def respond(*args, **kwargs):
        if result == 'error':
            raise RuntimeError('Owned reply failure')
        return '' if result == 'empty' else 'first'
    monkeypatch.setattr(reply, 'run_reply_engine', respond)
    if result == 'disabled':
        engine.enabled = False
    if result == 'failed-synthesis':
        monkeypatch.setattr(engine, '_speak_once', lambda request: False)
    listener._dispatch_query('question', generation=listener._capture_generation, language=None)
    state = activity.get_jarvis_state()
    if result in {'empty', 'disabled'}:
        assert state.state is activity.JarvisState.IDLE
    else:
        assert state.state is activity.JarvisState.THINKING
        assert listener.echo_detector._last_tts_text == ''
        engine._run()
        assert state.state is activity.JarvisState.IDLE
        if result == 'failed-synthesis':
            assert listener.echo_detector._last_tts_text == ''
        else:
            assert listener.echo_detector._tts_start_time > 0
            assert listener.echo_detector._last_tts_finish_time >= listener.echo_detector._tts_start_time


def test_stale_playback_finish_cannot_end_a_newer_echo_window(playback):
    listener, _ = playback
    first, second = speech.SpeechRequest('first'), speech.SpeechRequest('second')
    listener.on_playback_started(first, 1.0)
    listener.on_playback_started(second, 2.0)
    listener.on_playback_finished(first, 'interrupted')
    assert listener.echo_detector._last_tts_text == 'second'
    assert listener.echo_detector._last_tts_finish_time == 0
    listener.on_playback_finished(second, 'completed')
    assert listener.echo_detector._last_tts_finish_time > 0


@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'failed', 'error'])
def test_only_successful_voice_reply_opens_follow_up(playback, monkeypatch, outcome):
    import time
    listener, engine = playback
    from jarvis.reply import engine as reply
    def respond(*args, **kwargs):
        if outcome == 'error':
            raise RuntimeError('Owned reply failure')
        return 'first'
    monkeypatch.setattr(reply, 'run_reply_engine', respond)
    listener.state_manager.echo_tolerance = 0
    requests = []
    speak = engine.speak
    def submit(*args, **kwargs):
        request = speak(*args, **kwargs)
        requests.append(request)
        return request
    monkeypatch.setattr(engine, 'speak', submit)
    if outcome == 'failed':
        monkeypatch.setattr(engine, '_speak_once', lambda request: False)
    listener._dispatch_query('question', generation=listener._capture_generation, language=None)
    assert not listener.state_manager.is_hot_window_active()
    if outcome == 'cancelled':
        requests[0].cancel()
    engine._run()
    deadline = time.monotonic() + .5
    if outcome == 'completed':
        while not listener.state_manager.is_hot_window_active() and time.monotonic() < deadline:
            time.sleep(.001)
    assert listener.state_manager.is_hot_window_active() is (outcome == 'completed')
    assert activity.get_jarvis_state().state is activity.JarvisState.IDLE


def test_rejected_acknowledgement_releases_presence_without_a_thinking_tune(playback):
    listener, _ = playback
    state = activity.get_jarvis_state()
    listener._set_face_state_listening()
    assert state.state is activity.JarvisState.LISTENING
    listener._stop_thinking_tune()
    assert state.state is activity.JarvisState.IDLE
    listener._set_face_state_listening()
    state.set_state(activity.JarvisState.DICTATING)
    listener._stop_thinking_tune()
    assert state.state is activity.JarvisState.DICTATING


@pytest.mark.parametrize('control,interrupts', [('configured', True), ('ambient', False), ('stale', False)])
def test_synthesis_remains_interruptible_without_claiming_playback(playback, monkeypatch, control, interrupts):
    import threading
    import time
    listener, engine = playback
    entered, release = threading.Event(), threading.Event()
    completed, outcomes = [], []
    if isinstance(engine, speech.PiperTTS):
        target, method = engine._voice, 'synthesize'
    else:
        target, method = engine._model, 'generate'
    original = getattr(target, method)
    def held(*args, **kwargs):
        entered.set()
        assert release.wait(5), 'Owned synthesis was not released'
        return original(*args, **kwargs)
    monkeypatch.setattr(target, method, held)
    engine.speak('first', completion_callback=lambda: completed.append(True), terminal_callback=outcomes.append)
    worker = threading.Thread(target=engine._run)
    worker.start()
    try:
        assert entered.wait(3)
        assert not engine.is_speaking()
        assert listener.echo_detector._tts_start_time == 0
        utterance = 'unrelated ambient chatter' if control == 'ambient' else listener.cfg.stop_commands[0]
        now = time.time() - (60 if control == 'stale' else 0)
        listener._process_transcript(utterance, utterance_start_time=now, utterance_end_time=now,
                                     language=None, generation=listener._capture_generation,
                                     captured_during_tts=False, captured_tts_start_time=0)
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert outcomes == ['interrupted' if interrupts else 'completed']
        assert completed == ([] if interrupts else [True])
        assert bool(listener.echo_detector._tts_start_time) is not interrupts
    finally:
        release.set()
        worker.join(3)
        assert not worker.is_alive()
