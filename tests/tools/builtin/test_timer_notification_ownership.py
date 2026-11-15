"""Elapsed timers retain ownership until their asynchronous speech terminates."""
import threading
from types import SimpleNamespace

import pytest
from jarvis import activity
from jarvis.output import tts as speech
from jarvis.tools.builtin import timer

pytestmark = pytest.mark.unit


@pytest.fixture(params=[speech.PiperTTS, speech.ChatterboxTTS], ids=['piper', 'chatterbox'])
def service(request, make_speech_backend, monkeypatch):
    engine = make_speech_backend(request.param)
    submitted = threading.Event()
    requests = []
    speak = engine.speak
    def submit(*args, **kwargs):
        result = speak(*args, **kwargs)
        requests.append(result)
        submitted.set()
        return result
    monkeypatch.setattr(engine, 'speak', submit)
    monkeypatch.setattr(timer, '_tts_provider', lambda: engine)
    registry = SimpleNamespace(start=lambda entry: None, stop=lambda identity: None,
                               stop_by_label=lambda label: None, stop_all=lambda: None)
    monkeypatch.setattr(timer, 'get_alarm_registry', lambda: registry)
    manager = timer.TimerManager()
    try:
        yield manager, engine, submitted, requests
    finally:
        manager.shutdown(timeout_sec=1)
        engine.stop()
        assert manager.shutdown(timeout_sec=1)


def elapsed(service):
    manager, engine, submitted, requests = service
    entry = manager.start(.01, 'Owned timer', announcement='Owned timer finished')
    assert submitted.wait(3), 'Owned timer did not submit its announcement'
    entry.timer.join(3)
    assert not entry.timer.is_alive()
    return entry


def test_queued_notification_still_occupies_timer_admission_capacity(service, monkeypatch):
    manager, engine, _, _ = service
    monkeypatch.setattr(timer, '_MAX_ACTIVE_TIMERS', 1)
    elapsed(service)
    with pytest.raises(RuntimeError, match='too many'):
        manager.start(30, 'Over capacity')
    engine._run()
    replacement = manager.start(30, 'Capacity released at terminal')
    assert replacement.id


@pytest.mark.parametrize('route', ['id', 'label', 'all'])
def test_cancel_elapsed_timer_prevents_its_queued_speech(service, route):
    manager, engine, _, _ = service
    entry = elapsed(service)
    if route == 'id':
        assert manager.cancel(entry.id) is entry
    elif route == 'label':
        assert entry in manager.cancel_by_label(entry.label)
    else:
        assert entry in manager.cancel_all()
    engine._run()
    assert engine.get_last_spoken_text() == ''
    assert not engine.is_speaking()


def test_shutdown_cancels_only_its_notifications_and_preserves_unrelated_speech(service):
    manager, engine, _, _ = service
    engine.speak('first')
    service[2].clear()
    elapsed(service)
    assert manager.shutdown(timeout_sec=1)
    engine._run()
    assert engine.get_last_spoken_text() == 'first'


def test_timer_synthesis_preparation_does_not_replace_other_activity(service):
    _, engine, _, _ = service
    state = activity.get_jarvis_state()
    state.set_state(activity.JarvisState.THINKING)
    elapsed(service)
    assert state.state is activity.JarvisState.THINKING
    engine._run()
    assert state.state is activity.JarvisState.IDLE


def test_unavailable_timer_speech_preserves_other_activity(service):
    manager, engine, _, _ = service
    engine.enabled = False
    state = activity.get_jarvis_state()
    state.set_state(activity.JarvisState.DICTATING)
    entry = manager.start(.01, 'Silent timer', announcement='Owned timer finished')
    entry.timer.join(3)
    assert not entry.timer.is_alive()
    assert state.state is activity.JarvisState.DICTATING


def test_cancellation_during_cold_speech_preparation_prevents_late_playback(service, monkeypatch):
    manager, engine, _, _ = service
    entered, release = threading.Event(), threading.Event()
    def prepare(generation):
        entered.set()
        assert release.wait(5), 'Owned cold speech preparation was not released'
    monkeypatch.setattr(engine, '_start_if_current', prepare)
    entry = manager.start(.01, 'Cold timer', announcement='Owned timer finished')
    try:
        assert entered.wait(3)
        assert manager.cancel(entry.id) is entry
        release.set()
        entry.timer.join(3)
        assert not entry.timer.is_alive()
        engine._run()
        assert engine.get_last_spoken_text() == ''
    finally:
        release.set()
        entry.timer.join(3)
        assert not entry.timer.is_alive()


def test_shutdown_retains_notification_synthesis_until_terminal_release(service, monkeypatch):
    manager, engine, _, _ = service
    entered, release = threading.Event(), threading.Event()
    target, method = (engine._voice, 'synthesize') if isinstance(engine, speech.PiperTTS) else (engine._model, 'generate')
    original = getattr(target, method)
    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(5), 'Owned notification synthesis was not released'
        return original(*args, **kwargs)
    monkeypatch.setattr(target, method, generate)
    elapsed(service)
    worker = threading.Thread(target=engine._run)
    worker.start()
    try:
        assert entered.wait(3)
        assert not manager.shutdown(timeout_sec=.01)
        monkeypatch.setattr(timer, '_manager_instance', manager)
        with pytest.raises(RuntimeError, match='still draining'):
            timer.initialise_timer_manager()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert manager.shutdown(timeout_sec=1)
        replacement = timer.initialise_timer_manager()
        assert replacement is not manager
        assert replacement.shutdown(timeout_sec=1)
        assert engine.get_last_spoken_text() == ''
    finally:
        release.set()
        worker.join(3)
        assert not worker.is_alive()


def test_timer_output_uses_shared_echo_tracking_without_conversational_follow_up(service, mock_config, monkeypatch):
    from jarvis.listening.listener import VoiceListener
    manager, engine, _, _ = service
    mock_config.tune_enabled = False
    mock_config.hot_window_enabled = True
    monkeypatch.setattr('jarvis.listening.listener.create_intent_judge', lambda cfg: None)
    listener = VoiceListener(None, mock_config, engine, None)
    try:
        entry = elapsed(service)
        assert listener.echo_detector._tts_start_time == 0
        engine._run()
        assert listener.echo_detector._last_tts_text == entry.announcement.lower()
        assert listener.echo_detector._tts_exact_duration == 1.0
        assert listener.echo_detector._last_tts_finish_time >= listener.echo_detector._tts_start_time > 0
        assert not listener.state_manager.is_hot_window_active()
    finally:
        listener.stop()


@pytest.mark.parametrize('outcome', ['unavailable', 'synthesis-failure', 'engine-stop'])
def test_notification_terminal_paths_release_capacity(service, monkeypatch, outcome):
    manager, engine, _, _ = service
    monkeypatch.setattr(timer, '_MAX_ACTIVE_TIMERS', 1)
    if outcome == 'unavailable':
        engine.enabled = False
        entry = manager.start(.01, 'Unavailable notification')
        entry.timer.join(3)
        assert not entry.timer.is_alive()
    else:
        if outcome == 'synthesis-failure':
            target, method = (engine._voice, 'synthesize') if isinstance(engine, speech.PiperTTS) else (engine._model, 'generate')
            def fail(*args, **kwargs):
                raise RuntimeError('Owned timer synthesis failure')
            monkeypatch.setattr(target, method, fail)
        elapsed(service)
        if outcome == 'engine-stop':
            engine.stop()
        else:
            engine._run()
    replacement = manager.start(30, 'Terminal released capacity')
    assert replacement.id
    assert engine.get_last_spoken_text() == ''

@pytest.mark.parametrize('route', ['shutdown', 'request-cancel'])
def test_shutdown_does_not_wait_for_native_playback_start(service, monkeypatch, route):
    manager, engine, _, _ = service
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    results = []
    unrelated_completed = threading.Event()
    if isinstance(engine, speech.PiperTTS):
        import sounddevice
        target, method = sounddevice.OutputStream, 'start'
    else:
        import pygame
        target, method = pygame.mixer.music, 'play'
    original = getattr(target, method)
    def start(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5), 'Owned playback start was not released'
        return original(*args, **kwargs)
    monkeypatch.setattr(target, method, start)
    elapsed(service)
    engine.speak('second', completion_callback=unrelated_completed.set)
    worker = threading.Thread(target=engine._run)
    def shutdown():
        if route == 'request-cancel':
            service[3][0].cancel()
        results.append(manager.shutdown(timeout_sec=.01))
        returned.set()
    closer = threading.Thread(target=shutdown)
    worker.start()
    try:
        assert entered.wait(3)
        closer.start()
        assert returned.wait(1), 'Bounded shutdown waited for native playback start'
        assert results == [False]
    finally:
        release.set()
        worker.join(3)
        if closer.ident is not None:
            closer.join(3)
        assert not worker.is_alive()
        assert not closer.is_alive()
        assert manager.shutdown(timeout_sec=1)
    assert unrelated_completed.is_set()
    assert engine.get_last_spoken_text() == 'second'


@pytest.mark.parametrize('route', ['helper', 'new-timer'])
def test_dismissal_cancels_queued_notification_and_preserves_future_countdown(service, monkeypatch, route):
    manager, engine, _, _ = service
    monkeypatch.setattr(timer, '_manager_instance', manager)
    future = manager.start(60, 'future countdown')
    elapsed(service)
    if route == 'helper':
        timer.stop_all_alarms()
    else:
        manager.start(30, 'replacement countdown')
    engine._run()
    assert engine.get_last_spoken_text() == ''
    assert future.id in [entry.id for entry in manager.list()]


@pytest.mark.parametrize('route,command', [('configured', 'dur'), ('configured', '止めて'), ('intent', 'please be quiet for a moment')])
def test_configured_voice_stop_dismisses_alarm_without_cancelling_future_timer(service, mock_config, monkeypatch, command, route):
    import time
    from jarvis.listening.listener import VoiceListener
    manager, engine, _, _ = service
    monkeypatch.setattr(timer, '_manager_instance', manager)
    active = set()
    registry = timer.get_alarm_registry()
    monkeypatch.setattr(registry, 'start', lambda entry: active.add(entry.id))
    def stop_all():
        count = len(active)
        active.clear()
        return count
    monkeypatch.setattr(registry, 'stop_all', stop_all)
    mock_config.tune_enabled = False
    mock_config.stop_commands = [command] if route == 'configured' else []
    if route == 'intent':
        from jarvis.listening.intent_judge import IntentJudgment
        judgment = IntentJudgment(directed=True, stop=True, confidence='high', reasoning='Owned stop verdict')
        judge = SimpleNamespace(available=True, config=SimpleNamespace(model='owned-verdict'), judge=lambda **kwargs: judgment)
        command = 'please be quiet for a moment'
    else:
        judge = None
    monkeypatch.setattr('jarvis.listening.listener.create_intent_judge', lambda cfg: judge)
    listener = VoiceListener(None, mock_config, engine, None)
    entered, release = threading.Event(), threading.Event()
    def started(request, duration):
        listener.on_playback_started(request, duration)
        entered.set()
        assert release.wait(5), 'Owned timer playback was not released'
    engine.set_playback_callbacks(started, listener.on_playback_finished)
    future = manager.start(60, 'future countdown')
    entry = elapsed(service)
    worker = threading.Thread(target=engine._run)
    worker.start()
    try:
        assert entered.wait(3)
        assert active == {entry.id}
        now = time.time()
        listener._process_transcript(command, utterance_energy=1.0, utterance_start_time=now, utterance_end_time=now,
            captured_during_tts=True, captured_tts_start_time=listener.echo_detector._tts_start_time,
            generation=listener._capture_generation, language=None)
        assert active == set()
        assert future.id in [item.id for item in manager.list()]
        assert not listener.state_manager.is_hot_window_active()
    finally:
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        listener.stop()


@pytest.mark.parametrize('scenario,command,dismisses', [
    ('finished-speech', 'dur', True),
    ('disabled-speech', '止めて', True),
    ('stale-capture', 'dur', False),
    ('ambient', 'unrelated conversation', False),
])
@pytest.mark.parametrize("capture_during_tts", [False, True])
def test_configured_stop_reaches_bare_alarm_only_for_current_user_capture(
        service, mock_config, monkeypatch, scenario, command, dismisses, capture_during_tts):
    import time
    from jarvis.listening.listener import VoiceListener
    manager, engine, _, _ = service
    registry = timer.AlarmRegistry()
    monkeypatch.setattr(registry, '_stdout_is_tty', lambda: False)
    monkeypatch.setattr(timer, 'get_alarm_registry', lambda: registry)
    monkeypatch.setattr(timer, '_manager_instance', manager)
    monkeypatch.setattr('jarvis.listening.listener.create_intent_judge', lambda cfg: None)
    mock_config.tune_enabled = False
    mock_config.hot_window_enabled = True
    mock_config.stop_commands = ['dur', '止めて']
    if scenario == 'disabled-speech':
        engine.enabled = False
    listener = VoiceListener(None, mock_config, engine, None)
    try:
        future = manager.start(60, 'future countdown')
        before_alarm = time.time() - 60
        if scenario == 'disabled-speech':
            entry = manager.start(.01, 'Owned timer', announcement='Owned timer finished')
            entry.timer.join(3)
            assert not entry.timer.is_alive()
        else:
            entry = elapsed(service)
        engine._run()
        assert registry.active_ids() == [entry.id]
        assert not engine.is_speaking()
        captured_at = before_alarm if scenario == 'stale-capture' else time.time()
        unrelated_completed = []
        if scenario == 'finished-speech':
            # The owned draining worker exits on an empty queue; re-arm its next pass.
            engine._stop.clear()
            engine.speak('Unrelated queued speech', completion_callback=lambda: unrelated_completed.append(True))
        listener._process_transcript(command, utterance_energy=1.0,
            utterance_start_time=captured_at, utterance_end_time=captured_at,
            captured_during_tts=capture_during_tts, captured_tts_start_time=listener.echo_detector._tts_start_time,
            generation=listener._capture_generation, language=None)
        assert registry.active_ids() == ([] if dismisses else [entry.id])
        if scenario == 'finished-speech':
            engine._run()
            assert unrelated_completed == [True]
            assert engine.get_last_spoken_text() == 'Unrelated queued speech'
        assert future.id in [item.id for item in manager.list()]
        assert not listener.state_manager.is_hot_window_active()
    finally:
        registry.stop_all()
        listener.stop()
