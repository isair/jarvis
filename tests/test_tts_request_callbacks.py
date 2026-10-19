"""Speech callbacks belong to their utterance, including queued and reentrant work."""
import queue
import sys

import pytest

from jarvis.output import tts as speech

pytestmark = pytest.mark.unit


@pytest.fixture(params=[speech.PiperTTS, speech.ChatterboxTTS], ids=['piper', 'chatterbox'])
def voice(request, make_speech_backend):
    return make_speech_backend(request.param)


def test_queued_utterances_keep_their_own_callbacks(voice):
    events = []
    for name, duration in [('first', 1.0), ('second', 2.0)]:
        voice.speak(name,
                    duration_callback=lambda seconds, name=name: events.append((name, seconds)),
                    completion_callback=lambda name=name: events.append((name, voice.get_last_spoken_text())))
    voice._run()
    assert events == [('first', 1.0), ('first', 'first'),
                      ('second', 2.0), ('second', 'second')]


def test_a_later_unhooked_utterance_does_not_erase_an_earlier_callback(voice):
    completed = []
    voice.speak('first', completion_callback=lambda: completed.append(voice.get_last_spoken_text()))
    voice.speak('second')
    voice._run()
    assert completed == ['first']
    assert voice.get_last_spoken_text() == 'second'


@pytest.mark.parametrize('enqueue_at', ['duration', 'completion'])
def test_a_callback_can_enqueue_another_utterance_without_losing_ownership(voice, enqueue_at):
    completed = []

    def enqueue():
        voice.speak('second', completion_callback=lambda: completed.append(('second', voice.get_last_spoken_text())))

    def duration(seconds):
        if enqueue_at == 'duration':
            enqueue()

    def complete():
        completed.append(('first', voice.get_last_spoken_text()))
        if enqueue_at == 'completion':
            enqueue()

    voice.speak('first', duration_callback=duration, completion_callback=complete)
    voice._run()
    assert completed == [('first', 'first'), ('second', 'second')]


def test_interrupted_synthesis_cannot_complete_a_different_queued_utterance(voice):
    completed = []
    voice.speak('cancelled', completion_callback=lambda: completed.append(('cancelled', voice.get_last_spoken_text())))
    voice.speak('second', completion_callback=lambda: completed.append(('second', voice.get_last_spoken_text())))
    voice._run()
    assert completed == [('second', 'second')]


@pytest.mark.parametrize('failing_callback', ['duration', 'completion'])
def test_callback_failure_does_not_steal_later_notifications(voice, failing_callback):
    completed = []

    def failed(*args):
        raise RuntimeError('Owned callback failure')

    voice.speak('first',
                duration_callback=failed if failing_callback == 'duration' else None,
                completion_callback=failed if failing_callback == 'completion' else None)
    voice.speak('second', completion_callback=lambda: completed.append(voice.get_last_spoken_text()))
    voice._run()
    assert completed == ['second']


def test_real_worker_preserves_callback_ownership_with_concurrent_producers(voice, monkeypatch):
    import threading

    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    voice.start()
    worker = voice._thread
    completed = []
    done = threading.Event()
    expected = [(producer, name, name) for producer in range(4) for name in ('first', 'second')]

    def enqueue(producer):
        for name in ('first', 'second'):
            def complete(producer=producer, name=name):
                completed.append((producer, name, voice.get_last_spoken_text()))
                if len(completed) == len(expected):
                    done.set()
            voice.speak(name, completion_callback=complete)

    producers = [threading.Thread(target=enqueue, args=(producer,)) for producer in range(4)]
    try:
        for producer in producers:
            producer.start()
        for producer in producers:
            producer.join(timeout=3)
            assert not producer.is_alive()
        assert done.wait(timeout=3), completed
        assert sorted(completed) == sorted(expected)
    finally:
        voice.stop()
        for producer in producers:
            producer.join(timeout=3)
    assert worker is not None and not worker.is_alive()


def test_idle_worker_releases_its_completed_callback(voice, monkeypatch):
    import gc
    import threading
    import weakref

    completed = threading.Event()
    idle = threading.Event()

    class ObservableQueue(queue.Queue):
        def get(self, *args, **kwargs):
            if completed.is_set():
                idle.set()
            return super().get(*args, **kwargs)

    class Completion:
        def __call__(self):
            completed.set()

    voice._q = ObservableQueue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    voice.start()
    worker = voice._thread
    callback = Completion()
    reference = weakref.ref(callback)
    try:
        voice.speak('first', completion_callback=callback)
        del callback
        assert idle.wait(timeout=3)
        gc.collect()
        assert reference() is None, 'Completed callback remains owned by an idle worker'
    finally:
        voice.stop()
    assert worker is not None and not worker.is_alive()


@pytest.mark.parametrize('submit_while_draining', [False, True])
def test_stopped_synthesis_discards_pending_speech_and_restarts_cleanly(voice, monkeypatch, submit_while_draining):
    import threading

    entered = threading.Event()
    release = threading.Event()
    import gc
    import weakref

    completed = []
    durations = []
    synthesised = []
    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))

    def wait_for_release(text):
        synthesised.append(text)
        if text == 'first':
            entered.set()
            assert release.wait(timeout=6)

    if isinstance(voice, speech.PiperTTS):
        original = voice._voice.synthesize
        def synthesise(text, cfg):
            wait_for_release(text)
            return original(text, cfg)
        voice._voice.synthesize = synthesise
    else:
        original = voice._model.generate
        def generate(text, **kwargs):
            wait_for_release(text)
            return original(text, **kwargs)
        voice._model.generate = generate

    voice.speak('first', completion_callback=lambda: completed.append('first'),
                duration_callback=durations.append)
    worker = voice._thread
    restarted = threading.Event()
    try:
        assert entered.wait(timeout=3)
        class PendingCallback:
            def __call__(self):
                completed.append('stale')
        callback = PendingCallback()
        reference = weakref.ref(callback)
        voice.speak('second', completion_callback=callback)
        del callback
        voice.stop()
        gc.collect()
        assert reference() is None, 'Stopping retains a pending callback'
        if submit_while_draining:
            voice.speak('second', completion_callback=lambda: completed.append('during shutdown'))
        release.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert synthesised == ['first']
        assert completed == []
        assert durations == []
        voice.speak('second', completion_callback=restarted.set)
        assert restarted.wait(timeout=3), 'A drained engine cannot restart'
        assert synthesised == ['first', 'second']
    finally:
        release.set()
        voice.stop()
        voice._stop.set()
        voice._q.put_nowait(None)
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_completion_callback_can_stop_the_engine_without_joining_itself(voice, monkeypatch):
    import threading

    stopped = threading.Event()
    restarted = threading.Event()
    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))

    def complete():
        voice.stop()
        stopped.set()

    voice.speak('first', completion_callback=complete)
    worker = voice._thread
    try:
        assert stopped.wait(timeout=3), 'Stopping in a callback raised or deadlocked'
        worker.join(timeout=3)
        assert not worker.is_alive()
        voice.speak('second', completion_callback=restarted.set)
        assert restarted.wait(timeout=3)
    finally:
        voice.stop()
        voice._stop.set()
        voice._q.put_nowait(None)
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_stopping_from_duration_callback_does_not_start_playback(voice, monkeypatch):
    import threading

    stopped = threading.Event()
    playback = []
    completed = []
    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    monkeypatch.setattr(sys.modules['sounddevice'].OutputStream, 'start',
                        lambda self: playback.append('piper'))
    monkeypatch.setattr(sys.modules['pygame'].mixer.music, 'play',
                        lambda: playback.append('chatterbox'))

    def duration(seconds):
        voice.stop()
        stopped.set()

    voice.speak('first', duration_callback=duration, completion_callback=lambda: completed.append('first'))
    worker = voice._thread
    try:
        assert stopped.wait(timeout=3)
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert playback == []
        assert completed == []
    finally:
        voice.stop()
        voice._stop.set()
        voice._q.put_nowait(None)
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_shutdown_cannot_complete_speech_while_discarding_pending_requests(voice, monkeypatch):
    import threading

    synthesising = threading.Event()
    release_synthesis = threading.Event()
    discarding = threading.Event()
    release_discard = threading.Event()
    completed = []

    class PausedDiscardQueue(queue.Queue):
        def get_nowait(self):
            if voice._stop.is_set():
                discarding.set()
                assert release_discard.wait(timeout=5)
            return super().get_nowait()

    voice._q = PausedDiscardQueue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    if isinstance(voice, speech.PiperTTS):
        original = voice._voice.synthesize
        def synthesise(text, cfg):
            synthesising.set()
            assert release_synthesis.wait(timeout=5)
            return original(text, cfg)
        voice._voice.synthesize = synthesise
    else:
        original = voice._model.generate
        def generate(text, **kwargs):
            synthesising.set()
            assert release_synthesis.wait(timeout=5)
            return original(text, **kwargs)
        voice._model.generate = generate
    voice.speak('first', completion_callback=lambda: completed.append('first'))
    worker = voice._thread
    stopper = threading.Thread(target=voice.stop)
    try:
        assert synthesising.wait(timeout=3)
        stopper.start()
        assert discarding.wait(timeout=3)
        release_synthesis.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert completed == []
    finally:
        release_synthesis.set()
        release_discard.set()
        stopper.join(timeout=3)
        voice.stop()
        worker.join(timeout=3)
        assert not stopper.is_alive() and not worker.is_alive()


@pytest.mark.parametrize('voice', [speech.ChatterboxTTS], indirect=True)
def test_slow_audio_preparation_does_not_block_shutdown(voice, monkeypatch):
    import threading

    preparing = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    playback = []
    completed = []
    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))

    def prepare(**kwargs):
        preparing.set()
        assert release.wait(timeout=6)

    monkeypatch.setattr(sys.modules['pygame'].mixer, 'init', prepare)
    monkeypatch.setattr(sys.modules['pygame'].mixer.music, 'play', lambda: playback.append('started'))
    voice.speak('first', completion_callback=lambda: completed.append('first'))
    worker = voice._thread
    def stop():
        voice.stop()
        stopped.set()
    stopper = threading.Thread(target=stop)
    try:
        assert preparing.wait(timeout=3)
        stopper.start()
        assert stopped.wait(timeout=3), 'Native audio preparation blocks shutdown'
        release.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert playback == []
        assert completed == []
    finally:
        release.set()
        stopper.join(timeout=3)
        voice.stop()
        worker.join(timeout=3)
        assert not stopper.is_alive() and not worker.is_alive()


def test_model_preparation_does_not_hold_the_shutdown_lock(voice, monkeypatch):
    import threading

    loading = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    voice._q = queue.Queue()
    def load():
        loading.set()
        assert release.wait(timeout=6)
        return True
    loader = '_ensure_initialized' if isinstance(voice, speech.PiperTTS) else '_ensure_model'
    monkeypatch.setattr(voice, loader, load)
    starter = threading.Thread(target=voice.start)
    def stop():
        voice.stop()
        stopped.set()
    stopper = threading.Thread(target=stop)
    try:
        starter.start()
        assert loading.wait(timeout=3)
        stopper.start()
        assert stopped.wait(timeout=3), 'Model preparation holds the shutdown lock'
    finally:
        release.set()
        starter.join(timeout=3)
        stopper.join(timeout=3)
        voice.stop()
        assert not starter.is_alive() and not stopper.is_alive()


@pytest.mark.parametrize('newer_submission', [False, True])
def test_cancelled_model_load_cannot_submit_stale_speech(voice, monkeypatch, newer_submission):
    import threading

    loading = threading.Event()
    release = threading.Event()
    stale = threading.Event()
    fresh = threading.Event()
    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))

    def load():
        if threading.current_thread() is starter:
            loading.set()
            assert release.wait(timeout=6)
        return True

    loader = '_ensure_initialized' if isinstance(voice, speech.PiperTTS) else '_ensure_model'
    monkeypatch.setattr(voice, loader, load)
    starter = threading.Thread(target=lambda: voice.speak('first', completion_callback=stale.set))
    try:
        starter.start()
        assert loading.wait(timeout=3)
        voice.stop()
        if newer_submission:
            voice.speak('second', completion_callback=fresh.set)
            assert fresh.wait(timeout=3)
        release.set()
        starter.join(timeout=3)
        assert not starter.is_alive()
        assert not stale.wait(timeout=1), 'Cancelled model loading submitted stale speech'
        if not newer_submission:
            voice.speak('second', completion_callback=fresh.set)
            assert fresh.wait(timeout=3), 'A fresh request cannot use the prepared model'
        assert voice.get_last_spoken_text() == 'second'
    finally:
        release.set()
        starter.join(timeout=3)
        worker = voice._thread
        voice.stop()
        if worker is not None:
            worker.join(timeout=3)
            assert not worker.is_alive()
        assert not starter.is_alive()


@pytest.mark.parametrize('reason', ['disabled', 'empty'])
def test_rejected_speech_releases_ownership_without_success(voice, reason):
    outcomes = []
    completed = []
    if reason == 'disabled':
        voice.enabled = False
    request = voice.speak(
        '' if reason == 'empty' else 'first',
        completion_callback=lambda: completed.append(True),
        terminal_callback=outcomes.append,
    )
    assert outcomes == ['rejected']
    assert completed == []
    request.cancel()
    assert outcomes == ['rejected']


def test_success_releases_each_request_after_its_success_callback(voice):
    events = []
    for text in ('first', 'second'):
        voice.speak(
            text, completion_callback=lambda text=text: events.append((text, 'success')),
            terminal_callback=lambda outcome, text=text: events.append((text, outcome)),
        )
    voice._run()
    assert events == [('first', 'success'), ('first', 'completed'),
                      ('second', 'success'), ('second', 'completed')]


def test_failed_synthesis_releases_ownership_without_success(voice):
    outcomes = []
    completed = []

    def failed(*args, **kwargs):
        raise RuntimeError('Owned synthesis failure')

    if isinstance(voice, speech.PiperTTS):
        voice._voice.synthesize = failed
    else:
        voice._model.generate = failed
    voice.speak('first', completion_callback=lambda: completed.append(True),
                terminal_callback=outcomes.append)
    voice._run()
    assert completed == []
    assert outcomes == ['failed']


def test_stopping_releases_queued_ownership_exactly_once(voice):
    completed = []
    outcomes = []
    request = voice.speak('first', completion_callback=lambda: completed.append(True),
                          terminal_callback=outcomes.append)
    voice.stop()
    voice.stop()
    request.cancel()
    assert completed == []
    assert outcomes == ['cancelled']


def test_cancelling_one_pending_request_keeps_later_speech(voice):
    outcomes = []
    completed = []
    first = voice.speak('first', completion_callback=lambda: completed.append('first'),
                        terminal_callback=lambda outcome: outcomes.append(('first', outcome)))
    voice.speak('second', completion_callback=lambda: completed.append('second'),
                terminal_callback=lambda outcome: outcomes.append(('second', outcome)))
    first.cancel()
    assert outcomes == [('first', 'cancelled')]
    voice._run()
    assert completed == ['second']
    assert outcomes == [('first', 'cancelled'), ('second', 'completed')]


@pytest.mark.parametrize("action", ["cancel", "interrupt"])
def test_cancellation_between_claim_and_synthesis_cannot_start_playback(voice, monkeypatch, action):
    import threading

    claimed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    playback = []
    outcomes = []
    completed = []
    original = voice._speak_once

    def paused(request):
        claimed.set()
        assert release.wait(timeout=5)
        return original(request)

    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    monkeypatch.setattr(voice, '_speak_once', paused)
    monkeypatch.setattr(sys.modules['sounddevice'].OutputStream, 'start', lambda self: playback.append('piper'))
    monkeypatch.setattr(sys.modules['pygame'].mixer.music, 'play', lambda: playback.append('chatterbox'))

    def terminal(outcome):
        outcomes.append(outcome)
        finished.set()

    request = voice.speak('first', completion_callback=lambda: completed.append(True),
                          terminal_callback=terminal)
    worker = voice._thread
    try:
        assert claimed.wait(timeout=3)
        request.cancel() if action == "cancel" else voice.interrupt()
        assert outcomes == [], 'Active ownership was released before its worker ended'
        release.set()
        assert finished.wait(timeout=3)
        assert playback == []
        assert completed == []
        assert outcomes == ['cancelled' if action == 'cancel' else 'interrupted']
    finally:
        release.set()
        voice.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_terminal_callback_can_enqueue_and_fail_without_stealing_ownership(voice):
    events = []

    def terminal(outcome):
        events.append(('first', outcome))
        voice.speak('second', terminal_callback=lambda outcome: events.append(('second', outcome)))
        raise RuntimeError('Owned terminal callback failure')

    voice.speak('first', terminal_callback=terminal)
    voice._run()
    assert events == [('first', 'completed'), ('second', 'completed')]


@pytest.mark.parametrize('voice', [speech.PiperTTS], indirect=True)
@pytest.mark.parametrize('mode', ['callback-error', 'premature-stop'])
def test_native_callback_failure_cannot_report_success(voice, monkeypatch, mode):
    completed = []
    outcomes = []

    class BrokenOutput:
        def __setitem__(self, key, value):
            raise ValueError('Owned output callback failure')

    class FailedNativeStream:
        active = False

        def __init__(self, **kwargs):
            self.callback = kwargs['callback']

        def start(self):
            # Native callback threads terminate playback without propagating
            # callback exceptions to the thread waiting on stream.active.
            try:
                if mode == "callback-error":
                    self.callback(BrokenOutput(), 1024, None, None)
            except Exception:
                pass

        def close(self):
            pass

        def abort(self):
            pass

    monkeypatch.setattr(sys.modules['sounddevice'], 'OutputStream', FailedNativeStream)
    voice.speak('first', completion_callback=lambda: completed.append(True),
                terminal_callback=outcomes.append)
    voice._run()
    assert completed == []
    assert outcomes == ['failed']


def test_interrupt_before_terminal_commit_prevents_success(voice, monkeypatch):
    import threading

    playback_returned = threading.Event()
    observed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    completed = []
    outcomes = []
    original = voice._speak_once

    class PausedCompletionCheck(threading.Event):
        def is_set(self):
            result = super().is_set()
            if playback_returned.is_set() and not observed.is_set():
                observed.set()
                assert release.wait(timeout=5)
            return result

    def playback(request):
        result = original(request)
        playback_returned.set()
        return result

    def terminal(outcome):
        outcomes.append(outcome)
        finished.set()

    voice._q = queue.Queue()
    voice._should_interrupt = PausedCompletionCheck()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    monkeypatch.setattr(voice, '_speak_once', playback)
    voice.speak('first', completion_callback=lambda: completed.append(True), terminal_callback=terminal)
    worker = voice._thread
    try:
        assert observed.wait(timeout=3)
        voice.interrupt()
        release.set()
        assert finished.wait(timeout=3)
        assert completed == []
        assert outcomes == ['interrupted']
    finally:
        release.set()
        voice.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_interrupt_at_terminal_entry_keeps_active_request_ownership(voice, monkeypatch):
    import threading

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    completed = []
    outcomes = []
    original = speech.SpeechRequest._finish

    def paused(request, outcome):
        if request.text == 'first' and outcome == 'completed':
            entered.set()
            assert release.wait(timeout=5)
        original(request, outcome)

    def terminal(outcome):
        outcomes.append(outcome)
        finished.set()

    voice._q = queue.Queue()
    monkeypatch.setattr(voice, '_start_if_current', type(voice)._start_if_current.__get__(voice))
    monkeypatch.setattr(speech.SpeechRequest, '_finish', paused)
    voice.speak('first', completion_callback=lambda: completed.append(True), terminal_callback=terminal)
    worker = voice._thread
    try:
        assert entered.wait(timeout=3)
        voice.interrupt()
        release.set()
        assert finished.wait(timeout=3)
        assert completed == []
        assert outcomes == ['interrupted']
    finally:
        release.set()
        voice.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()


@pytest.mark.parametrize('voice', [speech.PiperTTS], indirect=True)
def test_failed_audio_start_closes_stream_before_terminal_release(voice, monkeypatch):
    events = []

    class FailedStart:
        active = False

        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError('Owned audio start failure')

        def close(self):
            events.append('closed')

        def abort(self):
            pass

    monkeypatch.setattr(sys.modules['sounddevice'], 'OutputStream', FailedStart)
    voice.speak('first', completion_callback=lambda: events.append('success'),
                terminal_callback=events.append)
    voice._run()
    assert events == ['closed', 'failed']


def test_playback_callbacks_and_activity_follow_actual_queued_output(voice):
    from jarvis import activity
    state = activity.get_jarvis_state()
    state.set_state(activity.JarvisState.THINKING)
    events = []

    def started(request, duration):
        events.append(('start', request.text, duration, state.state, voice.is_speaking()))

    def finished(request, outcome):
        events.append(('end', request.text, outcome, voice.is_speaking()))

    voice.set_playback_callbacks(started, finished)
    voice.speak('first')
    voice.speak('second')
    assert events == []
    assert state.state is activity.JarvisState.THINKING
    assert not voice.is_speaking()
    voice._run()
    assert events == [
        ('start', 'first', 1.0, activity.JarvisState.SPEAKING, True),
        ('end', 'first', 'completed', False),
        ('start', 'second', 2.0, activity.JarvisState.SPEAKING, True),
        ('end', 'second', 'completed', False),
    ]
    assert state.state is activity.JarvisState.IDLE


def test_synthesis_does_not_publish_speaking_or_replace_last_playback(voice):
    from jarvis import activity
    state = activity.get_jarvis_state()
    state.set_state(activity.JarvisState.THINKING)
    observations = []
    if isinstance(voice, speech.PiperTTS):
        original = voice._voice.synthesize
        def synthesise(*args, **kwargs):
            observations.append((voice.is_speaking(), voice.get_last_spoken_text(), state.state))
            return original(*args, **kwargs)
        voice._voice.synthesize = synthesise
    else:
        original = voice._model.generate
        def generate(*args, **kwargs):
            observations.append((voice.is_speaking(), voice.get_last_spoken_text(), state.state))
            return original(*args, **kwargs)
        voice._model.generate = generate
    voice.speak('first')
    voice._run()
    assert observations == [(False, '', activity.JarvisState.THINKING)]
    assert voice.get_last_spoken_text() == 'first'
    assert state.state is activity.JarvisState.IDLE


@pytest.mark.parametrize('protected', ['DICTATING', 'DICTATION_PROCESSING'])
def test_playback_keeps_protected_dictation_presence(voice, protected):
    from jarvis import activity
    state = activity.get_jarvis_state()
    value = activity.JarvisState[protected]
    state.set_state(value)
    completed = []
    voice.speak('first', completion_callback=lambda: completed.append(True))
    voice._run()
    assert completed == [True]
    assert state.state is value


@pytest.mark.parametrize('newer', ['LISTENING', 'THINKING', 'DICTATING', 'DICTATION_PROCESSING', 'SPEAKING'])
def test_playback_cleanup_preserves_a_newer_presence(voice, newer):
    from jarvis import activity
    state = activity.get_jarvis_state()
    value = activity.JarvisState[newer]
    voice.set_playback_callbacks(lambda request, duration: state.set_state(value), None)
    voice.speak('first')
    voice._run()
    assert state.state is value


@pytest.mark.parametrize('binding', ['before-playback', 'during-first'])
def test_playback_observer_binding_is_current_at_start_and_owned_until_finish(voice, binding):
    events = []

    def new_start(request, duration):
        events.append(('new-start', request.text))

    def new_end(request, outcome):
        events.append(('new-end', request.text, outcome))

    def old_start(request, duration):
        events.append(('old-start', request.text))
        if binding == 'during-first' and request.text == 'first':
            voice.set_playback_callbacks(new_start, new_end)

    def old_end(request, outcome):
        events.append(('old-end', request.text, outcome))

    voice.set_playback_callbacks(old_start, old_end)
    voice.speak('first')
    voice.speak('second')
    if binding == 'before-playback':
        voice.set_playback_callbacks(new_start, new_end)
    voice._run()
    first = 'new' if binding == 'before-playback' else 'old'
    assert events == [(first+'-start', 'first'), (first+'-end', 'first', 'completed'),
                      ('new-start', 'second'), ('new-end', 'second', 'completed')]


def test_pending_cancellation_never_publishes_playback_or_speech_presence(voice):
    from jarvis import activity
    events = []
    voice.set_playback_callbacks(lambda *args: events.append('start'), lambda *args: events.append('finish'))
    request = voice.speak('first')
    request.cancel()
    voice._run()
    assert events == []
    assert not voice.is_speaking()
    assert voice.get_last_spoken_text() == ''
    assert activity.get_jarvis_state().state is activity.JarvisState.IDLE


@pytest.mark.parametrize('stage', ['duration', 'start', 'finish', 'completion', 'terminal'])
@pytest.mark.parametrize('cancelled_callback', [False, True], ids=['exception', 'cancelled-error'])
def test_failed_callback_does_not_skip_cleanup_or_abandon_the_next_utterance(voice, stage, cancelled_callback):
    import asyncio
    from jarvis import activity
    outcomes = []
    completed = []

    def notify(label, *args):
        if label == 'terminal':
            outcomes.append(args[0])
        if label == 'completion':
            completed.append('first')
        if label == stage:
            if cancelled_callback:
                raise asyncio.CancelledError('Owned callback cancellation')
            raise RuntimeError('Owned callback failure')

    voice.set_playback_callbacks(
        lambda request, duration: notify('start') if request.text == 'first' else None,
        lambda request, outcome: notify('finish') if request.text == 'first' else None,
    )
    voice.speak('first', duration_callback=lambda seconds: notify('duration'),
                completion_callback=lambda: notify('completion'),
                terminal_callback=lambda outcome: notify('terminal', outcome))
    voice.speak('second', completion_callback=lambda: completed.append('second'),
                terminal_callback=outcomes.append)
    voice._run()
    assert outcomes == ['completed', 'completed']
    assert completed == ['first', 'second']
    assert activity.get_jarvis_state().state is activity.JarvisState.IDLE
    assert not voice.is_speaking()
