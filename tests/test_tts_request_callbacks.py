"""Speech callbacks belong to their utterance, including queued and reentrant work."""
import queue
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jarvis.output import tts as speech

pytestmark = pytest.mark.unit


@pytest.fixture(params=[speech.PiperTTS, speech.ChatterboxTTS], ids=['piper', 'chatterbox'])
def voice(request, monkeypatch):
    engine = request.param()
    monkeypatch.setattr(engine, 'start', lambda: None)
    monkeypatch.setattr(engine, '_notify_speaking_state', lambda speaking: None)

    class DrainingQueue(queue.Queue):
        def get(self, *args, **kwargs):
            try:
                return super().get(block=False)
            except queue.Empty:
                engine._stop.set()
                raise

    engine._q = DrainingQueue()
    sample_rate = 100
    lengths = {'first': sample_rate, 'second': sample_rate * 2, 'cancelled': sample_rate}

    def samples(text):
        if text == 'cancelled':
            engine.interrupt()
        return np.zeros(lengths[text], dtype=np.int16)

    class SilentStream:
        active = False

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    import sounddevice
    monkeypatch.setattr(sounddevice, 'OutputStream', SilentStream)
    config = ModuleType('piper.config')
    config.SynthesisConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'piper.config', config)
    engine._sample_rate = sample_rate
    engine._voice = SimpleNamespace(synthesize=lambda text, cfg: [
        SimpleNamespace(audio_int16_array=samples(text))])
    monkeypatch.setattr(engine, '_ensure_initialized', lambda: True, raising=False)

    engine._model = SimpleNamespace(sr=sample_rate, generate=lambda text, **kwargs: samples(text)[None, :])
    monkeypatch.setattr(engine, '_ensure_model', lambda: True, raising=False)
    audio = ModuleType('torchaudio')
    audio.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'torchaudio', audio)
    pygame = ModuleType('pygame')
    music = SimpleNamespace(load=lambda path: None, play=lambda: None,
                            stop=lambda: None, get_busy=lambda: False)
    pygame.mixer = SimpleNamespace(init=lambda **kwargs: None, quit=lambda: None, music=music)
    pygame.time = SimpleNamespace(wait=lambda duration: None)
    monkeypatch.setitem(sys.modules, 'pygame', pygame)
    yield engine
    engine._stop.set()


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
    monkeypatch.setattr(voice, 'start', type(voice).start.__get__(voice))
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
    monkeypatch.setattr(voice, 'start', type(voice).start.__get__(voice))
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
    monkeypatch.setattr(voice, 'start', type(voice).start.__get__(voice))

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
    monkeypatch.setattr(voice, 'start', type(voice).start.__get__(voice))

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
    monkeypatch.setattr(voice, 'start', type(voice).start.__get__(voice))
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
