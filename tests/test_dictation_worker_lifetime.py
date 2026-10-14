"""Hotkey release stays responsive while owned device teardown is blocked."""
from threading import Barrier, Event, Thread
from unittest.mock import Mock

import pytest

from jarvis.dictation import dictation_engine as dictation

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('release_count', [1, 2])
def test_release_returns_before_device_close_and_closes_once(monkeypatch, release_count):
    entered = Event()
    allow_close = Event()
    closed = []
    beeps = []
    errors = []
    workers = []
    callers = []
    returned = [Event() for _ in range(release_count)]
    ready = Barrier(release_count + 1)

    class Stream:
        def stop(self):
            entered.set()
            if not allow_close.wait(5):
                errors.append('Test did not release device teardown')

        def close(self):
            closed.append(self)

    def start_worker(*args, **kwargs):
        worker = Thread(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(dictation, 'parse_hotkey', lambda hotkey: (frozenset(), None))
    engine = dictation.DictationEngine(
        whisper_model_ref=lambda: None, whisper_backend_ref=lambda: 'faster-whisper',
        mlx_repo_ref=lambda: None, history=Mock(),
    )
    engine._recording = True
    engine._stream = Stream()

    def release_hotkey(index):
        try:
            ready.wait(timeout=2)
            engine._stop_recording()
        except Exception as error:
            errors.append(error)
        finally:
            returned[index].set()

    # Keep effects mocked until every caller and teardown worker has finished.
    with monkeypatch.context() as effects:
        effects.setattr(dictation, '_play_beep', lambda *args: beeps.append(True))
        effects.setattr(dictation.threading, 'Thread', start_worker)
        try:
            for index in range(release_count):
                caller = Thread(target=release_hotkey, args=(index,))
                callers.append(caller)
                caller.start()
            ready.wait(timeout=2)
            assert entered.wait(2), 'Device teardown did not begin'
            assert all(done.wait(.2) for done in returned), 'Hotkey callback blocked on device teardown'
            assert not closed, 'Device close ran before the test released teardown'
        finally:
            ready.abort()
            allow_close.set()
            for thread in callers:
                thread.join(timeout=2)
            for thread in workers:
                thread.join(timeout=2)
            assert all(not thread.is_alive() for thread in callers + workers), 'Test left a worker alive'

    assert not errors, errors
    assert len(closed) == 1, 'Concurrent release closed the same device more than once'
    assert len(beeps) == 1, 'Teardown notification escaped or repeated'
