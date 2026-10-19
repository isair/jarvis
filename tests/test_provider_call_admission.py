"""Queued provider work cannot start after its query has been abandoned."""
import threading

import pytest

from jarvis.reply import execution

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('outcome', ['deadline', 'cancel', 'success'])
def test_pending_provider_admission_respects_query_lifetime(monkeypatch, outcome):
    now = [0.0]
    cancelled = threading.Event()
    control = execution.ExecutionControl(cancelled, 10, clock=lambda: now[0])
    admitted = threading.Event()
    release = threading.Event()
    done = threading.Event()
    workers = []
    effects = []
    result = {}
    thread_type = threading.Thread

    class OwnedAdmissionThread(thread_type):
        def run(self):
            if self.name == 'jarvis-provider-call':
                workers.append(self)
                admitted.set()
                assert release.wait(3)
            super().run()

    monkeypatch.setattr(execution.threading, 'Thread', OwnedAdmissionThread)

    def provider():
        effects.append('request issued')
        return 'owned result'

    def request():
        try:
            result['value'] = control.call(provider)
        except BaseException as error:
            result['error'] = error
        finally:
            done.set()

    caller = thread_type(target=request, name='owned-query-wait')
    try:
        caller.start()
        assert admitted.wait(1)
        if outcome == 'deadline':
            now[0] = 11.0
        elif outcome == 'cancel':
            cancelled.set()
        else:
            release.set()
        assert done.wait(0.5)
        if outcome == 'success':
            assert result == {'value': 'owned result'}
        else:
            expected = execution.ExecutionDeadlineExceeded if outcome == 'deadline' else execution.ExecutionCancelled
            assert isinstance(result.get('error'), expected)
        release.set()
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        assert effects == (['request issued'] if outcome == 'success' else []), 'Abandoned query issued a late provider request'
    finally:
        release.set()
        caller.join(3)
        assert not caller.is_alive()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
