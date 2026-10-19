"""Task Centre validates local frames, retains drafts and bounds its history."""
import io
import json
from types import SimpleNamespace

import pytest

from desktop_app import app as desktop
from desktop_app.task_centre import TaskCentreWindow
from jarvis import daemon

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def task_event(identity='one', status='completed'):
    return {'type': 'task', 'id': identity, 'prompt': f'Request {identity}',
            'status': status, 'result': 'Done', 'error': None}


@pytest.fixture
def window(qapp):
    view = TaskCentreWindow(lambda *args, **kwargs: None, lambda task: None)
    yield view
    view.close()
    view.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize('line', [
    '__TASK__:[]', '__TASK__:{}', '__TASK__:{"id":[],"status":"running"}',
    '__TASK__:' + json.dumps(dict(task_event(), status=[])),
    '__TASK__:' + json.dumps(dict(task_event(), prompt=None)),
    '__TASK__:' + json.dumps(dict(task_event(), result={})),
    'Heard: __TASK__:' + json.dumps(task_event()),
    '  __TASK__:' + json.dumps(task_event()),
])
def test_malformed_or_embedded_frames_cannot_change_the_window(window, line):
    assert window.process_log_line(line) is False
    assert window.task_list.count() == 0


def test_retained_history_is_bounded_and_active_tasks_stay_usable(qapp):
    limit = 2
    view = TaskCentreWindow(lambda *args, **kwargs: None, lambda task: None, max_history=limit)
    try:
        view.add_or_update_task(task_event('active', 'running'))
        for index in range(limit + 3):
            view.add_or_update_task(task_event(str(index)))
        assert view.task_list.count() == limit + 1
        view.add_or_update_task(task_event('active', 'completed'))
        assert view.task_list.count() == limit
        assert any('active' in view.task_list.item(i).text() for i in range(view.task_list.count()))
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('accepted', [False, True], ids=['rejected', 'accepted'])
def test_source_submission_waits_for_correlated_admission(qapp, monkeypatch, capsys, accepted):
    wire = io.StringIO()
    tray = desktop.JarvisSystemTray.__new__(desktop.JarvisSystemTray)
    tray.is_bundled = False
    tray.daemon_process = SimpleNamespace(stdin=wire)
    manager = SimpleNamespace(submit=lambda prompt: 'accepted-task') if accepted else None
    monkeypatch.setattr(daemon, '_global_task_manager', manager)
    view = TaskCentreWindow(tray.submit_task, lambda task: None)
    try:
        draft = 'Keep this request until the daemon accepts it.'
        view.prompt.setPlainText(draft)
        view._submit()
        assert view.prompt.toPlainText() == draft
        assert not view.submit_button.isEnabled()
        line = wire.getvalue().strip()
        command = json.loads(line.split(':', 1)[1])
        assert isinstance(command['request_id'], str) and command['request_id']
        assert daemon.handle_task_stdin_line(line)
        frames = [line for line in capsys.readouterr().out.splitlines() if line.startswith('__TASK__:')]
        assert frames
        for frame in frames:
            view.process_log_line(frame)
        assert view.submit_button.isEnabled()
        if accepted:
            assert view.prompt.toPlainText() == ''
        else:
            assert view.prompt.toPlainText() == draft
            assert 'not ready' in view.details.text().lower()
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_submission_timeout_retains_the_draft_and_late_ack_cannot_clear_a_retry(qapp, monkeypatch):
    from PyQt6.QtTest import QTest
    from desktop_app import task_centre
    sent = []
    monkeypatch.setattr(task_centre, 'SUBMISSION_TIMEOUT_MS', 1)
    view = TaskCentreWindow(lambda prompt, **kwargs: sent.append(kwargs['request_id']), lambda task: None)
    try:
        view.prompt.setPlainText('Original draft')
        view._submit()
        QTest.qWait(20)
        assert view.prompt.toPlainText() == 'Original draft'
        assert view.submit_button.isEnabled()
        assert 'No confirmation' in view.details.text()
        view.prompt.setPlainText('Revised draft')
        view._submit()
        late = {'type': 'submission', 'request_id': sent[0],
                'accepted': True, 'task_id': 'old-admission'}
        assert view.process_log_line('__TASK__:' + json.dumps(late))
        assert view.prompt.toPlainText() == 'Revised draft'
        assert not view.submit_button.isEnabled()
        current = dict(late, request_id=sent[-1], task_id='current-admission')
        assert view.process_log_line('__TASK__:' + json.dumps(current))
        assert view.prompt.toPlainText() == ''
        assert view.submit_button.isEnabled()
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_task_frames_reach_the_gui_thread_without_entering_activity_logs(qapp):
    from threading import Thread
    view = TaskCentreWindow(lambda *args, **kwargs: None, lambda task: None)
    signals = desktop.LogSignals()
    logs = []
    signals.new_log.connect(logs.append)
    signals.task_event.connect(view.process_log_line)
    worker = Thread(target=lambda: signals.dispatch_output('__TASK__:' + json.dumps(task_event())))
    try:
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert view.task_list.count() == 0
        qapp.processEvents()
        assert view.task_list.count() == 1
        assert logs == []
        signals.dispatch_output('📋 A useful activity message')
        assert logs == ['📋 A useful activity message']
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_bundled_admission_is_immediate_and_errors_preserve_the_draft(qapp):
    def admit(prompt, *, request_id):
        if prompt == 'Reject this':
            raise RuntimeError('Task queue is full')
        return 'admitted'

    view = TaskCentreWindow(admit, lambda task: None)
    try:
        view.prompt.setPlainText('Accept this')
        view._submit()
        assert view.prompt.toPlainText() == ''
        assert view.submit_button.isEnabled()
        view.prompt.setPlainText('Reject this')
        view._submit()
        assert view.prompt.toPlainText() == 'Reject this'
        assert view.submit_button.isEnabled()
        assert 'queue is full' in view.details.text()
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()
