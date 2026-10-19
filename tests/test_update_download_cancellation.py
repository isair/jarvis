"""Update cancellation keeps the GUI responsive and ignores stale completion."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PyQt6.QtWidgets import QDialog

from desktop_app import updater, update_dialog

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def drain_until(qapp, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    assert predicate()


@pytest.mark.parametrize('action', ['button', 'close', 'escape'])
def test_cancel_stalled_download_keeps_gui_responsive(tmp_path, monkeypatch, qapp, action):
    entered = threading.Event()
    release = threading.Event()
    watchdog_fired = threading.Event()
    closed = threading.Event()

    class Response:
        headers = {}
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            yield b'partial'
            entered.set()
            assert release.wait(5)
            yield b'remainder'
        def close(self): closed.set()
        def __enter__(self): return self
        def __exit__(self, *args): self.close()

    monkeypatch.setattr(updater.requests, 'get', lambda *a, **kw: Response())
    staging = tmp_path / 'staging'
    staging.mkdir()
    monkeypatch.setattr(update_dialog.tempfile, 'mkdtemp', lambda: str(staging))
    install = Mock()
    monkeypatch.setattr(update_dialog, 'install_update', install)
    scheduled = []
    monkeypatch.setattr(update_dialog.QTimer, "singleShot", lambda delay, callback: scheduled.append(callback))
    callback = Mock()
    dialog = update_dialog.UpdateProgressDialog(SimpleNamespace(asset_name='bundle.zip', download_url='https://example.invalid/fixture', asset_id=1), pre_install_callback=callback)
    dialog.show()
    dialog.start_download()
    worker = dialog.download_worker
    timer = threading.Timer(2, lambda: (watchdog_fired.set(), release.set()))
    try:
        assert entered.wait(5)
        timer.start()
        if action == 'button': dialog.cancel_btn.click()
        elif action == 'close': dialog.close()
        else: dialog.reject()
        assert not watchdog_fired.is_set(), 'Cancelling blocked the GUI waiting for network I/O'
        assert worker.isRunning(), 'Controlled network wait should still be active'
        # A queued completion must not prepare or install a cancelled update.
        dialog._on_completed(str(staging / 'bundle.zip'))
        callback.assert_not_called()
        release.set()
        drain_until(qapp, lambda: not worker.isRunning())
        qapp.processEvents()
        assert not staging.exists()
        assert closed.is_set()
        assert not dialog.isVisible()
        for pending in scheduled:
            pending()
        install.assert_not_called()
    finally:
        timer.cancel()
        release.set()
        worker.wait(5000)
        qapp.processEvents()
        dialog.deleteLater()


@pytest.mark.parametrize('outcome', ['success', 'error', 'cancel', 'cancel_error'])
def test_download_response_is_closed_before_terminal_result(tmp_path, monkeypatch, qapp, outcome):
    closed = threading.Event()
    events = []
    signals = updater.DownloadSignals()
    signals.completed.connect(lambda path: events.append(('completed', path, closed.is_set())))
    signals.error.connect(lambda message: events.append(('error', message, closed.is_set())))
    target = tmp_path / 'download.zip'
    worker = updater.DownloadWorker('https://example.invalid/fixture', target, signals)

    class Response:
        headers = {}
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            yield b'owned payload'
            if outcome.startswith('cancel'):
                worker.cancel()
            if outcome in ('error', 'cancel_error'):
                raise OSError('controlled interrupted transfer')
        def close(self): closed.set()
        def __enter__(self): return self
        def __exit__(self, *args): self.close()

    monkeypatch.setattr(updater.requests, 'get', lambda *a, **kw: Response())
    worker.run()
    assert closed.is_set()
    if outcome.startswith('cancel'):
        assert events == []
    else:
        assert len(events) == 1
        assert events[0][0] == ('completed' if outcome == 'success' else 'error')
        assert events[0][2] is True
        if outcome == 'success':
            assert target.read_bytes() == b'owned payload'


def test_deferred_install_does_not_run_after_cancel(tmp_path, monkeypatch, qapp):
    scheduled = []
    monkeypatch.setattr(update_dialog.QTimer, 'singleShot', lambda delay, callback: scheduled.append(callback))
    install = Mock(return_value=True)
    monkeypatch.setattr(update_dialog, 'install_update', install)
    monkeypatch.setattr(update_dialog, 'save_installed_asset_id', Mock())
    dialog = update_dialog.UpdateProgressDialog(SimpleNamespace(asset_name='bundle.zip', download_url='https://example.invalid/fixture', asset_id=1))
    dialog._on_completed(str(tmp_path / 'bundle.zip'))
    assert scheduled
    dialog.reject()
    for callback in scheduled:
        callback()
    install.assert_not_called()
    dialog.deleteLater()


@pytest.mark.parametrize('action', ['close', 'escape'])
def test_staged_update_finishes_restart_after_handoff(tmp_path, qapp, action):
    pending = []
    dialog = update_dialog.UpdateProgressDialog(SimpleNamespace(asset_id=1))
    dialog.show()
    try:
        with (
            patch.object(update_dialog, 'install_update', return_value=True),
            patch.object(update_dialog, 'save_installed_asset_id'),
            patch.object(update_dialog.QTimer, 'singleShot', side_effect=lambda delay, callback: pending.append(callback)),
        ):
            dialog._on_completed(str(tmp_path / 'bundle.zip'))
            pending.pop(0)()
            if action == 'close':
                dialog.close()
            else:
                dialog.reject()
            assert dialog.isVisible(), 'Cancellation interrupted restart after installer handoff'
            for callback in pending:
                callback()
            assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        dialog.deleteLater()
