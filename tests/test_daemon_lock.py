"""daemon_lock.py — single-instance voice daemon."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_acquire_writes_current_pid(tmp_path):
    from jarvis import daemon_lock

    lock = tmp_path / "jarvis_daemon.lock"
    original = daemon_lock._lock_handle
    try:
        with patch.object(daemon_lock, "daemon_lock_path", return_value=lock):
            daemon_lock._lock_handle = None
            assert daemon_lock.acquire_daemon_lock() is True
            assert daemon_lock.read_lock_pid() == os.getpid()
    finally:
        daemon_lock.release_daemon_lock()
        daemon_lock._lock_handle = original


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows mandatory lock subprocess test")
def test_second_process_blocked_on_windows(tmp_path):
    import subprocess
    import sys

    from jarvis import daemon_lock

    lock = tmp_path / "jarvis_daemon.lock"
    original = daemon_lock._lock_handle
    child = f'''
import msvcrt
LOCK_OFFSET = 1024
fh = open(r"""{lock}""", "a+b")
fh.seek(LOCK_OFFSET)
try:
    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    print("LOCK_ACQUIRED")
except OSError:
    print("LOCK_BLOCKED")
fh.close()
'''
    try:
        with patch.object(daemon_lock, "daemon_lock_path", return_value=lock):
            daemon_lock._lock_handle = None
            assert daemon_lock.acquire_daemon_lock() is True
        proc = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "LOCK_BLOCKED" in proc.stdout
    finally:
        daemon_lock.release_daemon_lock()
        daemon_lock._lock_handle = original


@pytest.mark.unit
def test_is_daemon_running_stale_pid_cleared(tmp_path):
    from jarvis import daemon_lock

    lock = tmp_path / "jarvis_daemon.lock"
    lock.write_text("999999", encoding="utf-8")
    with patch.object(daemon_lock, "daemon_lock_path", return_value=lock):
        daemon_lock.clear_stale_lock_if_needed()
        assert daemon_lock.is_daemon_running() is False
        assert not lock.exists()


@pytest.mark.unit
def test_stop_locked_daemon_not_running(tmp_path):
    from jarvis import daemon_lock

    lock = tmp_path / "jarvis_daemon.lock"
    with patch.object(daemon_lock, "daemon_lock_path", return_value=lock):
        stopped, pid, message = daemon_lock.stop_locked_daemon()
        assert stopped is False
        assert pid is None
        assert message == "not running"
