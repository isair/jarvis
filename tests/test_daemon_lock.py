"""The daemon single-instance lock — the guard against the "two voices" bug
(two daemons alive, each with its own listener + TTS, both answering)."""

import errno
import os
import subprocess
import sys

from pathlib import Path

import pytest

from jarvis.daemon_lock import acquire_daemon_lock, lock_holder_pid, release_daemon_lock


@pytest.mark.unit
def test_second_acquire_in_the_same_process_is_refused(tmp_path):
    # The bundled app runs the daemon on a thread: a second DaemonThread is a
    # second acquire in the SAME process, and must be refused too.
    path = tmp_path / "jarvis_daemon.lock"
    first = acquire_daemon_lock(path)
    assert first is not None
    try:
        assert acquire_daemon_lock(path) is None
    finally:
        release_daemon_lock(first)


@pytest.mark.unit
def test_release_frees_the_lock_and_the_holder_pid_is_recorded(tmp_path):
    path = tmp_path / "jarvis_daemon.lock"
    first = acquire_daemon_lock(path)
    assert lock_holder_pid(path) == os.getpid()
    release_daemon_lock(first)
    again = acquire_daemon_lock(path)
    assert again is not None
    release_daemon_lock(again)


@pytest.mark.unit
def test_closing_the_handle_releases_it(tmp_path):
    # A crashed daemon's handle is closed by the OS — no stale lock left behind.
    path = tmp_path / "jarvis_daemon.lock"
    first = acquire_daemon_lock(path)
    first.close()
    again = acquire_daemon_lock(path)
    assert again is not None
    release_daemon_lock(again)


@pytest.mark.unit
def test_second_main_is_refused_without_reviving_a_stopping_daemon(tmp_path, monkeypatch):
    from jarvis import daemon
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "jarvis_daemon.lock"))
    ran = []
    monkeypatch.setattr(daemon, "_run_daemon", lambda smoke_test=False: ran.append(smoke_test))

    held = acquire_daemon_lock()  # a first daemon, mid-shutdown
    monkeypatch.setattr(daemon, "_global_stop_requested", False)
    daemon.request_stop()
    try:
        daemon.main()
        assert ran == [], "a second daemon must not start while one holds the lock"
        assert daemon._global_stop_requested is True, "the stopping daemon's flag must not be reset"
    finally:
        release_daemon_lock(held)

    daemon.main()  # the lock is free again: starts normally
    assert ran == [False]


@pytest.mark.unit
def test_non_contention_lock_errors_are_reported(tmp_path, monkeypatch):
    if sys.platform == "win32":
        import msvcrt
        monkeypatch.setattr(msvcrt, "locking", lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")))
    else:
        import fcntl
        monkeypatch.setattr(fcntl, "flock", lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")))
    with pytest.raises(OSError, match="disk error"):
        acquire_daemon_lock(tmp_path / "daemon.lock")


@pytest.mark.unit
def test_override_directory_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "nested" / "daemon.lock"))
    held = acquire_daemon_lock()
    assert held is not None
    release_daemon_lock(held)


@pytest.mark.unit
def test_process_lock_survives_refusal_and_recovers_after_process_death(tmp_path):
    path = tmp_path / "daemon.lock"
    child_code = """
import sys
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from jarvis.daemon_lock import acquire_daemon_lock
held = acquire_daemon_lock(Path(sys.argv[1]))
assert held is not None
print("LOCKED", flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", child_code, str(path), str(Path(__file__).resolve().parents[1] / "src")],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = child.stdout.readline().strip()
        if ready != "LOCKED":
            _, error = child.communicate(timeout=10)
            pytest.fail(f"Child did not acquire the lock: {ready} {error}")
        assert lock_holder_pid(path) == child.pid
        assert acquire_daemon_lock(path) is None
        assert lock_holder_pid(path) == child.pid
    finally:
        child.kill()
        child.communicate(timeout=10)
    held = acquire_daemon_lock(path)
    assert held is not None
    release_daemon_lock(held)


@pytest.mark.unit
def test_exception_in_daemon_releases_lock(tmp_path, monkeypatch):
    from jarvis import daemon
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "daemon.lock"))
    def fail(smoke_test=False):
        raise RuntimeError("startup failed")
    monkeypatch.setattr(daemon, "_run_daemon", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        daemon.main()
    held = acquire_daemon_lock()
    assert held is not None
    release_daemon_lock(held)


@pytest.mark.unit
def test_smoke_initialisation_cannot_revive_active_daemon(tmp_path, monkeypatch):
    from jarvis import daemon
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "daemon.lock"))
    held = acquire_daemon_lock()
    ran = []
    monkeypatch.setattr(daemon, "_run_daemon", lambda smoke_test=False: ran.append(smoke_test))
    monkeypatch.setattr(daemon, "_global_stop_requested", True)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            daemon.main(smoke_test=True)
        assert ran == []
        assert daemon.is_stop_requested()
    finally:
        release_daemon_lock(held)


@pytest.mark.unit
def test_pid_write_failure_releases_ownership(tmp_path, monkeypatch):
    from jarvis import daemon_lock
    path = tmp_path / "daemon.lock"
    fdopen = os.fdopen
    class FailedWriter:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def write(self, data):
            raise OSError(errno.EIO, "PID write failed")
    with monkeypatch.context() as patch:
        patch.setattr(daemon_lock.os, "fdopen", lambda *a: FailedWriter(fdopen(*a)))
        with pytest.raises(OSError, match="PID write failed"):
            acquire_daemon_lock(path)
    handle = acquire_daemon_lock(path)
    assert handle is not None
    release_daemon_lock(handle)
