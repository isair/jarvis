"""The daemon single-instance lock — the guard against the "two voices" bug
(two daemons alive, each with its own listener + TTS, both answering)."""

import importlib
import os

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
    daemon = pytest.importorskip("jarvis.daemon")
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "jarvis_daemon.lock"))
    ran = []
    monkeypatch.setattr(daemon, "_run_daemon", lambda smoke_test=False: ran.append(smoke_test))

    held = acquire_daemon_lock()  # a first daemon, mid-shutdown
    daemon.request_stop()
    try:
        daemon.main()
        assert ran == [], "a second daemon must not start while one holds the lock"
        assert daemon._global_stop_requested is True, "the stopping daemon's flag must not be reset"
    finally:
        release_daemon_lock(held)

    daemon.main()  # the lock is free again: starts normally
    assert ran == [False]
