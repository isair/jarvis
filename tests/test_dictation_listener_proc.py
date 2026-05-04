"""Tests for the subprocess-isolated pynput keyboard listener.

The wrapper exists because pynput's Listener has crashed with SIGILL/SIGABRT
in C-level code (CGEventTap on macOS, low-level Win32 hooks on Windows),
and those signals can't be caught from Python. Running it in a separate
process keeps a native crash from taking down the daemon.

These tests exercise the parent-side wrapper without spawning real
subprocesses so they are fast and deterministic, then a smaller integration
test verifies that crash isolation actually holds end-to-end.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# Skip the whole module if pynput isn't installed (CI without GUI deps).
pytest.importorskip("pynput")


# ---------------------------------------------------------------------------
# Key serialisation round-trips
# ---------------------------------------------------------------------------

class TestKeySerialisation:
    """Pynput key/keycode objects must survive a process boundary as plain dicts."""

    def test_named_key_round_trip(self):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import _serialise_key, _deserialise_key

        original = pk.Key.ctrl_l
        wire = _serialise_key(original)
        restored = _deserialise_key(wire)
        assert restored == original

    def test_char_keycode_round_trip(self):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import _serialise_key, _deserialise_key

        original = pk.KeyCode.from_char("a")
        wire = _serialise_key(original)
        restored = _deserialise_key(wire)
        assert restored == original

    def test_vk_only_keycode_round_trip(self):
        """Modifier combos sometimes give KeyCode with vk but char=None."""
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import _serialise_key, _deserialise_key

        original = pk.KeyCode.from_vk(65)
        wire = _serialise_key(original)
        restored = _deserialise_key(wire)
        assert restored.vk == original.vk

    def test_unknown_kind_deserialises_to_none(self):
        from src.jarvis.dictation.listener_proc import _deserialise_key
        assert _deserialise_key({"kind": "garbage"}) is None
        assert _deserialise_key({}) is None


# ---------------------------------------------------------------------------
# Test doubles for spawn / Pipe
# ---------------------------------------------------------------------------

class _FakeConn:
    """One-way Pipe end stand-in."""

    def __init__(self):
        self._inbox: list = []
        self._lock = threading.Lock()
        self._closed = False
        self._eof = False

    def send_to_parent(self, msg) -> None:
        with self._lock:
            self._inbox.append(msg)

    def signal_eof(self) -> None:
        with self._lock:
            self._eof = True

    def poll(self, timeout: float = 0.0) -> bool:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._inbox or self._eof or self._closed:
                    return True
            if time.time() >= deadline:
                return False
            time.sleep(0.01)

    def recv(self):
        with self._lock:
            if self._inbox:
                return self._inbox.pop(0)
            if self._eof:
                raise EOFError
            raise OSError("recv on empty closed pipe")

    def close(self) -> None:
        with self._lock:
            self._closed = True


class _FakeProc:
    """multiprocessing.Process stand-in with controllable liveness."""

    def __init__(self):
        self._alive = False
        self.terminate_called = False
        self.kill_called = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def die(self) -> None:
        self._alive = False

    def terminate(self) -> None:
        self.terminate_called = True
        self._alive = False

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False

    def join(self, timeout=None) -> None:
        return None


def _install_pluggable_fake_context(monkeypatch):
    """Patch ``multiprocessing.get_context`` to hand out fresh fakes per spawn.

    Returns a list that captures every (conn, proc) pair the wrapper allocates,
    so tests can drive specific generations.
    """
    spawned: list[tuple[_FakeConn, _FakeProc]] = []

    def make_pair():
        conn, child_conn = _FakeConn(), _FakeConn()
        proc = _FakeProc()
        spawned.append((conn, proc))
        return conn, child_conn, proc

    fake_ctx = MagicMock()

    pending: list[tuple[_FakeConn, _FakeConn, _FakeProc]] = []

    def pipe_side_effect(*_a, **_kw):
        triple = make_pair()
        pending.append(triple)
        return triple[0], triple[1]

    def process_side_effect(*_a, **_kw):
        return pending.pop(0)[2]

    fake_ctx.Pipe.side_effect = pipe_side_effect
    fake_ctx.Process.side_effect = process_side_effect

    from src.jarvis.dictation import listener_proc
    monkeypatch.setattr(listener_proc.multiprocessing, "get_context", lambda *_a, **_kw: fake_ctx)
    return spawned


def _accelerate_supervisor(monkeypatch):
    """Shrink retry caps and backoff so respawn behaviour is testable in <1 s."""
    from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener
    monkeypatch.setattr(SubprocessKeyboardListener, "_BACKOFF_SCHEDULE", (0.0,))
    monkeypatch.setattr(SubprocessKeyboardListener, "_CLEAN_RUN_WINDOW", 30.0)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Lifecycle (mocked subprocess)
# ---------------------------------------------------------------------------

class TestSubprocessListenerLifecycle:
    def test_start_spawns_child_process(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            assert _wait_until(lambda: len(spawned) >= 1)
            conn, proc = spawned[0]
            assert proc.is_alive()
        finally:
            listener.stop()

    def test_relays_press_and_release_events(self, monkeypatch):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener, _serialise_key

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        seen: list = []
        listener = SubprocessKeyboardListener(
            on_press=lambda k: seen.append(("press", k)),
            on_release=lambda k: seen.append(("release", k)),
        )
        try:
            listener.start()
            assert _wait_until(lambda: spawned and spawned[0][0] is not None)
            conn, _proc = spawned[0]

            conn.send_to_parent({"event": "press", "key": _serialise_key(pk.Key.ctrl_l)})
            conn.send_to_parent({"event": "release", "key": _serialise_key(pk.KeyCode.from_char("d"))})

            assert _wait_until(lambda: len(seen) >= 2)
        finally:
            listener.stop()

        assert ("press", pk.Key.ctrl_l) in seen
        assert any(tag == "release" and getattr(k, "char", None) == "d" for tag, k in seen)

    def test_callback_exception_does_not_kill_reader(self, monkeypatch):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener, _serialise_key

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        deliveries: list = []

        def boom(_key):
            deliveries.append("boom")
            raise RuntimeError("callback exploded")

        listener = SubprocessKeyboardListener(on_press=boom, on_release=lambda _k: deliveries.append("ok"))
        try:
            listener.start()
            assert _wait_until(lambda: spawned and spawned[0][0] is not None)
            conn, _proc = spawned[0]
            conn.send_to_parent({"event": "press", "key": _serialise_key(pk.Key.ctrl_l)})
            conn.send_to_parent({"event": "release", "key": _serialise_key(pk.Key.ctrl_l)})

            assert _wait_until(lambda: len(deliveries) >= 2)
        finally:
            listener.stop()

        assert deliveries == ["boom", "ok"]

    def test_error_event_does_not_trigger_respawn(self, monkeypatch):
        """A child-side error payload is logged but the live child stays alive."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            assert _wait_until(lambda: spawned and spawned[0][0] is not None)
            conn, proc = spawned[0]
            conn.send_to_parent({"event": "error", "message": "transient pynput hiccup"})

            time.sleep(0.1)
            assert proc.is_alive()
            assert len(spawned) == 1  # no respawn
        finally:
            listener.stop()

    def test_stop_terminates_subprocess(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        listener.start()
        assert _wait_until(lambda: spawned and spawned[0][1].is_alive())
        listener.stop()

        _conn, proc = spawned[0]
        assert proc.terminate_called
        assert listener._supervisor_thread is None

    def test_stop_is_idempotent(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        listener.start()
        listener.stop()
        listener.stop()  # Must not raise.


# ---------------------------------------------------------------------------
# Auto-respawn (the whole point of the supervisor)
# ---------------------------------------------------------------------------

class TestAutoRespawn:
    def test_subprocess_death_triggers_respawn(self, monkeypatch):
        """When the child dies, the supervisor must spawn a fresh child."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            assert _wait_until(lambda: spawned and spawned[0][1].is_alive())
            spawned[0][1].die()  # simulate native crash

            # Supervisor should respawn — wait for a second generation.
            assert _wait_until(lambda: len(spawned) >= 2, timeout=3.0)
            assert spawned[1][1].is_alive()
        finally:
            listener.stop()

    def test_gives_up_after_max_consecutive_failures(self, monkeypatch):
        """If every spawned child dies immediately, the supervisor must stop
        spawning after the cap so we don't burn CPU in a crash loop."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            # Kill each generation as it appears, never letting any run for the
            # clean-window threshold.
            seen = 0
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if len(spawned) > seen:
                    spawned[seen][1].die()
                    seen += 1
                if not (
                    listener._supervisor_thread is not None
                    and listener._supervisor_thread.is_alive()
                ):
                    break
                time.sleep(0.02)
        finally:
            listener.stop()

        # Cap is 3 consecutive failures, so we expect at most 3 spawn attempts
        # before the supervisor gives up.
        assert len(spawned) == SubprocessKeyboardListener._MAX_CONSECUTIVE_FAILURES, (
            f"expected exactly {SubprocessKeyboardListener._MAX_CONSECUTIVE_FAILURES} "
            f"spawn attempts before give-up, saw {len(spawned)}"
        )

    def test_clean_long_run_resets_failure_counter(self, monkeypatch):
        """A spawn that ran for longer than ``_CLEAN_RUN_WINDOW`` is treated as
        the start of a fresh failure series."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        monkeypatch.setattr(SubprocessKeyboardListener, "_BACKOFF_SCHEDULE", (0.0,))
        # A very small clean-window so the test runs fast.
        monkeypatch.setattr(SubprocessKeyboardListener, "_CLEAN_RUN_WINDOW", 0.05)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            # Let the first child run past the clean window before killing it.
            assert _wait_until(lambda: spawned and spawned[0][1].is_alive())
            time.sleep(0.1)
            spawned[0][1].die()

            assert _wait_until(lambda: len(spawned) >= 2, timeout=2.0)
            # Failure counter should have been reset by the clean run, so it is
            # at 1 (this latest death), not 2.
            assert listener._failure_count == 1
        finally:
            listener.stop()

    def test_spawn_failure_counts_toward_give_up(self, monkeypatch):
        """If ``proc.start()`` raises every time, give up after the cap."""
        from src.jarvis.dictation import listener_proc
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        attempts = {"n": 0}

        class _AlwaysFailingProc:
            def start(self):
                attempts["n"] += 1
                raise OSError("nope")

            def is_alive(self):
                return False

            def terminate(self):
                pass

            def kill(self):
                pass

            def join(self, timeout=None):
                pass

        fake_ctx = MagicMock()
        fake_ctx.Pipe.side_effect = lambda *_a, **_kw: (_FakeConn(), _FakeConn())
        fake_ctx.Process.side_effect = lambda *_a, **_kw: _AlwaysFailingProc()
        monkeypatch.setattr(
            listener_proc.multiprocessing, "get_context", lambda *_a, **_kw: fake_ctx
        )
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            # Wait for supervisor to exit (give-up).
            deadline = time.time() + 3.0
            while time.time() < deadline:
                supervisor = listener._supervisor_thread
                if supervisor is None or not supervisor.is_alive():
                    break
                time.sleep(0.02)
        finally:
            listener.stop()

        assert attempts["n"] == SubprocessKeyboardListener._MAX_CONSECUTIVE_FAILURES

    def test_restart_after_give_up_resets_state(self, monkeypatch):
        """After giving up, calling ``start()`` again must allow a fresh
        attempt — the user has presumably fixed whatever caused the loop."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        spawned = _install_pluggable_fake_context(monkeypatch)
        _accelerate_supervisor(monkeypatch)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            # First session: kill every child until give-up.
            listener.start()
            seen = 0
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if len(spawned) > seen:
                    spawned[seen][1].die()
                    seen += 1
                supervisor = listener._supervisor_thread
                if supervisor is None or not supervisor.is_alive():
                    break
                time.sleep(0.02)
            assert len(spawned) == SubprocessKeyboardListener._MAX_CONSECUTIVE_FAILURES

            # Second session: should attempt again from a clean state.
            listener.start()
            assert _wait_until(
                lambda: len(spawned) > SubprocessKeyboardListener._MAX_CONSECUTIVE_FAILURES,
                timeout=3.0,
            )
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Error-message truncation
# ---------------------------------------------------------------------------

class TestErrorMessageTruncation:
    def test_oversized_error_message_is_truncated(self):
        from src.jarvis.dictation.listener_proc import _MAX_ERROR_MESSAGE_LEN, _listener_main

        sent: list = []

        class _CapturingConn:
            def send(self, msg):
                sent.append(msg)

        import builtins
        original_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "pynput" or name.startswith("pynput."):
                raise RuntimeError("x" * 5000)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = boom
        try:
            _listener_main(_CapturingConn())
        finally:
            builtins.__import__ = original_import

        assert sent and sent[0]["event"] == "error"
        assert len(sent[0]["message"]) <= _MAX_ERROR_MESSAGE_LEN + len("...[truncated]")


# ---------------------------------------------------------------------------
# End-to-end crash isolation (real subprocess)
# ---------------------------------------------------------------------------

def _crash_target(conn):
    """Module-level target so multiprocessing.spawn can pickle it."""
    os._exit(134)


@pytest.mark.skipif(
    sys.platform == "linux" and not os.environ.get("DISPLAY"),
    reason="pynput Listener requires X11 on Linux",
)
class TestRealSubprocessCrashIsolation:
    """Verify that a child crash actually does not take down the parent."""

    def test_child_crash_triggers_respawn_then_give_up(self, monkeypatch):
        """A child that crashes immediately should be respawned, then the
        supervisor should give up cleanly without ever taking the parent down."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        # Tighter cap and zero backoff so the test finishes promptly.
        monkeypatch.setattr(SubprocessKeyboardListener, "_BACKOFF_SCHEDULE", (0.0,))

        with patch(
            "src.jarvis.dictation.listener_proc._listener_main",
            new=_crash_target,
        ):
            listener = SubprocessKeyboardListener(
                on_press=MagicMock(),
                on_release=MagicMock(),
            )
            try:
                listener.start()
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    supervisor = listener._supervisor_thread
                    if supervisor is None or not supervisor.is_alive():
                        break
                    time.sleep(0.1)
                supervisor = listener._supervisor_thread
                assert supervisor is None or not supervisor.is_alive(), (
                    "supervisor never gave up after repeated child crashes"
                )
            finally:
                listener.stop()
