"""Tests for the subprocess-isolated pynput keyboard listener.

The pynput Listener has been observed to crash with SIGILL/SIGABRT in C-level
code (CGEventTap on macOS, low-level Win32 hooks on Windows). Running it in a
separate process means a crash kills the child only — the daemon stays up
(issues #252, #353, #354).

These tests exercise the parent-side wrapper without spawning real subprocesses
so they are fast and deterministic, then a smaller integration test verifies
that crash isolation actually holds end-to-end.
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

        original = pk.KeyCode.from_vk(65)  # 'A' on Windows VK
        wire = _serialise_key(original)
        restored = _deserialise_key(wire)
        # Round-trip should preserve vk equality
        assert restored.vk == original.vk

    def test_unknown_kind_deserialises_to_none(self):
        from src.jarvis.dictation.listener_proc import _deserialise_key
        assert _deserialise_key({"kind": "garbage"}) is None
        assert _deserialise_key({}) is None


# ---------------------------------------------------------------------------
# Wrapper lifecycle (mocked subprocess)
# ---------------------------------------------------------------------------

class _FakeConn:
    """Minimal stand-in for a one-way multiprocessing.Pipe end."""

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
        # Real multiprocessing.Pipe.poll returns True for a closed/EOF pipe
        # so the next recv() raises EOFError immediately.
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
    """Stand-in for a multiprocessing.Process with controllable liveness."""

    def __init__(self):
        self._alive = True
        self.terminate_called = False
        self.kill_called = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def die(self) -> None:
        """Simulate the child process crashing."""
        self._alive = False

    def terminate(self) -> None:
        self.terminate_called = True
        self._alive = False

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False

    def join(self, timeout=None) -> None:
        return None


def _install_fake_context(monkeypatch, conn: _FakeConn, proc: _FakeProc):
    """Patch multiprocessing.get_context so the wrapper uses our fakes."""
    fake_ctx = MagicMock()
    fake_ctx.Pipe.return_value = (conn, MagicMock())  # parent_conn, child_conn
    fake_ctx.Process.return_value = proc

    from src.jarvis.dictation import listener_proc
    monkeypatch.setattr(listener_proc.multiprocessing, "get_context", lambda *_a, **_kw: fake_ctx)
    return fake_ctx


class TestSubprocessListenerLifecycle:
    """The wrapper must spawn, relay, stop, and survive child crashes."""

    def test_start_spawns_child_process(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        conn, proc = _FakeConn(), _FakeProc()
        ctx = _install_fake_context(monkeypatch, conn, proc)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            assert ctx.Process.called
            # Child end of pipe must be passed to the subprocess function.
            args, kwargs = ctx.Process.call_args
            assert kwargs.get("daemon") is True
        finally:
            listener.stop()

    def test_relays_press_events_to_callback(self, monkeypatch):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener, _serialise_key

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        seen: list = []
        listener = SubprocessKeyboardListener(
            on_press=lambda k: seen.append(("press", k)),
            on_release=lambda k: seen.append(("release", k)),
        )
        try:
            listener.start()
            conn.send_to_parent({"event": "press", "key": _serialise_key(pk.Key.ctrl_l)})
            conn.send_to_parent({"event": "release", "key": _serialise_key(pk.KeyCode.from_char("d"))})

            deadline = time.time() + 2.0
            while time.time() < deadline and len(seen) < 2:
                time.sleep(0.02)
        finally:
            listener.stop()

        assert ("press", pk.Key.ctrl_l) in seen
        assert any(tag == "release" and getattr(k, "char", None) == "d" for tag, k in seen)

    def test_callback_exception_does_not_kill_reader(self, monkeypatch):
        from pynput import keyboard as pk
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener, _serialise_key

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        deliveries: list = []

        def boom(_key):
            deliveries.append("boom")
            raise RuntimeError("callback exploded")

        def ok(_key):
            deliveries.append("ok")

        listener = SubprocessKeyboardListener(on_press=boom, on_release=ok)
        try:
            listener.start()
            conn.send_to_parent({"event": "press", "key": _serialise_key(pk.Key.ctrl_l)})
            conn.send_to_parent({"event": "release", "key": _serialise_key(pk.Key.ctrl_l)})

            deadline = time.time() + 2.0
            while time.time() < deadline and len(deliveries) < 2:
                time.sleep(0.02)
        finally:
            listener.stop()

        assert deliveries == ["boom", "ok"]

    def test_subprocess_death_stops_reader_without_exception(self, monkeypatch):
        """If the child crashes (SIGILL), the parent must stay up."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            # Simulate the child crashing — no EOF on pipe, just process gone.
            proc.die()
            # Reader thread should exit on its own within a couple of poll cycles.
            deadline = time.time() + 3.0
            reader = listener._reader_thread
            while time.time() < deadline and reader is not None and reader.is_alive():
                time.sleep(0.05)
            assert reader is None or not reader.is_alive(), (
                "reader thread did not exit after subprocess died — "
                "would leak forever and never report the crash"
            )
        finally:
            listener.stop()

    def test_pipe_eof_exits_reader(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        try:
            listener.start()
            conn.signal_eof()
            deadline = time.time() + 3.0
            reader = listener._reader_thread
            while time.time() < deadline and reader is not None and reader.is_alive():
                time.sleep(0.05)
            assert reader is None or not reader.is_alive()
        finally:
            listener.stop()

    def test_stop_terminates_subprocess(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        listener.start()
        listener.stop()

        assert proc.terminate_called

    def test_stop_is_idempotent(self, monkeypatch):
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

        conn, proc = _FakeConn(), _FakeProc()
        _install_fake_context(monkeypatch, conn, proc)

        listener = SubprocessKeyboardListener(on_press=MagicMock(), on_release=MagicMock())
        listener.start()
        listener.stop()
        listener.stop()  # Must not raise.


# ---------------------------------------------------------------------------
# End-to-end crash isolation (real subprocess)
# ---------------------------------------------------------------------------

def _crash_target(conn):
    """Module-level target so multiprocessing.spawn can pickle it."""
    # Simulate a native abort() — same blast radius as a SIGILL from C code.
    os._exit(134)


@pytest.mark.skipif(
    sys.platform == "linux" and not os.environ.get("DISPLAY"),
    reason="pynput Listener requires X11 on Linux",
)
class TestRealSubprocessCrashIsolation:
    """Verify that a child crash actually does not take down the parent."""

    def test_child_crash_does_not_kill_parent(self):
        """Spawn a child that immediately exits abnormally; parent's reader
        thread must exit cleanly and the parent process must keep running."""
        from src.jarvis.dictation.listener_proc import SubprocessKeyboardListener

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
                # Reader thread should detect the dead process and exit.
                deadline = time.time() + 8.0
                reader = listener._reader_thread
                while time.time() < deadline and reader is not None and reader.is_alive():
                    time.sleep(0.1)
                assert reader is None or not reader.is_alive(), (
                    "reader thread did not exit after child died"
                )
            finally:
                listener.stop()

        # Parent process is obviously still alive — that's the whole point.
