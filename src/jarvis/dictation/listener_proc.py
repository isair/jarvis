"""Subprocess-isolated pynput keyboard listener.

The pynput Listener owns native event taps (CGEventTap on macOS, low-level
Win32 hooks on Windows) that have crashed with SIGILL/SIGABRT in C-level
code. Those signals can't be caught from Python, so a single bad event tears
down the whole daemon. Running the Listener in a separate process means a
crash kills the child only; the daemon stays up and dictation goes offline
silently until the engine is restarted.

The wrapper exposes a minimal ``Listener``-like API (``start`` / ``stop``)
so the engine can treat it as a drop-in replacement.

Privacy and trust model
-----------------------
Every keystroke (press and release) is forwarded over a local
``multiprocessing.Pipe`` so the parent can detect hotkey combos. The pipe is
an OS-level handle owned exclusively by the parent and the spawned child;
no other process can read it. Wire payloads contain only key identity
(``name`` / ``char`` / ``vk``), never window context or user content, and
``debug_log`` calls in this module never include key data. The child is our
own spawn and is therefore trusted on the receive side.
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import Any, Callable, Optional

from ..debug import debug_log


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def _serialise_key(key: Any) -> dict:
    """Convert a pynput Key/KeyCode to a plain dict for cross-process transport.

    pynput is imported lazily so this module loads in environments where it
    isn't installed (e.g. unit tests that only exercise the deserialiser).
    The output never contains window context or user content, only key
    identity.
    """
    from pynput import keyboard as pk

    if isinstance(key, pk.Key):
        return {"kind": "named", "name": key.name}

    char = getattr(key, "char", None)
    vk = getattr(key, "vk", None)
    if char is not None:
        return {"kind": "char", "char": char, "vk": vk}
    if vk is not None:
        return {"kind": "vk", "vk": vk}
    return {"kind": "unknown"}


def _deserialise_key(payload: dict) -> Optional[Any]:
    """Reconstruct a pynput Key/KeyCode from the wire dict.

    The payload originates from a child process we spawned ourselves over a
    private OS pipe, so it is trusted. ``getattr`` is used with a known enum
    (``pk.Key``) and a default of ``None``, so unknown names cannot resolve
    to arbitrary attributes. Returns ``None`` for unrecognised payloads so
    callers can skip them rather than crash on garbled input.
    """
    if not isinstance(payload, dict):
        return None

    kind = payload.get("kind")
    try:
        from pynput import keyboard as pk
    except ImportError:
        return None

    if kind == "named":
        return getattr(pk.Key, payload.get("name", ""), None)
    if kind == "char":
        char = payload.get("char")
        if char is None:
            return None
        return pk.KeyCode.from_char(char)
    if kind == "vk":
        vk = payload.get("vk")
        if vk is None:
            return None
        return pk.KeyCode.from_vk(vk)
    return None


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

_MAX_ERROR_MESSAGE_LEN = 200


def _listener_main(conn: Any) -> None:
    """Run the real pynput Listener and emit events on *conn*.

    Top-level so ``multiprocessing.spawn`` can pickle and re-import it. If
    pynput's C-level event tap aborts the process, only this child dies; the
    parent's reader thread notices via ``proc.is_alive()`` and exits cleanly.
    """
    try:
        from pynput import keyboard as pk

        def on_press(key):
            try:
                conn.send({"event": "press", "key": _serialise_key(key)})
            except Exception:
                # Parent pipe closed — nothing to do; listener will be torn
                # down when the parent terminates the process.
                pass

        def on_release(key):
            try:
                conn.send({"event": "release", "key": _serialise_key(key)})
            except Exception:
                pass

        listener = pk.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()
    except Exception as exc:
        # Truncate so a noisy traceback message can't flood the parent's debug
        # log if pynput ever decides to dump file paths or stack frames.
        message = repr(exc)
        if len(message) > _MAX_ERROR_MESSAGE_LEN:
            message = message[:_MAX_ERROR_MESSAGE_LEN] + "...[truncated]"
        try:
            conn.send({"event": "error", "message": message})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class SubprocessKeyboardListener:
    """Drop-in replacement for ``pynput.keyboard.Listener``.

    Spawns a child process that owns the real listener and relays key events
    over a pipe. A reader thread in the parent calls the supplied callbacks.
    Crashes in the child do not propagate — the reader exits and dictation
    silently goes offline until the engine is restarted.
    """

    _POLL_INTERVAL = 0.5  # seconds between liveness checks

    def __init__(
        self,
        on_press: Callable[[Any], None],
        on_release: Callable[[Any], None],
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._proc: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def start(self) -> None:
        if self._proc is not None:
            return

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_listener_main,
            args=(child_conn,),
            daemon=True,
        )

        # The spawn re-execs the bundled exe in frozen builds; on a starved
        # system that can fail. Treat any failure as "dictation offline" and
        # leave the wrapper in its initial state so the caller can retry.
        try:
            proc.start()
        except Exception as exc:
            debug_log(f"failed to spawn dictation listener: {exc}", "dictation")
            self._safe_close(parent_conn)
            self._safe_close(child_conn)
            return

        self._conn = parent_conn
        self._proc = proc

        # Only the child writes; close the child end in the parent so an EOF
        # is delivered if the child exits without sending anything else.
        self._safe_close(child_conn)

        try:
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="dictation-listener-reader",
            )
            self._reader_thread.start()
        except Exception as exc:
            # Out-of-resources for thread creation — abandon the child rather
            # than leak it.
            debug_log(f"failed to start dictation listener reader: {exc}", "dictation")
            self._reader_thread = None
            self.stop()
            return

        debug_log("dictation listener subprocess started", "dictation")

    @staticmethod
    def _safe_close(conn: Any) -> None:
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass

    # Total worst-case teardown is ~2.5 s; daemon=True backstops anything
    # the OS still has alive after that.
    _TERMINATE_JOIN_TIMEOUT = 1.0
    _KILL_JOIN_TIMEOUT = 0.5
    _READER_JOIN_TIMEOUT = 1.0

    def stop(self) -> None:
        self._stop_flag.set()

        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=self._TERMINATE_JOIN_TIMEOUT)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=self._KILL_JOIN_TIMEOUT)
            except Exception as exc:
                debug_log(f"dictation listener subprocess teardown error: {exc}", "dictation")
        self._proc = None

        self._safe_close(self._conn)
        self._conn = None

        reader = self._reader_thread
        if reader is not None and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=self._READER_JOIN_TIMEOUT)
        self._reader_thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        conn = self._conn
        while not self._stop_flag.is_set():
            if conn is None:
                return

            try:
                has_data = conn.poll(self._POLL_INTERVAL)
            except (EOFError, OSError):
                debug_log("dictation listener pipe closed", "dictation")
                return

            if not has_data:
                proc = self._proc
                if proc is not None and not proc.is_alive():
                    debug_log(
                        "dictation listener subprocess exited unexpectedly "
                        "(pynput crashed?) — dictation disabled until restart",
                        "dictation",
                    )
                    return
                continue

            try:
                msg = conn.recv()
            except (EOFError, OSError):
                debug_log("dictation listener pipe closed", "dictation")
                return

            if not isinstance(msg, dict):
                continue

            event = msg.get("event")
            if event == "press":
                self._dispatch(self._on_press, msg.get("key"))
            elif event == "release":
                self._dispatch(self._on_release, msg.get("key"))
            elif event == "error":
                debug_log(
                    f"dictation listener subprocess reported error: {msg.get('message')}",
                    "dictation",
                )

    @staticmethod
    def _dispatch(callback: Callable[[Any], None], key_payload: Any) -> None:
        key = _deserialise_key(key_payload) if isinstance(key_payload, dict) else None
        if key is None:
            return
        try:
            callback(key)
        except Exception as exc:
            debug_log(f"dictation listener callback error: {exc}", "dictation")
