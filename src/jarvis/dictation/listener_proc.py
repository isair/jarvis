"""Subprocess-isolated pynput keyboard listener.

The pynput Listener has been observed to crash with SIGILL/SIGABRT in C-level
code (CGEventTap on macOS, low-level Win32 hooks on Windows). Running it in a
separate process means a crash kills the child only — the daemon stays up and
dictation just stops working until the next restart, instead of taking down the
whole app (issues #252, #353, #354).

The wrapper exposes a minimal ``Listener``-like API (``start`` / ``stop``) so
the engine can treat it as a drop-in replacement.

Architecture::

    DictationEngine               child subprocess
    ───────────────               ─────────────────
    SubprocessKeyboardListener
      ├─ start() ──spawn──▶  pynput.Listener
      ├─ reader thread ◀─pipe──  on_press / on_release
      │   └─ on_press / on_release callbacks
      └─ stop() ──terminate──▶  exit

Key events are serialised to plain dicts on the wire so neither end has to
unpickle pynput objects across the process boundary.
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
    """Convert a pynput Key/KeyCode to a JSON-safe dict.

    pynput is imported lazily so this module loads in environments where it
    isn't installed (e.g. unit tests that only exercise the deserialiser).
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

    Returns ``None`` if the payload is unrecognised so callers can skip it
    rather than crash on garbled input.
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
        try:
            conn.send({"event": "error", "message": repr(exc)})
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
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_listener_main,
            args=(child_conn,),
            daemon=True,
        )
        self._proc.start()
        # Only the child writes; close the child end in the parent so an EOF
        # is delivered if the child exits without sending anything else.
        try:
            child_conn.close()
        except Exception:
            pass

        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="dictation-listener-reader",
        )
        self._reader_thread.start()
        debug_log("dictation listener subprocess started", "dictation")

    def stop(self) -> None:
        self._stop_flag.set()

        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
            except Exception as exc:
                debug_log(f"dictation listener subprocess teardown error: {exc}", "dictation")
        self._proc = None

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        reader = self._reader_thread
        if reader is not None and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=2.0)
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
