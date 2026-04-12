"""Subprocess helper — runs pynput's keyboard listener on the main thread.

macOS 26+ (Tahoe) enforces that TSM (Text Services Manager) calls happen on
the main dispatch queue.  pynput's ``keyboard.Listener`` normally runs on a
background thread, violating this assertion and crashing the process with
SIGTRAP.  By running the listener on the **main thread** of a dedicated
subprocess, the CGEventTap's CFRunLoop sits on the main dispatch queue and
TSM requirements are satisfied.

Protocol (JSON lines on stdout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Outgoing (helper → parent):
  ``{"type": "hotkey_press"}``   — full hotkey combo activated
  ``{"type": "hotkey_release"}`` — a required modifier/trigger released
  ``{"type": "escape"}``         — Escape key pressed
  ``{"type": "ready"}``          — listener started successfully
  ``{"type": "error", "msg": "..."}`` — fatal error

The helper exits when **stdin** is closed (parent died / called stop).
"""

from __future__ import annotations

import json
import os
import sys
import threading

from ._hotkey_shared import (
    all_modifiers_held,
    key_matches,
    normalise_key,
    parse_hotkey,
)


# ── JSON-line helpers ──────────────────────────────────────────────────────

def _send(obj: dict) -> None:
    """Write a JSON line to stdout and flush immediately."""
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        # Parent is gone — exit quietly.
        os._exit(0)


# ── main entry point ──────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        _send({"type": "error", "msg": "usage: _hotkey_helper <hotkey>"})
        sys.exit(1)

    hotkey_str = sys.argv[1]

    try:
        from pynput import keyboard as kb  # noqa: WPS433
    except ImportError:
        _send({"type": "error", "msg": "pynput not installed"})
        sys.exit(1)

    try:
        modifiers, trigger = parse_hotkey(kb, hotkey_str)
    except ValueError as exc:
        _send({"type": "error", "msg": str(exc)})
        sys.exit(1)

    pressed_modifiers: set = set()
    hotkey_active = False

    # ── callbacks ──────────────────────────────────────────────────────

    def on_press(key):
        nonlocal hotkey_active
        nkey = normalise_key(kb, key)

        # Escape
        if getattr(key, "name", None) == "esc" or getattr(nkey, "name", None) == "esc":
            _send({"type": "escape"})
            return

        # Track modifiers currently held
        if any(key_matches(kb, key, nkey, m) for m in modifiers):
            pressed_modifiers.add(nkey if nkey in modifiers else key)

        # Check activation
        mods_held = all_modifiers_held(pressed_modifiers, modifiers)

        if not hotkey_active:
            if trigger is not None:
                if mods_held and key_matches(kb, key, nkey, trigger):
                    hotkey_active = True
                    _send({"type": "hotkey_press"})
            else:
                if mods_held and len(pressed_modifiers) >= len(modifiers):
                    hotkey_active = True
                    _send({"type": "hotkey_press"})
        else:
            # Already active — re-press sends another event (hands-free stop)
            if trigger is not None:
                if mods_held and key_matches(kb, key, nkey, trigger):
                    _send({"type": "hotkey_press"})
            elif mods_held and len(pressed_modifiers) >= len(modifiers):
                _send({"type": "hotkey_press"})

    def on_release(key):
        nonlocal hotkey_active
        nkey = normalise_key(kb, key)

        # Remove from pressed set
        pressed_modifiers.discard(nkey)
        pressed_modifiers.discard(key)
        for m in list(pressed_modifiers):
            if getattr(m, "name", None) == getattr(key, "name", None):
                pressed_modifiers.discard(m)

        # If hotkey was active and a required key released
        if hotkey_active:
            trigger_released = key_matches(kb, key, nkey, trigger)
            modifier_released = any(
                key_matches(kb, key, nkey, m) for m in modifiers
            )
            if trigger_released or modifier_released:
                hotkey_active = False
                _send({"type": "hotkey_release"})

    # ── stdin watcher (exit when parent dies) ──────────────────────────

    def _watch_stdin():
        try:
            sys.stdin.read()
        except Exception:
            pass
        os._exit(0)

    watcher = threading.Thread(target=_watch_stdin, daemon=True)
    watcher.start()

    # ── run the listener on the main thread ────────────────────────────
    # Calling listener.run() directly (instead of listener.start()) keeps
    # the CFRunLoop on the main thread, satisfying macOS TSM requirements.

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    _send({"type": "ready"})
    listener.run()


if __name__ == "__main__":
    main()
