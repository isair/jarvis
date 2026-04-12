"""Shared hotkey-matching utilities for the dictation engine and subprocess helper.

These functions are used by both ``dictation_engine.py`` (in-process listener)
and ``_hotkey_helper.py`` (subprocess listener on macOS 26+).  Keeping them in
a single module avoids duplication and prevents the two implementations from
drifting apart.
"""

from __future__ import annotations

from typing import Any


# ── modifier name → pynput Key attribute name ──────────────────────────────

MODIFIER_MAP = {
    "ctrl": "ctrl_l",
    "shift": "shift_l",
    "alt": "alt_l",
    "cmd": "cmd",
    "super": "cmd",
}


# ── hotkey parsing ─────────────────────────────────────────────────────────

def parse_hotkey(kb: Any, combo: str):
    """Parse a hotkey string like ``'ctrl+shift+d'`` into pynput key objects.

    *kb* is the ``pynput.keyboard`` module.

    Returns ``(frozenset_of_modifiers, trigger_key_or_None)``.
    Modifier-only combos (e.g. ``'ctrl+cmd'``) are valid — *trigger* is
    ``None`` and the hotkey activates when all modifiers are held.
    """
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey string")

    modifiers: set = set()
    trigger = None

    for part in parts:
        mapped = MODIFIER_MAP.get(part)
        if mapped:
            key_obj = getattr(kb.Key, mapped, None)
            if key_obj is not None:
                modifiers.add(key_obj)
        else:
            if len(part) == 1:
                trigger = kb.KeyCode.from_char(part)
            else:
                key_obj = getattr(kb.Key, part, None)
                if key_obj is not None:
                    trigger = key_obj
                else:
                    raise ValueError(f"unknown key: {part}")

    if not modifiers and trigger is None:
        raise ValueError("hotkey must contain at least one key")

    return frozenset(modifiers), trigger


# ── key comparison helpers ─────────────────────────────────────────────────

def normalise_key(kb: Any, key: Any) -> Any:
    """Normalise a key event for comparison against parsed trigger/modifiers."""
    if hasattr(key, "char") and key.char is not None:
        return kb.KeyCode.from_char(key.char.lower())
    return key


def key_matches(kb: Any, key: Any, nkey: Any, target: Any) -> bool:
    """Check whether *key* (raw) / *nkey* (normalised) matches *target*."""
    if target is None:
        return False
    if nkey == target or key == target:
        return True
    if getattr(key, "name", None) == getattr(target, "name", None):
        return True
    if hasattr(key, "char") and key.char:
        if kb.KeyCode.from_char(key.char.lower()) == target:
            return True
    return False


def all_modifiers_held(pressed: set, modifiers: frozenset) -> bool:
    """Return True when every required modifier is currently pressed."""
    return all(
        m in pressed or any(
            getattr(p, "name", None) == getattr(m, "name", None)
            for p in pressed
        )
        for m in modifiers
    )
