"""
Regression tests for the most-reported desktop crash (#503 and friends).

When the PortAudio shared library is missing (common on Linux),
``import sounddevice`` raises **OSError**, not ImportError. Modules that
treat audio as optional must survive that, otherwise the setup wizard's
dictation page import chain kills the whole app at first launch.

These tests import the modules in a subprocess where ``sounddevice``
raises OSError at import time, and assert the modules still load with
audio disabled (behaviour: app keeps running without audio).
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCENARIO = r"""
import sys, types

sys.path.insert(0, r"{src}")

# Simulate a missing PortAudio library: sounddevice raises OSError on import.
class _Raiser:
    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return None

sys.modules.pop("sounddevice", None)
sys.meta_path.insert(0, _Raiser())

import {module} as m
assert getattr(m, "sd") is None, "sd should be None when PortAudio is missing"

{extra}
print("OK")
"""


def _run(module: str, extra: str = "") -> subprocess.CompletedProcess:
    code = SCENARIO.format(src=str(ROOT / "src"), module=module, extra=extra)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )


def test_dictation_engine_imports_without_portaudio():
    """Dictation engine must load (audio disabled) when PortAudio is absent."""
    extra = (
        "from jarvis.dictation.dictation_engine import format_hotkey_display\n"
        "assert format_hotkey_display('ctrl+alt')"
    )
    result = _run("jarvis.dictation.dictation_engine", extra)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout


def test_listener_imports_without_portaudio():
    """Voice listener module must load (audio disabled) when PortAudio is absent."""
    result = _run("jarvis.listening.listener")
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout
