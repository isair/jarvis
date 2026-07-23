"""
Regression tests for the most-reported desktop crash (#503 and friends).

When the PortAudio shared library is missing (common on Linux),
``import sounddevice`` raises **OSError**, not ImportError. Similarly,
``pynput`` can raise non-ImportError exceptions on headless Linux (no X
display). Modules that treat audio as optional must survive both,
otherwise the setup wizard's dictation page import chain kills the whole
app at first launch.

These tests import the modules in a subprocess where the dependency
raises at import time, and assert the modules still load with the
feature disabled (behaviour: app keeps running without audio).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIO = r"""
import sys

sys.path.insert(0, r"{src}")

# Simulate a broken optional dependency raising at import time.
class _Raiser:
    def find_spec(self, name, path=None, target=None):
        if name == "{dep}":
            raise {error}
        return None

sys.modules.pop("{dep}", None)
sys.meta_path.insert(0, _Raiser())

import {module} as m
assert getattr(m, "{attr}") is None, "{attr} should be None when {dep} is broken"

{extra}
print("OK")
"""


def _run(module: str, dep: str, error: str, attr: str, extra: str = "") -> subprocess.CompletedProcess:
    code = SCENARIO.format(src=str(ROOT / "src"), module=module, dep=dep, error=error, attr=attr, extra=extra)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout


def test_dictation_engine_imports_without_portaudio():
    """Dictation engine must load (audio disabled) when PortAudio is absent."""
    extra = (
        "from jarvis.dictation.dictation_engine import format_hotkey_display\n"
        "assert format_hotkey_display('ctrl+alt')"
    )
    result = _run(
        "jarvis.dictation.dictation_engine",
        dep="sounddevice",
        error='OSError("PortAudio library not found")',
        attr="sd",
        extra=extra,
    )
    _assert_ok(result)


def test_dictation_engine_imports_with_broken_pynput():
    """Dictation engine must load (hotkey disabled) when pynput fails to import."""
    result = _run(
        "jarvis.dictation.dictation_engine",
        dep="pynput",
        error='RuntimeError("no X display available")',
        attr="pynput_keyboard",
    )
    _assert_ok(result)


def test_listener_imports_without_portaudio():
    """Voice listener module must load (audio disabled) when PortAudio is absent."""
    result = _run(
        "jarvis.listening.listener",
        dep="sounddevice",
        error='OSError("PortAudio library not found")',
        attr="sd",
    )
    _assert_ok(result)
