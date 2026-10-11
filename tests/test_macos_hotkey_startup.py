"""Hotkey startup keeps event callbacks without background TSM queries."""
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarvis.dictation import dictation_engine as module

pytestmark = pytest.mark.unit


def _engine(monkeypatch, platform, mac_version="15.4"):
    events = []

    class QuartzLoop:
        def _run(self):
            self.on_press("ctrl")
            self.on_release("ctrl")

    class KeyboardListener(QuartzLoop):
        def __init__(self, *, on_press, on_release):
            self.on_press, self.on_release = on_press, on_release
            self.failure = None

        def _run(self):
            if platform == "darwin":
                raise RuntimeError("TSM requires the main queue")
            super()._run()

        def start(self):
            def run():
                try:
                    self._run()
                except RuntimeError as error:
                    self.failure = error
            worker = threading.Thread(target=run)
            worker.start()
            worker.join(2)
            assert not worker.is_alive()

        def stop(self):
            events.append("stopped")

    monkeypatch.setattr(module, "pynput_keyboard", SimpleNamespace(Listener=KeyboardListener))
    monkeypatch.setattr(module, "sd", object())
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.platform, "mac_ver", lambda: (mac_version, (), ""))
    monkeypatch.setitem(sys.modules, "pynput._util.darwin", SimpleNamespace(ListenerMixin=QuartzLoop))
    monkeypatch.setattr(module, "parse_hotkey", lambda hotkey: (frozenset(), None))
    engine = module.DictationEngine(lambda: None, lambda: 'faster-whisper',
                                    lambda: None, hotkey="ctrl", history=Mock())
    engine._on_key_press = lambda key: events.append(("pressed", key))
    engine._on_key_release = lambda key: events.append(("released", key))
    return engine, events


@pytest.mark.parametrize("version", ["14.7.6", "15.4"])
def test_mac_hotkey_starts_without_background_tsm(monkeypatch, version):
    engine, events = _engine(monkeypatch, "darwin", version)
    engine.start()
    assert events == [("pressed", "ctrl"), ("released", "ctrl")]
    engine.stop()
    assert events[-1] == "stopped"


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_other_platforms_retain_keyboard_callbacks(monkeypatch, platform):
    engine, events = _engine(monkeypatch, platform)
    engine.start()
    assert events == [("pressed", "ctrl"), ("released", "ctrl")]
    engine.stop()
    assert events[-1] == "stopped"


def test_macos_26_guard_remains(monkeypatch, capsys):
    engine, events = _engine(monkeypatch, "darwin", "26.6")
    engine.start()
    assert events == []
    assert "not available" in capsys.readouterr().out
