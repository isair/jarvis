"""Only the configured shortcut controls dictation capture."""
from dataclasses import dataclass
from enum import Enum

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def shortcut_engine(monkeypatch, tmp_path):
    from jarvis.dictation import dictation_engine
    from jarvis.dictation.history import DictationHistory

    class Key(Enum):
        ctrl_l = 'control'
        shift_l = 'shift'
        alt_l = 'alt'
        cmd = 'command'

    @dataclass(frozen=True)
    class KeyCode:
        char: str | None = None
        vk: int | None = None

        @staticmethod
        def from_char(character):
            return KeyCode(char=character)

    class Keyboard:
        pass

    Keyboard.Key, Keyboard.KeyCode = Key, KeyCode

    class DeferredWorker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(dictation_engine, 'pynput_keyboard', Keyboard)
    monkeypatch.setattr(dictation_engine.threading, 'Thread', DeferredWorker)
    engine = dictation_engine.DictationEngine(
        whisper_model_ref=lambda: None,
        whisper_backend_ref=lambda: 'faster-whisper',
        mlx_repo_ref=lambda: None,
        hotkey='ctrl+shift+d',
        history=DictationHistory(path=tmp_path / 'history.json'),
    )
    engine._on_key_press(Key.ctrl_l)
    engine._on_key_press(Key.shift_l)
    return engine, Keyboard


@pytest.mark.parametrize('kind', ['character', 'virtual'])
def test_unrelated_key_cannot_start_capture(shortcut_engine, kind):
    engine, keyboard = shortcut_engine
    unrelated = keyboard.KeyCode(char='x') if kind == 'character' else keyboard.KeyCode(vk=99)
    engine._on_key_press(unrelated)
    assert not engine.is_recording


@pytest.mark.parametrize('kind', ['character', 'virtual'])
def test_unrelated_release_cannot_end_capture(shortcut_engine, kind):
    engine, keyboard = shortcut_engine
    engine._on_key_press(keyboard.KeyCode(char='d'))
    assert engine.is_recording
    unrelated = keyboard.KeyCode(char='x') if kind == 'character' else keyboard.KeyCode(vk=99)
    engine._on_key_release(unrelated)
    assert engine.is_recording


@pytest.mark.parametrize('released', ['trigger', 'control', 'shift'])
def test_configured_key_release_ends_capture(shortcut_engine, released):
    engine, keyboard = shortcut_engine
    engine._on_key_press(keyboard.KeyCode(char='D'))
    assert engine.is_recording
    key = {
        'trigger': keyboard.KeyCode(char='D'),
        'control': keyboard.Key.ctrl_l,
        'shift': keyboard.Key.shift_l,
    }[released]
    engine._on_key_release(key)
    assert not engine.is_recording
