"""Only the configured shortcut controls dictation capture."""
from dataclasses import dataclass
from enum import Enum

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def shortcut_factory(monkeypatch, tmp_path):
    from jarvis.dictation import dictation_engine
    from jarvis.dictation.history import DictationHistory

    class Key(Enum):
        ctrl = 'control'
        ctrl_l = 'control'
        ctrl_r = 'right-control'
        shift = 'shift'
        shift_l = 'shift'
        shift_r = 'right-shift'
        alt = 'alt'
        alt_l = 'alt'
        alt_r = 'right-alt'
        cmd = 'command'
        cmd_l = 'command'
        cmd_r = 'right-command'

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
    def create(hotkey):
        return dictation_engine.DictationEngine(
            whisper_model_ref=lambda: None,
            whisper_backend_ref=lambda: 'faster-whisper',
            mlx_repo_ref=lambda: None,
            hotkey=hotkey,
            history=DictationHistory(path=tmp_path / 'history.json'),
        ), Keyboard

    return create


@pytest.fixture
def shortcut_engine(shortcut_factory):
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    engine._on_key_press(keyboard.Key.ctrl_l)
    engine._on_key_press(keyboard.Key.shift_l)
    return engine, keyboard


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


@pytest.mark.parametrize('modifier', ['ctrl', 'shift', 'alt', 'cmd'])
@pytest.mark.parametrize('side', ['', '_l', '_r'])
def test_generic_modifier_accepts_either_physical_side(shortcut_factory, modifier, side):
    engine, keyboard = shortcut_factory(f'{modifier}+d')
    key = getattr(keyboard.Key, modifier + side)
    engine._on_key_press(key)
    engine._on_key_press(keyboard.KeyCode(char='d'))
    assert engine.is_recording
    engine._on_key_release(key)
    assert not engine.is_recording


@pytest.mark.parametrize('hotkey,modifiers', [
    ('ctrl+alt', ('ctrl_r', 'alt_r')),
    ('ctrl+cmd', ('ctrl_r', 'cmd_r')),
    ('ctrl+shift', ('ctrl_r', 'shift_r')),
])
def test_modifier_only_shortcut_accepts_right_hand_keys(shortcut_factory, hotkey, modifiers):
    engine, keyboard = shortcut_factory(hotkey)
    keys = [getattr(keyboard.Key, name) for name in modifiers]
    engine._on_key_press(keys[0])
    assert not engine.is_recording
    engine._on_key_press(keys[1])
    assert engine.is_recording
    engine._on_key_release(keys[1])
    assert not engine.is_recording
