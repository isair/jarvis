"""Only the configured shortcut controls dictation capture."""
from dataclasses import dataclass
from enum import Enum
from threading import Event, Thread

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


@pytest.fixture
def listener_factory(monkeypatch):
    from unittest.mock import Mock
    from jarvis.dictation import dictation_engine

    class Listener:
        def __init__(self, on_press, on_release):
            self.on_press, self.on_release = on_press, on_release

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(dictation_engine, '_create_keyboard_listener', Listener)
    monkeypatch.setattr(dictation_engine, 'sd', Mock())
    monkeypatch.setattr(dictation_engine.platform, 'mac_ver', lambda: ('15.0', ('', '', ''), ''))

    def start(engine):
        engine.start()
        return engine._listener

    return start


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


@pytest.mark.parametrize('hotkey,other_modifier', [
    ('ctrl+alt', 'alt_l'),
    ('ctrl+shift+d', 'shift_l'),
])
@pytest.mark.parametrize('first,last', [('ctrl_l', 'ctrl_r'), ('ctrl_r', 'ctrl_l')])
def test_capture_continues_while_the_other_modifier_side_is_held(
        shortcut_factory, hotkey, other_modifier, first, last):
    engine, keyboard = shortcut_factory(hotkey)
    for name in (other_modifier, 'ctrl_l', 'ctrl_r'):
        engine._on_key_press(getattr(keyboard.Key, name))
    engine._on_key_press(keyboard.KeyCode(char='d'))
    assert engine.is_recording
    engine._on_key_release(getattr(keyboard.Key, first))
    assert engine.is_recording
    engine._on_key_release(getattr(keyboard.Key, last))
    assert not engine.is_recording


@pytest.fixture
def timed_shortcut(shortcut_factory, listener_factory, monkeypatch):
    from unittest.mock import Mock
    from jarvis.dictation import dictation_engine

    clock = {'wall': 1700000000.0, 'steady': 100.0}
    monkeypatch.setattr(dictation_engine.time, 'time', lambda: clock['wall'])
    monkeypatch.setattr(dictation_engine.time, 'monotonic', lambda: clock['steady'])
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    engine._whisper_model_ref = lambda: object()
    backend = Mock()
    backend.query_devices.return_value = {'default_samplerate': engine._target_sample_rate}
    monkeypatch.setattr(dictation_engine, 'sd', backend)
    monkeypatch.setattr(dictation_engine, '_play_beep', lambda *args: None)
    monkeypatch.setattr(dictation_engine, 'resolve_input_device', lambda *args: {'device': 0})
    monkeypatch.setattr(dictation_engine, 'open_input_stream',
                        lambda *args, **kwargs: (Mock(), engine._target_sample_rate, 1))
    listener_factory(engine)

    def tap():
        listener = engine._listener
        listener.on_press(keyboard.KeyCode(char='d'))
        engine._begin_recording(engine._session)
        clock['wall'] += 0.1
        clock['steady'] += 0.1
        listener.on_release(keyboard.KeyCode(char='d'))

    return engine, keyboard, clock, tap


@pytest.mark.parametrize('origin', [0.0, 100.0])
@pytest.mark.parametrize('gap,clock_adjustment,continues', [
    (0.1, 0, True),
    (3.0, 0, False),
    (0.1, 3600, True),
    (3.0, -3600, False),
])
def test_double_tap_uses_elapsed_time_despite_system_clock_adjustment(
        timed_shortcut, origin, gap, clock_adjustment, continues):
    engine, keyboard, clock, tap = timed_shortcut
    clock['steady'] = origin
    engine._listener.on_press(keyboard.Key.ctrl_l)
    engine._listener.on_press(keyboard.Key.shift_l)
    tap()
    assert not engine.is_recording
    clock['wall'] += gap + clock_adjustment
    clock['steady'] += gap
    tap()
    assert engine.is_recording is continues


def test_restarting_engine_does_not_retain_held_keys(shortcut_factory, listener_factory):
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    listener = listener_factory(engine)
    for key in (keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.KeyCode(char='d')):
        listener.on_press(key)
    assert engine.is_recording
    engine.stop()
    assert not engine.is_recording
    listener = listener_factory(engine)
    listener.on_press(keyboard.KeyCode(char='d'))
    assert not engine.is_recording
    for key in (keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.KeyCode(char='d')):
        listener.on_press(key)
    assert engine.is_recording


def test_retired_listener_cannot_begin_capture_after_restart(shortcut_factory, listener_factory):
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    retired = listener_factory(engine)
    engine.stop()
    listener_factory(engine)
    for key in (keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.KeyCode(char='d')):
        retired.on_press(key)
    assert not engine.is_recording


def test_retired_listener_cannot_end_new_capture(shortcut_factory, listener_factory):
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    retired = listener_factory(engine)
    engine.stop()
    active = listener_factory(engine)
    for key in (keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.KeyCode(char='d')):
        active.on_press(key)
    assert engine.is_recording
    retired.on_release(keyboard.KeyCode(char='d'))
    assert engine.is_recording
    active.on_release(keyboard.KeyCode(char='d'))
    assert not engine.is_recording


def test_double_tap_cannot_span_an_engine_restart(timed_shortcut, listener_factory):
    engine, keyboard, clock, tap = timed_shortcut
    engine._listener.on_press(keyboard.Key.ctrl_l)
    engine._listener.on_press(keyboard.Key.shift_l)
    tap()
    assert not engine.is_recording
    engine.stop()
    listener_factory(engine)
    clock['wall'] += 0.1
    clock['steady'] += 0.1
    engine._listener.on_press(keyboard.Key.ctrl_l)
    engine._listener.on_press(keyboard.Key.shift_l)
    tap()
    assert not engine.is_recording


def test_stop_serialises_with_an_in_flight_listener_event(
        shortcut_factory, listener_factory, monkeypatch):
    engine, keyboard = shortcut_factory('ctrl+shift+d')
    listener = listener_factory(engine)
    listener.on_press(keyboard.Key.ctrl_l)
    listener.on_press(keyboard.Key.shift_l)
    entered, release, stopping, stopped = Event(), Event(), Event(), Event()
    errors = []
    handle = engine._on_key_press

    def paused(key):
        entered.set()
        assert release.wait(5), 'Test did not release the paused key event'
        handle(key)

    monkeypatch.setattr(engine, '_on_key_press', paused)

    def press():
        try:
            listener.on_press(keyboard.KeyCode(char='d'))
        except Exception as error:
            errors.append(error)

    def stop():
        stopping.set()
        engine.stop()
        stopped.set()

    worker, stopper = Thread(target=press), Thread(target=stop)
    worker.start()
    try:
        assert entered.wait(5), 'The key event did not start'
        stopper.start()
        assert stopping.wait(5), 'Stop was not requested'
        assert not stopped.wait(0.05), 'Stop returned before the active key event finished'
    finally:
        release.set()
        worker.join(5)
        if stopper.ident is not None:
            stopper.join(5)
    assert not worker.is_alive() and not stopper.is_alive()
    assert not errors
    assert not engine.is_recording
