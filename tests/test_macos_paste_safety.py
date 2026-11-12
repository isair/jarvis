"""Native paste failures keep dictation text without unsafe keyboard fallbacks."""
import time
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.dictation import dictation_engine as module

pytestmark = pytest.mark.unit


@pytest.fixture
def paste_clock(monkeypatch, no_network):
    clock = MagicMock(wraps=time)
    clock.sleep.return_value = None
    monkeypatch.setattr(module, 'time', clock)
    return clock.sleep


@pytest.fixture
def paste_scene(monkeypatch, paste_clock):
    clipboard, pasted = [], []
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(module, '_clipboard_macos', clipboard.append)
    monkeypatch.setattr(module, '_accessibility_warned', False)
    def unsafe_controller():
        raise RuntimeError('unsafe background TSM access')
    monkeypatch.setattr(module, 'pynput_keyboard', SimpleNamespace(Controller=unsafe_controller))
    monkeypatch.setattr(module, '_paste_cgevent', lambda: pasted.append('paste') or True)
    return clipboard, pasted


def test_failed_native_paste_retains_text_and_explains_manual_paste(paste_scene, monkeypatch, capsys):
    clipboard, pasted = paste_scene
    monkeypatch.setattr(module, '_check_macos_accessibility', lambda: True)
    monkeypatch.setattr(module, '_paste_cgevent', lambda: False)
    module._clipboard_paste('fixture dictated text')
    assert clipboard == ['fixture dictated text']
    assert pasted == []
    output = capsys.readouterr().out.lower()
    assert 'clipboard' in output and 'manually' in output
    assert 'fixture dictated text' not in output


def test_permission_denial_stays_effective_until_permission_is_granted(paste_scene, monkeypatch, capsys):
    clipboard, pasted = paste_scene
    trusted = False
    monkeypatch.setattr(module, '_check_macos_accessibility', lambda: trusted)
    module._clipboard_paste('first dictation')
    assert pasted == []
    module._clipboard_paste('second dictation')
    assert pasted == []
    trusted = True
    module._clipboard_paste('third dictation')
    assert pasted == ['paste']
    assert clipboard == ['first dictation', 'second dictation', 'third dictation']
    assert 'manually' in capsys.readouterr().out.lower()


def test_unknown_accessibility_permission_fails_closed(monkeypatch):
    def unavailable_framework(path):
        raise OSError('fixture permission API unavailable')
    monkeypatch.setattr('ctypes.cdll.LoadLibrary', unavailable_framework)
    assert module._check_macos_accessibility() is False


def test_denied_accessibility_opens_settings_only_once(monkeypatch):
    opened = []
    monkeypatch.setattr(module, '_accessibility_warned', False)
    framework = SimpleNamespace(AXIsProcessTrusted=lambda: False)
    monkeypatch.setattr('ctypes.cdll.LoadLibrary', lambda path: framework)
    monkeypatch.setattr('subprocess.Popen', lambda command: opened.append(command))
    assert module._check_macos_accessibility() is False
    assert module._check_macos_accessibility() is False
    assert len(opened) == 1
    assert 'Privacy_Accessibility' in opened[0][-1]


@pytest.fixture
def native_events(monkeypatch, paste_clock):
    posted, released, flags = [], [], {}
    events = {True: 101, False: 102}
    cg = SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda source, key, down: events[down],
        CGEventSetFlags=lambda event, value: flags.update({event: value}),
        CGEventPost=lambda tap, event: posted.append((event, flags[event])),
    )
    cf = SimpleNamespace(CFRelease=lambda event: released.append(event))
    monkeypatch.setattr('ctypes.cdll.LoadLibrary', lambda path: cg if 'CoreGraphics' in path else cf)
    return cg, events, posted, released


@pytest.mark.parametrize('missing_down', [True, False])
def test_incomplete_native_key_pair_posts_no_events(native_events, missing_down):
    _, events, posted, released = native_events
    events[missing_down] = None
    assert module._paste_cgevent() is False
    assert posted == []
    assert set(released) == ({101} if not missing_down else set())


def test_native_key_pair_preserves_command_flags_and_releases_both_events(native_events):
    _, _, posted, released = native_events
    assert module._paste_cgevent() is True
    assert posted == [(101, 0x100000), (102, 0x100000)]
    assert set(released) == {101, 102}


def test_native_event_preparation_failure_posts_nothing_and_releases_both_events(native_events):
    cg, _, posted, released = native_events
    set_flags = cg.CGEventSetFlags
    def fail_on_key_up(event, flags):
        if event == 102:
            raise OSError('fixture event preparation failed')
        set_flags(event, flags)
    cg.CGEventSetFlags = fail_on_key_up
    assert module._paste_cgevent() is False
    assert posted == []
    assert set(released) == {101, 102}


@pytest.mark.parametrize('scene', ['paste_scene', 'native_events'])
def test_paste_waits_do_not_replace_other_workers_sleep(request, paste_clock, scene):
    request.getfixturevalue(scene)
    before = list(paste_clock.call_args_list)
    worker = Thread(target=lambda: time.sleep(0.001))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert paste_clock.call_args_list == before, 'An unrelated worker entered simulated paste waits'
