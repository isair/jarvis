"""Automatic paste releases modifiers and retains a usable manual recovery."""
from types import SimpleNamespace

import pytest

from jarvis.dictation import dictation_engine

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('system', ['Windows', 'Linux'])
@pytest.mark.parametrize('failure', [None, 'missing', 'controller', 'press', 'tap', 'release'],
                         ids=['success', 'missing-backend', 'controller', 'modifier-press', 'paste', 'modifier-release'])
def test_paste_failure_retains_clipboard_and_releases_modifier(monkeypatch, capsys, system, failure):
    monkeypatch.setenv('JARVIS_VOICE_DEBUG', '1')
    monkeypatch.setattr(dictation_engine.platform, 'system', lambda: system)
    monkeypatch.setattr(dictation_engine.time, 'sleep', lambda _: None)
    text = 'Synthetic dictated text.'
    copied, inserted, held = [], [], set()
    monkeypatch.setattr(dictation_engine, '_clipboard_windows' if system == 'Windows' else '_clipboard_linux',
                        copied.append)
    keys = SimpleNamespace(ctrl='ctrl', ctrl_l='ctrl_l', ctrl_r='ctrl_r', alt_l='alt_l', alt_r='alt_r',
                           shift_l='shift_l', shift_r='shift_r', cmd='cmd', cmd_r='cmd_r')

    class SyntheticController:
        def press(self, key):
            held.add(key)
            if failure == 'press':
                raise RuntimeError('private-paste-error-marker')

        def tap(self, key):
            if failure == 'tap':
                raise RuntimeError('private-paste-error-marker')
            if key == 'v' and 'ctrl' in held:
                inserted.extend(copied)

        def release(self, key):
            held.discard(key)
            if failure == 'release' and key == 'ctrl':
                raise RuntimeError('private-paste-error-marker')

    def controller():
        if failure == 'controller':
            raise RuntimeError('private-paste-error-marker')
        return SyntheticController()

    monkeypatch.setattr(dictation_engine, 'pynput_keyboard',
                        None if failure == 'missing' else SimpleNamespace(Controller=controller, Key=keys))
    dictation_engine._clipboard_paste(text)
    assert copied == [text]
    assert held == set()
    output = capsys.readouterr()
    assert 'private-paste-error-marker' not in output.out + output.err
    if failure is None:
        assert inserted == [text]
        assert '⚠️' not in output.out
    else:
        assert 'automatic paste' in output.out
        assert 'clipboard' in output.out and 'Ctrl+V' in output.out and 'if needed' in output.out
        assert '⚠️' in output.out
