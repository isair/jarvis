"""Timer preparation preserves activity and playback remains independent of Qt."""
import importlib.abc
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from jarvis import activity
from jarvis.output import tts as speech
from jarvis.tools.builtin import timer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tts_available', [False, True])
@pytest.mark.parametrize('backend', [speech.PiperTTS, speech.ChatterboxTTS])
def test_timer_activity_does_not_require_a_desktop(monkeypatch, tmp_path, tts_available, backend, make_speech_backend):
    engine = make_speech_backend(backend)
    for name in tuple(sys.modules):
        if name.split('.')[0] in {'desktop_app', 'PyQt6'}:
            monkeypatch.delitem(sys.modules, name)
    class RejectGUI(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'desktop_app', 'PyQt6'}:
                raise AssertionError('Timer announcement imported the GUI')
    monkeypatch.setattr(sys, 'meta_path', [RejectGUI(), *sys.meta_path])
    path = tmp_path / 'state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    state = activity.JarvisStateManager()
    monkeypatch.setattr(activity, '_jarvis_state_instance', state)
    state.set_state(activity.JarvisState.THINKING)
    monkeypatch.setattr(timer, '_tts_provider', lambda: engine if tts_available else None)
    monkeypatch.setattr(timer, 'get_alarm_registry', lambda: SimpleNamespace(start=lambda entry: None))
    now = datetime.now(timezone.utc)
    entry = timer.TimerEntry('owned', 'tea', 60, now, now, 'Tea is ready.')
    outcomes = []
    timer._default_announcer(entry, outcomes.append)
    assert state.state is activity.JarvisState.THINKING
    if tts_available:
        engine._run()
        assert state.state is activity.JarvisState.IDLE
        assert engine.get_last_spoken_text() == 'Tea is ready.'
        assert outcomes == ['completed']
    else:
        assert outcomes == ['rejected']
        assert engine.get_last_spoken_text() == ''
    assert not any(name.split('.')[0] in {'desktop_app', 'PyQt6'} for name in sys.modules)
