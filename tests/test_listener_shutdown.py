"""Listener exit releases audible feedback and scheduled engagement."""
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from jarvis.listening import listener as voice

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('fails', [False, True])
def test_listener_exit_closes_thinking_stream_and_cancels_follow_up(monkeypatch, mock_config, fails):
    started, closed = Event(), Event()
    class Stream:
        def __init__(self, **kwargs):
            pass
        def start(self):
            started.set()
        def close(self):
            closed.set()
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Stream))
    monkeypatch.setattr(voice, 'create_intent_judge', lambda cfg: None)
    monkeypatch.setattr(voice, 'webrtcvad', None)
    mock_config.tune_enabled = True
    mock_config.echo_tolerance = 60
    instance = voice.VoiceListener(None, mock_config, None, None)
    players = []
    def listen():
        instance._start_thinking_tune()
        players.append(instance._tune_player)
        assert started.wait(2), 'Synthetic output stream did not start'
        instance.state_manager.schedule_hot_window_activation()
        assert instance.state_manager.was_speech_during_hot_window(0)
        if fails:
            raise RuntimeError('intent processing failed')
    monkeypatch.setattr(instance, '_run', listen)
    try:
        if fails:
            with pytest.raises(RuntimeError, match='intent processing failed'):
                instance.run()
        else:
            instance.run()
        assert closed.is_set(), 'Listener exited with a thinking output stream still open'
        assert not players[0].is_playing(), 'Listener exited while its thinking sound was playing'
        assert not instance.state_manager.was_speech_during_hot_window(0)
    finally:
        instance.stop()
