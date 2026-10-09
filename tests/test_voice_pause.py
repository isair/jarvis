"""Independent voice pause controls preserve intentional dictation and text chat."""
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener

pytestmark = pytest.mark.unit


@pytest.fixture
def voice(mock_config):
    with patch('jarvis.listening.listener.create_intent_judge', return_value=None):
        subject = VoiceListener(None, mock_config, None, None)
    yield subject
    subject.stop()


def test_ending_dictation_does_not_resume_user_paused_voice(voice):
    voice.set_capture_paused('user', True)
    voice.set_capture_paused('dictation', True)
    voice.set_capture_paused('dictation', False)
    voice._on_audio(np.ones((320, 1), dtype=np.float32), 320, None, None)
    assert voice.capture_paused
    assert voice._audio_q.empty()

    voice.set_capture_paused('user', False)
    fresh = np.ones((320, 1), dtype=np.float32) * .2
    voice._on_audio(fresh, 320, None, None)
    assert not voice.capture_paused
    np.testing.assert_array_equal(voice._audio_q.get_nowait().audio, fresh)


def test_resuming_user_pause_does_not_resume_active_dictation(voice):
    voice.set_capture_paused('dictation', True)
    voice.set_capture_paused('user', True)
    voice.set_capture_paused('user', False)
    voice._on_audio(np.ones((320, 1), dtype=np.float32), 320, None, None)
    assert voice.capture_paused
    assert voice._audio_q.empty()


def test_resume_discards_old_follow_up_context(voice):
    voice.state_manager.start_collection('a query before the pause')
    now = time.time()
    voice._transcript_buffer.add('words before the pause', now, now + .1, .1, False)
    assert voice._transcript_buffer.get_all()
    voice.set_capture_paused('user', True)
    voice.set_capture_paused('user', False)
    voice.apply_capture_pause_reset()
    assert not voice.state_manager.is_collecting()
    assert not voice.state_manager.is_hot_window_active()
    assert voice.state_manager.get_pending_query() == ''
    assert voice._transcript_buffer.get_all() == []


@pytest.fixture
def running_voice(voice, monkeypatch):
    import threading
    from jarvis import daemon
    started = threading.Event()
    finish = threading.Event()
    def run_fixture():
        started.set()
        finish.wait(5)
    voice.run = run_fixture
    voice.start()
    assert started.wait(1)
    monkeypatch.setattr(daemon, '_global_voice_listener', voice, raising=False)
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    yield voice
    finish.set()
    voice.join(2)
    assert not voice.is_alive()


def test_core_pause_preserves_running_daemon_and_resumes(running_voice):
    from jarvis import daemon
    assert daemon.set_voice_listening_paused(True) is True
    assert running_voice.is_alive()
    assert not daemon.is_stop_requested()
    assert daemon.set_voice_listening_paused(False) is False
    assert running_voice.is_alive()


def test_subprocess_pause_acknowledges_applied_state(running_voice, capsys):
    import json
    from jarvis import daemon
    request = {'paused': True, 'request_id': 'request-1'}
    assert daemon.handle_voice_pause_stdin_line(daemon.VOICE_PAUSE_IPC_PREFIX + json.dumps(request))
    lines = capsys.readouterr().out.splitlines()
    reply = json.loads(next(line for line in lines if line.startswith(daemon.VOICE_STATUS_IPC_PREFIX))[len(daemon.VOICE_STATUS_IPC_PREFIX):])
    assert reply == {'type': 'status', 'data': {'request_id': 'request-1', 'available': True, 'paused': True}}
    assert running_voice.is_capture_paused('user')


@pytest.mark.parametrize('invalid', ['true', 1, None])
def test_subprocess_pause_requires_a_boolean(running_voice, invalid, capsys):
    import json
    from jarvis import daemon
    request = {'paused': invalid, 'request_id': 'invalid-request'}
    assert daemon.handle_voice_pause_stdin_line(daemon.VOICE_PAUSE_IPC_PREFIX + json.dumps(request))
    assert not running_voice.capture_paused
    assert daemon.VOICE_STATUS_IPC_PREFIX not in capsys.readouterr().out


def test_pause_without_live_voice_does_not_claim_success(monkeypatch):
    from jarvis import daemon
    monkeypatch.setattr(daemon, '_global_voice_listener', None, raising=False)
    assert daemon.set_voice_listening_paused(True) is None


def test_typed_reply_remains_available_while_voice_is_paused(running_voice, mock_config, monkeypatch):
    import threading
    from jarvis import daemon
    from jarvis.memory.conversation import DialogueMemory
    monkeypatch.setattr(daemon, '_global_cfg', mock_config)
    monkeypatch.setattr(daemon, '_global_db', object())
    monkeypatch.setattr(daemon, '_global_dialogue_memory', DialogueMemory(inactivity_timeout=300, max_interactions=20))
    complete = threading.Event()
    replies = []
    def on_complete(reply):
        replies.append(reply)
        complete.set()
    assert daemon.set_voice_listening_paused(True) is True
    with patch('jarvis.reply.engine.run_reply_engine', return_value='Typed answer'):
        daemon.submit_text_query('hello', on_complete=on_complete)
        assert complete.wait(2)
    assert replies == ['Typed answer']
    assert running_voice.capture_paused

@pytest.mark.parametrize('entry', ['timer', 'empty_transcript'])
def test_pause_resume_during_collection_timeout_discards_old_query(voice, monkeypatch, entry):
    from jarvis.reply import engine
    voice.state_manager.start_collection('old query')
    original_clear = voice.state_manager.clear_collection
    def clear_with_pause():
        query = original_clear()
        voice.set_capture_paused('user', True)
        voice.set_capture_paused('user', False)
        return query
    monkeypatch.setattr(voice.state_manager, 'check_collection_timeout', lambda: True)
    monkeypatch.setattr(voice.state_manager, 'clear_collection', clear_with_pause)
    engine_run = Mock(return_value='old answer')
    monkeypatch.setattr(engine, 'run_reply_engine', engine_run)
    if entry == 'timer':
        voice._check_query_timeout()
    else:
        voice._process_transcript('', generation=voice._capture_generation,
                                  captured_during_tts=False, captured_tts_start_time=0)
    assert not engine_run.called


def test_old_speech_completion_cannot_reopen_follow_up_after_resume(voice, monkeypatch):
    from jarvis.reply import engine
    from types import SimpleNamespace
    callbacks = []
    voice.cfg.hot_window_enabled = True
    voice.tts = SimpleNamespace(enabled=True, speak=lambda text, **kw: callbacks.append(kw['completion_callback']))
    monkeypatch.setattr(engine, 'run_reply_engine', lambda *args, **kw: 'An answer')
    voice._dispatch_query('question', generation=voice._capture_generation)
    assert len(callbacks) == 1
    voice.set_capture_paused('user', True)
    voice.set_capture_paused('user', False)
    voice.apply_capture_pause_reset()
    callbacks[0]()
    voice.state_manager.check_hot_window_expiry(False)
    assert not voice.state_manager.is_hot_window_active()
    assert not voice.state_manager.was_speech_during_hot_window(0)


def test_resume_does_not_treat_fresh_audio_as_part_of_old_hot_window(voice):
    voice.state_manager.echo_tolerance = 0
    voice.state_manager.schedule_hot_window_activation(False)
    deadline = time.monotonic() + 1
    while not voice.state_manager.is_hot_window_active() and time.monotonic() < deadline:
        time.sleep(.001)
    assert voice.state_manager.is_hot_window_active()
    voice.set_capture_paused('user', True)
    voice.set_capture_paused('user', False)
    fresh_start = time.time()
    voice.apply_capture_pause_reset()
    assert not voice.state_manager.was_speech_during_hot_window(fresh_start, time.time())
