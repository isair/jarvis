"""Saved runtime controls reach assistant configuration and spoken actions."""
import json
from unittest.mock import MagicMock

import pytest

from jarvis.config import get_config_path, get_default_config, load_settings
from jarvis.listening.listener import VoiceListener

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def settings(**overrides):
    saved = dict(db_path=':memory:', vad_enabled=False, tune_enabled=False, hot_window_enabled=False, **overrides)
    get_config_path().write_text(json.dumps(saved), encoding='utf-8')
    return load_settings()


def test_declared_configuration_is_available_to_runtime_consumers():
    assert set(get_default_config()) <= set(vars(load_settings()))


@pytest.mark.parametrize('key', [
    'llm_thinking_enabled', 'intent_judge_thinking_enabled', 'dictation_thinking_enabled',
])
@pytest.mark.parametrize('enabled', [False, True])
def test_saved_thinking_switches_are_independent(key, enabled):
    keys = ['llm_thinking_enabled', 'intent_judge_thinking_enabled', 'dictation_thinking_enabled']
    saved = {name: not enabled for name in keys}
    saved[key] = enabled
    cfg = settings(**saved)
    assert {name: getattr(cfg, name) for name in keys} == saved


class SpeakingTTS:
    enabled = True

    def __init__(self):
        self.speaking = True

    def set_playback_callbacks(self, started, finished):
        self.playback_callbacks = started, finished

    def preparing_request_at(self, capture_time):
        return None

    def is_speaking(self):
        return self.speaking

    def interrupt(self):
        self.speaking = False


def reply_keeps_playing(cfg, heard, spoken='The forecast is sunny today.'):
    tts = SpeakingTTS()
    listener = VoiceListener(MagicMock(), cfg, tts, MagicMock())
    try:
        listener.echo_detector.track_tts_start(spoken)
        start = listener.echo_detector._tts_start_time
        listener._process_transcript(
            heard, utterance_energy=0.005, utterance_start_time=start + 0.2,
            utterance_end_time=start + 1, captured_during_tts=True,
            captured_tts_start_time=start, generation=listener._capture_generation, language=None,
        )
        return tts.speaking
    finally:
        listener.state_manager.stop()


@pytest.mark.parametrize('command', ['ruhig', 'توقف', '停止', 'σιωπή'])
def test_saved_multilingual_stop_control_interrupts_the_reply(command):
    assert not reply_keeps_playing(settings(stop_commands=[command]), command)


def test_saved_empty_control_list_disables_text_interrupts():
    assert reply_keeps_playing(settings(stop_commands=[]), 'stop')


@pytest.mark.parametrize('ratio,keeps_playing', [(0.95, True), (0.75, False)])
def test_saved_fuzzy_threshold_controls_approximate_interrupts(ratio, keeps_playing):
    assert reply_keeps_playing(settings(stop_commands=['stop'], stop_command_fuzzy_ratio=ratio),
                              'stopp') is keeps_playing


@pytest.mark.parametrize('ratio,keeps_playing', [(0.95, True), (0.75, False)])
def test_saved_fuzzy_threshold_preserves_echo_control_priority(ratio, keeps_playing):
    assert reply_keeps_playing(settings(stop_commands=['stop'], stop_command_fuzzy_ratio=ratio),
                              'stopp', 'The command sounds like stopp.') is keeps_playing


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('enabled', [False, True])
def test_saved_intent_thinking_switch_reaches_the_local_request(monkeypatch, enabled, provider):
    from contextlib import nullcontext
    from jarvis.llm import http
    from jarvis.listening.intent_judge import create_intent_judge
    from jarvis.listening.transcript_buffer import TranscriptSegment

    cfg = settings(llm_provider=provider, llm_base_url='http://127.0.0.1:1',
                   ollama_base_url='http://127.0.0.1:1', intent_judge_thinking_enabled=enabled,
                   llm_thinking_enabled=not enabled)

    def post(url, **kwargs):
        payload = kwargs['json']
        if provider == 'ollama':
            assert url == cfg.ollama_base_url + '/api/chat'
            assert payload['think'] is enabled
        else:
            assert url == cfg.llm_base_url + '/chat/completions'
            assert 'think' not in payload
        assert 'What is the weather, Jarvis?' in payload['messages'][-1]['content']
        response = MagicMock()
        content = json.dumps({'directed': True, 'stop': False, 'confidence': 'high',
                              'reasoning': 'A question.'})
        response.json.return_value = (
            {'message': {'content': content}} if provider == 'ollama' else
            {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        )
        return nullcontext(response)

    monkeypatch.setattr(http, 'post', post)
    judge = create_intent_judge(cfg)
    result = judge.judge([TranscriptSegment('What is the weather, Jarvis?', 1, 2)],
                         current_text='What is the weather, Jarvis?')
    assert result is not None and result.directed and not result.stop
