"""Max-turn replies survive reasoning before their grounded caveat."""
from unittest.mock import MagicMock

import pytest
import requests

from evals.helpers import voice_config
from jarvis.reply.enrichment import digest_loop_for_max_turns

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('thinking', [False, True])
def test_max_turn_digest_returns_complete_caveated_reply(monkeypatch, provider, thinking):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'selected-fast-digest-model')
    cfg = voice_config()
    cfg.llm_chat_model = 'separate-chat-model'
    cfg.llm_thinking_enabled = thinking
    reply = "I couldn't finish this request. London is 12 degrees with rain, but no forecast was obtained."
    required_tokens = 300 + (len(reply) + 3) // 4

    def post(url, **kwargs):
        payload = kwargs['json']
        assert payload['model'] == cfg.fast_model
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        content = reply if cap >= required_tokens else ''
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': content},
                                     'choices': [{'message': {'content': content}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    messages = [{'role': 'tool', 'name': 'getWeather', 'content': 'London: 12 degrees with rain.'}]
    assert digest_loop_for_max_turns('Check London weather and tomorrow.', messages, cfg) == reply
