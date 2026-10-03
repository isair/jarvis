"""Daily summaries can emit both diary text and topics after reasoning."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.memory.conversation import generate_conversation_summary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_diary_answer_and_topics_survive_local_reasoning(monkeypatch, provider):
    cfg = SimpleNamespace(llm_provider=provider, llm_base_url='http://127.0.0.1:1/v1',
                          ollama_base_url='http://127.0.0.1:1', llm_chat_model='local-reasoning-model')
    summary, topics = 'The user prefers Celsius and lives in Hackney.', 'preferences, temperature, location'
    answer = f'SUMMARY: {summary}\nTOPICS: {topics}'
    reasoning = ' '.join(['Review the dialogue and preserve attribution.'] * 100)
    required = len(reasoning.split()) + len(answer.split())
    timeout = 7.3
    observed_timeouts = []
    def post(url, **kwargs):
        observed_timeouts.append(kwargs['timeout'])
        payload = kwargs['json']
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        message = {'content': answer if cap >= required else '', 'reasoning': reasoning}
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = ({'message': message} if provider == 'ollama'
                                     else {'choices': [{'message': message}]})
        return response
    monkeypatch.setattr('requests.post', post)
    result = generate_conversation_summary(['User: I prefer Celsius.','User: I live in Hackney.'],
                                           None, cfg, timeout_sec=timeout)
    assert result == (summary, topics)
    assert observed_timeouts and all(value == timeout for value in observed_timeouts)
