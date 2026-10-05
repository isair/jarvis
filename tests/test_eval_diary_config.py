"""Diary hygiene evaluations use the configured production summariser."""
from unittest.mock import MagicMock

import pytest

from evals.test_diary_summariser_hygiene import TestDiarySummariserHygieneLive as DiaryCases

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_diary_hygiene_eval_reaches_selected_model_and_transport(monkeypatch, provider):
    import requests
    from evals import helpers
    base = 'http://127.0.0.1:1'
    model = 'selected-diary-model'
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    observed = []
    summary, topics = 'The user prefers Celsius.', 'temperatures, preference'
    def post(url, **kwargs):
        observed.append((url, kwargs))
        response = MagicMock()
        response.__enter__.return_value = response
        message = {'content': f'SUMMARY: {summary}\nTOPICS: {topics}'}
        response.json.return_value = ({'message': message} if provider == 'ollama'
                                     else {'choices': [{'message': message}]})
        return response
    monkeypatch.setattr(requests, 'post', post)
    assert DiaryCases()._summarise(['User: I prefer Celsius.']) == (summary, topics)
    assert observed
    url, kwargs = observed[0]
    assert url == base + ('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert kwargs['json']['model'] == helpers.voice_config().llm_chat_model
    assert 'User: I prefer Celsius.' in kwargs['json']['messages'][-1]['content']


@pytest.mark.parametrize('answer', ['', 'SUMMARY: A summary without topics'])
def test_empty_or_incomplete_diary_inference_cannot_pass_hygiene(monkeypatch, answer):
    import requests
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', 'ollama')
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'selected-diary-model')
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {'message': {'content': answer}}
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: response)
    with pytest.raises(AssertionError, match='Empty or incomplete'):
        DiaryCases()._summarise(['User: I prefer Celsius.'])
