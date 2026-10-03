"""Recall extraction leaves room for reasoning and complete parameter JSON."""
import json
from unittest.mock import MagicMock

import pytest
import requests

from evals.helpers import voice_config
from jarvis.reply.enrichment import extract_search_params_for_memory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('thinking', [False, True])
def test_recall_extraction_returns_complete_parameters_after_reasoning(monkeypatch, provider, thinking):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'selected-recall-model')
    cfg = voice_config()
    params = {'keywords': ['food', 'cooking', 'nutrition'],
              'from': '2026-10-02T00:00:00Z', 'to': '2026-10-02T23:59:59Z'}
    content = json.dumps(params)
    required_tokens = 300 + (len(content) + 3) // 4

    def post(url, **kwargs):
        payload = kwargs['json']
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        reply = content if cap >= required_tokens else ''
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': reply},
                                     'choices': [{'message': {'content': reply}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    result = extract_search_params_for_memory(
        'What did I eat yesterday?', cfg, cfg.fast_model, thinking=thinking,
    )
    assert result == params
