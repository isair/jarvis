"""Reasoning-capable backends can return complete relevant memory digests."""
from unittest.mock import MagicMock

import pytest
import requests

from evals.helpers import voice_config
from jarvis.reply.enrichment import _DIGEST_MIN_CHARS, digest_memory_for_query

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('thinking', [False, True])
def test_memory_digest_retains_relevant_note_after_reasoning(monkeypatch, provider, thinking):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'selected-memory-digest-model')
    cfg = voice_config()
    snippet = 'The user enjoys ramen and Thai curry. '
    entry = snippet * ((_DIGEST_MIN_CHARS + len(snippet) - 1) // len(snippet))
    note = 'The user enjoys ramen and Thai curry, useful preferences for dinner.'
    required_tokens = 300 + (len(note) + 3) // 4

    def post(url, **kwargs):
        payload = kwargs['json']
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        content = note if cap >= required_tokens else ''
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': content},
                                     'choices': [{'message': {'content': content}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    result = digest_memory_for_query(
        'Recommend dinner for me.', [entry], [], cfg, cfg.llm_chat_model, thinking=thinking,
    )
    assert result == note
