"""Tool digests retain their facts after a backend's reasoning tokens."""
from unittest.mock import MagicMock

import pytest
import requests

from evals.helpers import voice_config
from jarvis.reply.enrichment import _TOOL_DIGEST_MIN_CHARS, digest_tool_result_for_query

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('thinking', [False, True])
def test_tool_digest_preserves_facts_after_reasoning(monkeypatch, provider, thinking):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'selected-tool-digest-model')
    cfg = voice_config()
    snippet = 'Kyoto station reports 19.5 degrees Celsius. '
    payload = snippet * ((_TOOL_DIGEST_MIN_CHARS + len(snippet) - 1) // len(snippet))
    note = 'The page reports that Kyoto station measured 19.5 degrees Celsius.'
    required_tokens = 300 + (len(note) + 3) // 4

    def post(url, **kwargs):
        wire = kwargs['json']
        assert wire['model'] == cfg.llm_chat_model
        cap = wire.get('max_tokens', wire.get('options', {}).get('num_predict', 0))
        content = note if cap >= required_tokens else ''
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': content},
                                     'choices': [{'message': {'content': content}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    result = digest_tool_result_for_query(
        'What temperature does Kyoto station report?', 'fetchWebPage', payload,
        cfg, cfg.llm_chat_model, thinking=thinking,
    )
    assert result == note
