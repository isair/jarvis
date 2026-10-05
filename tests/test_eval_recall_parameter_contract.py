"""Existing memory extractor evals reach the selected local transport."""
import importlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from evals.helpers import MockConfig

pytestmark = pytest.mark.unit
ENTRIES = [
    ('test_enrichment_extracts_correct_keywords', {'query': 'what news might interest me?', 'expected_keywords': ['interests']}),
    ('test_enrichment_extracts_correct_keywords', {'query': 'what did we discuss about the python project?', 'expected_keywords': ['python']}),
    ('test_enrichment_extracts_correct_keywords', {'query': 'what did I eat yesterday?', 'expected_keywords': ['food']}),
    ('test_enrichment_skips_questions_answered_by_context', {}),
]


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_extractor_eval_uses_the_selected_backend(monkeypatch, entry, provider):
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    path = Path(__file__).resolve().parents[1] / 'evals' / 'test_agent_behavior.py'
    monkeypatch.syspath_prepend(str(path.parent))
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, 'conftest', importlib.import_module('evals.conftest'))
        cases = runpy.run_path(str(path))
    base = 'http://127.0.0.1:11439'
    model = 'selected-memory-model'
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    observed = []

    def post(url, **kwargs):
        observed.append((url, kwargs['json']['model']))
        response = MagicMock()
        response.__enter__.return_value = response
        answer = '{"keywords": ["interests", "python", "food"], "questions": []}'
        response.json.return_value = {'message': {'content': answer},
                                     'choices': [{'message': {'content': answer}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    name, arguments = entry
    getattr(cases['TestMemoryEnrichment'](), name)(mock_config=MockConfig(), **arguments)
    endpoint = base + ('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert observed and set(observed) == {(endpoint, model)}
