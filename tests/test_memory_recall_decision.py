"""Explicit recall decisions control search parameters without guessing intent."""
import json

import pytest

from jarvis.reply import enrichment

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('decision', [False, True, None, 'false'])
def test_explicit_no_recall_discards_search_parameters(monkeypatch, decision):
    response = {
        'from': '2030-01-01T00:00:00Z', 'to': '2030-01-02T00:00:00Z',
        'keywords': ['fixture topic'], 'questions': ['fixture personal question'],
    }
    if decision is not None:
        response['recall_required'] = decision
    monkeypatch.setattr(enrichment, 'call_llm_direct', lambda **kwargs: json.dumps(response))
    result = enrichment.extract_search_params_for_memory('fixture request', object(), 'fixture model')
    if decision is False:
        assert result == {'from': None, 'to': None, 'keywords': [], 'questions': []}
    else:
        assert result == {key: value for key, value in response.items() if key != 'recall_required'}
