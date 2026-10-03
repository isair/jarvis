"""Memory digest evals exercise the selected transport and reject hidden failures."""
import importlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from jarvis.reply import enrichment

pytestmark = pytest.mark.unit
IDENTITY = 'TestMemoryDigestSurfacesIdentityFacts'
PREFERENCES = 'TestMemoryDigestSurfacesPreferenceSignals'
CASES = [
    ('identity', IDENTITY, 'test_identity_query_surfaces_user_stated_fact_over_past_qa', 'The user goes boxing near E3 2WS.', True),
    ('identity', IDENTITY, 'test_identity_query_surfaces_multiple_user_facts_when_present', 'The user lives in East London, is vegetarian and is learning Japanese.', True),
    ('identity', IDENTITY, 'test_identity_query_with_only_past_qa_returns_none_or_no_false_facts', 'NONE', False),
    ('identity', IDENTITY, 'test_identity_query_does_not_trigger_recommendation_engagement_rule', 'The user lives in East London and works as a software engineer.', True),
    ('identity', IDENTITY, 'test_recommendation_query_still_surfaces_engagement_when_user_facts_present', 'The user discussed Titanic and Possessor.', True),
    ('preferences', PREFERENCES, 'test_watch_recommendation_surfaces_recently_discussed_films', 'The user discussed Titanic and Possessor.', True),
    ('preferences', PREFERENCES, 'test_restaurant_recommendation_surfaces_past_cuisine_interest', 'The user discussed ramen and Thai curry.', True),
    ('preferences', PREFERENCES, 'test_unrelated_domain_still_returns_none', 'NONE', False),
]


def load_case(monkeypatch, case):
    module, cls, method, _, _ = case
    path = Path(__file__).resolve().parents[1] / 'evals' / f'test_memory_digest_{module}.py'
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, 'conftest', importlib.import_module('evals.conftest'))
        namespace = runpy.run_path(str(path))
    return getattr(namespace[cls](), method)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_digest_eval_reaches_selected_transport(monkeypatch, case, provider):
    run = load_case(monkeypatch, case)
    base, model = 'http://127.0.0.1:11439', 'selected-digest-model'
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    observed = []

    def post(url, **kwargs):
        observed.append((url, kwargs['json']['model']))
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {'message': {'content': case[3]},
                                     'choices': [{'message': {'content': case[3]}}]}
        return response

    monkeypatch.setattr(requests, 'post', post)
    run()
    endpoint = base + ('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert observed and set(observed) == {(endpoint, model)}


@pytest.mark.parametrize('case', [case for case in CASES if case[4]])
@pytest.mark.parametrize('missing', ['', None])
def test_positive_digest_eval_rejects_missing_output(monkeypatch, case, missing):
    run = load_case(monkeypatch, case)
    monkeypatch.setattr(enrichment, 'digest_memory_for_query', lambda **kwargs: missing)
    def unexpected_post(*args, **kwargs):
        pytest.fail('🧠 Missing-output guards must not contact a live model')
    monkeypatch.setattr(requests, 'post', unexpected_post)
    try:
        with pytest.raises(AssertionError):
            run()
    except pytest.xfail.Exception:
        pytest.fail('🧠 Empty digest output must fail a positive eval instead of becoming an xfail')
