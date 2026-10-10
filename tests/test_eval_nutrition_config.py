"""Nutrition evaluations exercise the selected production extraction path."""
from unittest.mock import MagicMock

import pytest

from evals.helpers import MockConfig

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_eval_environment(monkeypatch):
    for key in ('EVAL_JUDGE_MODEL', 'EVAL_JUDGE_BASE_URL', 'EVAL_JUDGE_PROVIDER'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('answer', ['meal', 'NONE', '```text\nNONE\n```'])
def test_nutrition_eval_uses_selected_transport_and_production_fence(monkeypatch, provider, answer):
    from evals.test_nutrition_extraction import call_nutrition_extraction
    import requests
    import json

    cfg = MockConfig(llm_provider=provider, llm_base_url='http://127.0.0.1:1/v1',
                     ollama_base_url='http://127.0.0.1:2',
                     llm_chat_model='selected-nutrition-model',
                     ollama_chat_model='obsolete-model', llm_chat_timeout_sec=7.3)
    expected = {'description': 'eggs', 'calories_kcal': 150, 'protein_g': 12}
    content = json.dumps(expected) if answer == 'meal' else answer
    observed = []
    def post(url, **kwargs):
        observed.append((url, kwargs))
        response = MagicMock()
        response.__enter__.return_value = response
        message = {'content': content}
        response.json.return_value = ({'message': message} if provider == 'ollama'
                                     else {'choices': [{'message': message}]})
        return response
    monkeypatch.setattr(requests, 'post', post)

    result = call_nutrition_extraction(cfg, 'I had eggs')
    assert observed, 'Extraction did not reach the selected backend'
    url, kwargs = observed[0]
    payload = kwargs['json']
    assert payload['model'] == cfg.llm_chat_model
    expected_url = (cfg.ollama_base_url + '/api/chat' if provider == 'ollama'
                    else cfg.llm_base_url + '/chat/completions')
    assert url == expected_url
    assert kwargs['timeout'] == cfg.llm_chat_timeout_sec
    prompt = payload['messages'][-1]['content']
    assert '<<<BEGIN UNTRUSTED USER TEXT>>>\nI had eggs' in prompt
    assert '<<<END UNTRUSTED USER TEXT>>>' in prompt
    sampling = payload.get('options', payload)
    assert ('num_predict' if provider == 'ollama' else 'max_tokens') in sampling
    assert (result and {key: result[key] for key in expected}) == (expected if answer == 'meal' else None)


def test_empty_inference_is_not_successful_non_food_rejection(monkeypatch):
    from evals.test_nutrition_extraction import call_nutrition_extraction
    import requests
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {'message': {'content': ''}}
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: response)
    cfg = MockConfig(llm_provider='ollama', llm_chat_model='local-model')
    with pytest.raises(AssertionError, match='Empty'):
        call_nutrition_extraction(cfg, 'I went for a walk')


@pytest.mark.parametrize('answer', ['{invalid', 'null'])
def test_invalid_inference_is_not_successful_non_food_rejection(monkeypatch, answer):
    from evals.test_nutrition_extraction import call_nutrition_extraction
    import requests
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {'message': {'content': answer}}
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: response)
    cfg = MockConfig(llm_provider='ollama', llm_chat_model='local-model')
    with pytest.raises(AssertionError, match='Invalid'):
        call_nutrition_extraction(cfg, 'I went for a walk')
