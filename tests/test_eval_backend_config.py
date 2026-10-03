"""Evaluation transport configuration uses the requested local backend."""
from types import SimpleNamespace

import pytest

from evals import helpers

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_eval_environment(monkeypatch):
    for key in ('EVAL_JUDGE_MODEL', 'EVAL_JUDGE_BASE_URL', 'EVAL_JUDGE_PROVIDER'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('provider,url,expected', [
    ('ollama', 'http://127.0.0.1:11439', 'http://127.0.0.1:11439'),
    ('openai_compatible', 'http://127.0.0.1:8000', 'http://127.0.0.1:8000/v1'),
    ('openai_compatible', 'http://127.0.0.1:8000/v1/', 'http://127.0.0.1:8000/v1'),
])
def test_requested_transport_and_endpoint_are_used(monkeypatch, provider, url, expected):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', url)
    cfg = helpers.MockConfig()
    assert cfg.llm_provider == provider
    assert cfg.llm_base_url == expected
    if provider == 'ollama':
        assert cfg.ollama_base_url == expected


def test_model_override_applies_without_an_endpoint_override(monkeypatch):
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'local-test-model')
    cfg = helpers.MockConfig()
    assert cfg.llm_chat_model == cfg.ollama_chat_model == 'local-test-model'


def test_versioned_endpoint_can_detect_an_available_model(monkeypatch):
    import requests
    monkeypatch.setattr(helpers, 'JUDGE_BASE_URL', 'http://127.0.0.1:8000/v1')
    def get(url, **kwargs):
        if url == 'http://127.0.0.1:8000/v1/models':
            return SimpleNamespace(status_code=200, json=lambda: {'data': [{'id': helpers.JUDGE_MODEL}]})
        return SimpleNamespace(status_code=404)
    monkeypatch.setattr(requests, 'get', get)
    assert helpers.is_judge_llm_available()


def test_versioned_endpoint_returns_a_judge_response(monkeypatch):
    import requests
    monkeypatch.setattr(helpers, 'JUDGE_BASE_URL', 'http://127.0.0.1:8000/v1')
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=404))
    def post(url, **kwargs):
        assert url == 'http://127.0.0.1:8000/v1/chat/completions'
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'PASS'}}],
        })
    monkeypatch.setattr(requests, 'post', post)
    assert helpers.call_judge_llm('Evaluate the response.', 'Synthetic example.') == 'PASS'


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_judge_uses_requested_transport_when_both_are_available(monkeypatch, provider):
    import requests
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setattr(helpers, 'JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: SimpleNamespace(status_code=200))
    expected = '/api/chat' if provider == 'ollama' else '/v1/chat/completions'
    def post(url, **kwargs):
        assert url == 'http://127.0.0.1:11439' + expected
        payload = {'message': {'content': 'PASS'}} if provider == 'ollama' else {
            'choices': [{'message': {'content': 'PASS'}}],
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(requests, 'post', post)
    assert helpers.call_judge_llm('Evaluate.', 'Example.') == 'PASS'


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_availability_does_not_fall_back_from_requested_provider(monkeypatch, provider):
    import requests
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setattr(helpers, 'JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    expected = '/api/tags' if provider == 'ollama' else '/v1/models'
    def get(url, **kwargs):
        if url.endswith(expected):
            return SimpleNamespace(status_code=404)
        return SimpleNamespace(status_code=200, json=lambda: {
            'models': [{'name': helpers.JUDGE_MODEL}], 'data': [{'id': helpers.JUDGE_MODEL}],
        })
    monkeypatch.setattr(requests, 'get', get)
    assert not helpers.is_judge_llm_available()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', 'unsupported')
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', 'http://127.0.0.1:11439')
    with pytest.raises(ValueError, match='EVAL_JUDGE_PROVIDER'):
        helpers.MockConfig()


def test_provider_override_applies_to_the_default_endpoint(monkeypatch):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', 'openai_compatible')
    cfg = helpers.MockConfig()
    assert cfg.llm_provider == 'openai_compatible'
    assert cfg.llm_base_url == helpers.JUDGE_BASE_URL.rstrip('/') + '/v1'


@pytest.mark.parametrize('provider,url,expected', [
    ('', '', 'http://localhost:11434/api/chat'),
    ('ollama', 'http://127.0.0.1:11439', 'http://127.0.0.1:11439/api/chat'),
    ('openai_compatible', 'http://127.0.0.1:8000', 'http://127.0.0.1:8000/v1/chat/completions'),
])
def test_planner_eval_reaches_the_selected_backend(monkeypatch, provider, url, expected):
    """The eval must generate a plan, rather than silently skip an unset model."""
    import requests
    from jarvis.reply.planner import plan_query
    if provider:
        monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    if url:
        monkeypatch.setenv('EVAL_JUDGE_BASE_URL', url)
    monkeypatch.setattr(helpers, 'JUDGE_BASE_URL', 'http://localhost:11434')
    monkeypatch.setattr(helpers, 'JUDGE_MODEL', 'synthetic-planner-model')
    def post(endpoint, **kwargs):
        assert endpoint == expected
        assert kwargs['json']['model'] == helpers.JUDGE_MODEL
        payload = kwargs['json']
        sampling = payload.get('options', payload)
        assert sampling['temperature'] == 0.0
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {
            'message': {'content': 'Reply to the user.'},
            'choices': [{'message': {'content': 'Reply to the user.'}}],
        }
        return response
    monkeypatch.setattr(requests, 'post', post)
    assert plan_query(helpers.planner_config(), 'What is two plus two?', '', []) == [
        'Reply to the user.',
    ]


@pytest.mark.parametrize('provider,url,expected', [
    ('ollama', 'http://127.0.0.1:11439', 'http://127.0.0.1:11439/api/chat'),
    ('openai_compatible', 'http://127.0.0.1:8000/v1', 'http://127.0.0.1:8000/v1/chat/completions'),
])
@pytest.mark.parametrize('entry', ['single', 'multi', 'listener', 'processed'])
def test_voice_evals_reach_selected_backend(monkeypatch, provider, url, expected, entry):
    import requests
    import json
    from unittest.mock import MagicMock
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', url)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', 'synthetic-voice-model')
    monkeypatch.setattr(requests, 'get', lambda *args, **kw: SimpleNamespace(status_code=404))
    monkeypatch.setattr('desktop_app.face_widget.get_jarvis_state', lambda: MagicMock())
    calls = []
    content = json.dumps(dict(directed=True, query=('random topic' if entry == 'processed' else 'weather'), stop=False,
                             confidence='high', reasoning='Addressed to the assistant'))
    def post(endpoint, **kwargs):
        calls.append((endpoint, kwargs['json']['model']))
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {
            'message': {'content': content},
            'choices': [{'message': {'content': content}}],
        }
        return response
    monkeypatch.setattr(requests, 'post', post)
    from evals import test_intent_judge as intent, test_listener_integration as listener_eval
    if entry == 'processed':
        monkeypatch.setattr(intent, 'is_intent_judge_available', lambda: True)
        try:
            intent.TestProcessedSegmentFiltering().test_processed_segment_not_reextracted()
        except pytest.skip.Exception:
            pytest.fail('An available selected voice model must be evaluated')
        import os
        assert calls == [(expected, os.environ['EVAL_JUDGE_MODEL'])]
        return
    if entry == 'single':
        result = intent.run_intent_judge(intent.INTENT_JUDGE_TEST_CASES[0])
    elif entry == 'multi':
        result = intent.run_intent_judge_multi_segment(intent.MULTI_SEGMENT_TEST_CASES[0])
    else:
        obj, _ = listener_eval._create_listener()
        try:
            result = obj._intent_judge.judge([intent.create_transcript_segment('Jarvis weather')])
        finally:
            obj.state_manager.stop()
    assert result is not None and result.query == 'weather'
    import os
    assert calls == [(expected, os.environ['EVAL_JUDGE_MODEL'])]
