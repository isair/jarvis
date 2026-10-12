"""Speech qualification reaches the selected local-capable transport."""
from unittest.mock import MagicMock
import pytest

from evals.test_speech_constraint_resolution import (
    test_step_resolution_uses_the_current_request_count as resolve_case,
)
from evals.test_speech_context_routing import (
    test_router_resolves_speech_reference_before_selecting_capabilities as routing_case,
)

pytestmark = pytest.mark.unit

@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('operation', ['routing', 'resolution'])
def test_speech_evals_reach_selected_endpoint_and_model(monkeypatch, provider, operation):
    import requests
    base, model = 'http://127.0.0.1:1', 'selected-speech-model'
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_JUDGE_BASE_URL', base)
    monkeypatch.setenv('EVAL_JUDGE_MODEL', model)
    observed = []
    def post(url, **kwargs):
        observed.append((url, kwargs))
        response = MagicMock()
        response.__enter__.return_value = response
        content = ('{"name":"selectItems","arguments":{"count":3}}' if operation == 'resolution'
                   else '{"requested_operation":"search the item price","tools":["webSearch"]}')
        message = {'content': content}
        response.json.return_value = ({'message': message} if provider == 'ollama'
                                     else {'choices': [{'message': message}]})
        return response
    monkeypatch.setattr(requests, 'post', post)
    if operation == 'resolution':
        resolve_case('Pick three items', 3)
    else:
        routing_case('Copper lantern', 'Jarvis price?', 'webSearch')
    endpoint = base+('/api/chat' if provider == 'ollama' else '/v1/chat/completions')
    assert {(url, kwargs['json']['model']) for url, kwargs in observed} == {(endpoint, model)}
