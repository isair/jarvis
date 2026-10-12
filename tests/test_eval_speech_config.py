"""Speech qualification reaches the selected local-capable transport."""
from unittest.mock import MagicMock
import pytest


pytestmark = pytest.mark.unit

@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('operation', ['routing', 'resolution'])
def test_speech_evals_reach_selected_endpoint_and_model(monkeypatch, provider, operation):
    import requests
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError('synthetic unavailable endpoint')
    with monkeypatch.context() as availability:
        availability.setattr(requests, 'get', unavailable)
        from evals.test_speech_constraint_resolution import (
            test_step_resolution_uses_the_current_request_count as resolve_case,
        )
        from evals.test_speech_context_routing import (
            test_router_resolves_speech_reference_before_selecting_capabilities as routing_case,
        )
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


def test_collecting_speech_eval_contracts_does_not_probe_inference_endpoints():
    import json
    import subprocess
    import sys
    from pathlib import Path
    program = '''
import json, requests, runpy, sys
probes = []
def get(*args, **kwargs):
    probes.append(args[0])
    raise requests.ConnectionError('synthetic unavailable endpoint')
requests.get = get
runpy.run_path(sys.argv[1])
print(json.dumps(probes))
'''
    completed = subprocess.run([sys.executable, '-B', '-c', program, str(Path(__file__).resolve())],
                               check=True, capture_output=True, text=True, timeout=15)
    assert json.loads(completed.stdout.splitlines()[-1]) == []
