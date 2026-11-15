"""Structured reply schemas stay provider-neutral and preserve complete tool batches."""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import json
import requests

import pytest

from jarvis.llm.ollama import OllamaBackend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('provider', ['ollama', 'compatible'])
def test_reply_schema_reaches_the_local_provider_without_becoming_a_sampling_option(provider):
    schema = {'type': 'object', 'properties': {'reply': {'type': 'string'}}, 'required': ['reply']}
    backend = OllamaBackend('http://owned.invalid') if provider == 'ollama' else OpenAICompatibleBackend('http://owned.invalid')
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {'message': {'content': '{"reply":"Done"}'}, 'choices': [{'message': {'content': '{"reply":"Done"}'}}]}
    with patch('jarvis.llm.http.post', return_value=response) as post:
        assert backend.chat('fixture', [{'role': 'user', 'content': 'Owned request'}], extra_options={'response_schema': schema})
    payload = post.call_args.kwargs['json']
    assert 'response_schema' not in payload and 'response_schema' not in payload.get('options', {})
    actual = payload['format'] if provider == 'ollama' else payload['response_format']['json_schema']['schema']
    assert actual == schema


@pytest.mark.parametrize('expired', [False, True])
def test_unsupported_schema_falls_back_within_the_original_deadline(monkeypatch, expired):
    from jarvis.llm import openai_compatible as module
    clock = [0.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    observed = []
    def post(url, **kwargs):
        observed.append((json.loads(json.dumps(kwargs['json'])), kwargs['timeout']))
        response = MagicMock()
        response.__enter__.return_value = response
        if len(observed) == 1:
            clock[0] = 2.0 if expired else .25
            rejected = requests.Response()
            rejected.status_code = 400
            response.raise_for_status.side_effect = requests.HTTPError(response=rejected)
        else:
            response.json.return_value = {'choices': [{'message': {'content': '{"reply":"Done"}'}}]}
        return response
    with patch('jarvis.llm.http.post', side_effect=post):
        result = OpenAICompatibleBackend('http://owned.invalid').chat('fixture', [{'role': 'user', 'content': 'Owned'}],
            timeout_sec=1.0, extra_options={'response_schema': {'type': 'object'}})
    assert 'response_format' in observed[0][0]
    if expired:
        assert result is None and len(observed) == 1
    else:
        assert result['message']['content'] == '{"reply":"Done"}'
        assert len(observed) == 2 and 'response_format' not in observed[1][0]
        assert 0 < observed[1][1] < observed[0][1]


@pytest.mark.parametrize('reply', [False, True])
def test_structured_arguments_preserve_actual_timer_schema_constraints(reply):
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema, reply_output_schema
    from jarvis.tools.builtin.timer import TimerTool
    tools = {'timer': TimerTool().inputSchema}
    schema = reply_output_schema(tools, 2) if reply else tool_call_output_schema(tools)
    def envelope(arguments):
        call = {'name': 'timer', 'arguments': arguments}
        return {'tool_calls': [{'function': call}]} if reply else call
    jsonschema.validate(envelope({'action': 'list'}), schema)
    jsonschema.validate(envelope({'action': 'set', 'minutes': 5, 'expiry_announcement': 'Tea is ready'}), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(envelope({'action': 'set', 'minutes': 5}), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(envelope({'action': 'set', 'minutes': 'five', 'expiry_announcement': 'Tea is ready'}), schema)
