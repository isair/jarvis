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


@pytest.mark.parametrize('reply', [False, True])
def test_tool_schema_local_references_keep_separate_definition_scopes(reply):
    from copy import deepcopy
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema, reply_output_schema
    def arguments_schema(kind):
        return {'type': 'object', '$defs': {'Item': {'type': kind}},
                'properties': {'value': {'$ref': '#/$defs/Item'}}, 'required': ['value']}
    tools = {'textTool': arguments_schema('string'), 'numberTool': arguments_schema('integer')}
    before = deepcopy(tools)
    schema = reply_output_schema(tools, 2) if reply else tool_call_output_schema(tools)
    def envelope(name, value):
        call = {'name': name, 'arguments': {'value': value}}
        return {'tool_calls': [{'function': call}]} if reply else call
    jsonschema.validate(envelope('textTool', 'one'), schema)
    jsonschema.validate(envelope('numberTool', 1), schema)
    for name, value in [('textTool', 1), ('numberTool', 'one')]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(envelope(name, value), schema)
    assert tools == before


def test_recursive_tool_schema_keeps_its_own_root():
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema
    schema = tool_call_output_schema({'tree': {'type': 'object', 'properties': {
        'value': {'type': 'integer'}, 'child': {'anyOf': [{'type': 'null'}, {'$ref': '#'}]}},
        'required': ['value']}})
    jsonschema.validate({'name': 'tree', 'arguments': {'value': 1, 'child': {'value': 2}}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'name': 'tree', 'arguments': {'value': 1, 'child': {'value': 'bad'}}}, schema)


@pytest.mark.parametrize('scope', [{'$id': 'urn:example:tool'}, {'$anchor': 'item'},
                                  {'$dynamicRef': '#item'}, {'$ref': 'https://example.invalid/schema'}])
def test_unsupported_reference_scopes_omit_the_whole_format(scope):
    from jarvis.tools.schema import tool_call_output_schema, reply_output_schema
    tools = {'external': {'type': 'object', **scope}, 'simple': {'type': 'object'}}
    assert tool_call_output_schema(tools) is None
    assert reply_output_schema(tools, 2) is None


def test_reference_like_literal_data_is_not_rewritten_or_rejected():
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema
    literal = {'$ref': 'literal data, not a schema reference'}
    schema = tool_call_output_schema({'literal': {'type': 'object', 'properties': {
        'value': {'const': literal}}, 'required': ['value']}})
    assert schema is not None
    jsonschema.validate({'name': 'literal', 'arguments': {'value': literal}}, schema)


def test_nullable_call_keeps_definitions_at_the_actual_root():
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema
    schema = tool_call_output_schema({'name': {'type': 'object', '$defs': {'Value': {'type': 'integer'}},
        'properties': {'value': {'$ref': '#/$defs/Value'}}, 'required': ['value']}}, nullable=True)
    jsonschema.validate(None, schema)
    jsonschema.validate({'name': 'name', 'arguments': {'value': 1}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'name': 'name', 'arguments': {'value': 'bad'}}, schema)


def test_shared_reference_nodes_keep_their_single_definition_scope():
    import jsonschema
    from jarvis.tools.schema import tool_call_output_schema
    reference = {'$ref': '#/$defs/Value'}
    schema = tool_call_output_schema({'pair': {'type': 'object', '$defs': {'Value': {'type': 'integer'}},
        'properties': {'left': reference, 'right': reference}, 'required': ['left', 'right']}})
    jsonschema.validate({'name': 'pair', 'arguments': {'left': 1, 'right': 2}}, schema)


@pytest.mark.parametrize('provider', ['ollama', 'compatible'])
def test_unsupported_reference_scope_uses_the_unconstrained_transport(provider):
    from jarvis.tools.schema import tool_call_output_schema
    schema = tool_call_output_schema({'external': {'$id': 'urn:external:tool', 'type': 'object'}})
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {'message': {'content': '{"reply":"Done"}'}, 'choices': [{'message': {'content': '{"reply":"Done"}'}}]}
    backend = OllamaBackend('http://owned.invalid') if provider == 'ollama' else OpenAICompatibleBackend('http://owned.invalid')
    with patch('jarvis.llm.http.post', return_value=response) as post:
        assert backend.chat('fixture', [{'role': 'user', 'content': 'Owned request'}], extra_options={'response_schema': schema})
    payload = post.call_args.kwargs['json']
    assert 'format' not in payload and 'response_format' not in payload
    assert 'response_schema' not in payload and 'response_schema' not in payload.get('options', {})
