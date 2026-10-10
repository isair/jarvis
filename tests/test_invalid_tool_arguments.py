"""Malformed arguments never become an empty call with different semantics."""
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('content', [
    'tool_calls: [{"function": {"name": "getWeather", "arguments": "{\\"location\\": \\"Ankara\\"}}}]',
    '```tool_call\n{"name": "getWeather", "arguments": ["Ankara"]}\n```',
    'getWeather({"location": "Ankara)',
    'tool_calls: [{"function": {"name": "getWeather", "arguments": "null"}}]',
])
def test_malformed_argument_objects_are_not_replaced_with_defaults(mock_config, db, dialogue_memory, content):
    operations = []
    responses = iter([{'message': {'content': content}},
                      {'message': {'content': 'tool_calls: [{"function": {"name": "getWeather", "arguments": {"location": "Ankara"}}}]'}},
                      {'message': {'content': 'Ankara weather checked.'}}])
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args)
        return ToolExecutionResult(success=True, reply_text='Ankara: 24 C, clear.')
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(responses)):
        reply = engine.run_reply_engine(db, mock_config, None, 'Weather in Ankara?', dialogue_memory)
    assert operations == [{'location': 'Ankara'}]
    assert 'checked' in reply


@pytest.mark.parametrize('args', ['{"location": "Ankara"}', '["Ankara"]', None, 42])
def test_native_argument_shape_is_decoded_or_rejected(mock_config, db, dialogue_memory, args):
    mock_config.llm_chat_model = 'gpt-oss:20b'
    operations = []
    responses = iter([{'message': {'content': '', 'tool_calls': [{'id': 'c1', 'function': {'name': 'getWeather', 'arguments': args}}]}},
                      {'message': {'content': 'tool_calls: [{"function": {"name": "getWeather", "arguments": {"location": "Ankara"}}}]'}},
                      {'message': {'content': 'Ankara weather checked.'}}])
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args)
        return ToolExecutionResult(success=True, reply_text='Ankara: 24 C, clear.')
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(responses)):
        engine.run_reply_engine(db, mock_config, None, 'Weather in Ankara?', dialogue_memory)
    assert operations == [{'location': 'Ankara'}]


def test_text_argument_arrays_and_literal_brackets_remain_complete():
    import json
    arguments = {'items': ['first', 'second]value'], 'options': {'enabled': True}}
    content = 'tool_calls: ' + json.dumps([{'function': {'name': 'fixtureTool', 'arguments': arguments}}])
    name, decoded, _ = engine._extract_text_tool_call(content, {'fixtureTool'})
    assert name == 'fixtureTool'
    assert decoded == arguments


def test_incomplete_or_ambiguous_argument_strings_cannot_run_defaults():
    contents = [
        'tool_calls: [{"function": {"name": "getWeather", "arguments": "{\\"location\\": \\"Ankara}}}]',
        'tool_calls: [{"function": {"name": "getWeather", "arguments": "{\\"location\\": \\"Ankara\\"} extra-data}}}]',
    ]
    for content in contents:
        name, arguments, _ = engine._extract_text_tool_call(content, {'getWeather'})
        assert name == 'getWeather'
        assert arguments is None


def test_invalid_model_call_cannot_advance_to_dependent_plan_action(mock_config, db, dialogue_memory):
    operations = []
    responses = iter([
        {'message': {'content': '```tool_call\n{"name": "localFiles", "arguments": ["read", "report.txt"]}\n```'}},
        {'message': {'content': '{"question":"Please clarify the read arguments. The file was not deleted."}'}},
    ])
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args)
        return ToolExecutionResult(success=True, reply_text='Operation completed.')
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=["localFiles operation='read' path='report.txt'", "localFiles operation='delete' path='report.txt'"]), \
            patch.object(engine, '_resolve_plan_step', side_effect=[None, ('localFiles', {'operation': 'delete', 'path': 'report.txt'})]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(responses)):
        reply = engine.run_reply_engine(db, mock_config, None, 'Read report.txt, then delete only if reading succeeded.', dialogue_memory)
    assert not operations
    assert 'not deleted' in reply


@pytest.mark.parametrize('arguments', [None, ['Ankara'], 42])
def test_invalid_native_arguments_retain_valid_provider_wire_shape(arguments):
    import json
    from unittest.mock import MagicMock
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend
    received = []
    def post(url, **kwargs):
        received.append(kwargs['json'])
        response = MagicMock()
        response.__enter__.return_value = response
        response.raise_for_status.return_value = None
        response.json.return_value = {'choices': [{'message': {'role': 'assistant', 'content': 'Please correct the arguments.'}}]}
        return response
    messages = [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
            'function': {'name': 'getWeather', 'arguments': arguments}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Invalid arguments, tool not executed.'},
    ]
    with patch('jarvis.llm.openai_compatible.requests.post', side_effect=post):
        result = OpenAICompatibleBackend('http://127.0.0.1:1234/v1').chat('fixture-model', messages)
    encoded = received[0]['messages'][0]['tool_calls'][0]['function']['arguments']
    assert isinstance(encoded, str), 'Provider history requires JSON strings, even for failed calls'
    assert json.loads(encoded) == arguments
    assert result['message']['content'] == 'Please correct the arguments.'


def test_ambiguous_multi_call_envelope_cannot_mix_names_and_arguments():
    content = 'tool_calls: [{"function": {"name": "getWeather"}}, {"function": {"name": "localFiles", "arguments": {"path": "report.txt"}}}'
    name, arguments, _ = engine._extract_text_tool_call(content, {'getWeather', 'localFiles'})
    assert name == 'getWeather'
    assert arguments is None, 'Later function arguments must never be attached to an earlier tool'


def test_text_protocol_examples_use_consistent_argument_objects():
    import json
    from jarvis.tools.registry import generate_tools_description
    prompts = [generate_tools_description(['webSearch']), engine._text_tool_call_guidance(['webSearch'])]
    for prompt in prompts:
        call_line = next(line for line in prompt.splitlines() if line.startswith('tool_calls: '))
        call = json.loads(call_line.removeprefix('tool_calls: '))[0]
        assert isinstance(call['function']['arguments'], dict)


def test_text_search_example_matches_the_actual_tool_schema():
    import json
    from jarvis.tools.registry import BUILTIN_TOOLS
    guidance = engine._text_tool_call_guidance(['webSearch'])
    call = json.loads(next(line for line in guidance.splitlines() if line.startswith('tool_calls: ')).removeprefix('tool_calls: '))[0]['function']
    schema = BUILTIN_TOOLS[call['name']].inputSchema
    arguments = call['arguments']
    assert set(schema.get('required', [])) <= set(arguments)
    assert set(arguments) <= set(schema['properties'])
