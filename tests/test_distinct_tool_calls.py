"""Distinct operations can share a tool while duplicate loops stay bounded."""
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


def _call(args):
    return {'message': {'content': '', 'tool_calls': [{'function': {
        'name': 'getWeather', 'arguments': args}}]}}


def _run(cfg, db, memory, responses, outcome):
    with patch.object(engine, 'run_tool_with_retries', side_effect=outcome), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=responses), \
            patch.object(engine, 'digest_loop_for_max_turns', return_value='The request remains incomplete.'):
        return engine.run_reply_engine(db, cfg, None,
                                      'Compare weather in London, Paris and Ankara.', memory)


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
@pytest.mark.parametrize('carryover', [False, True])
def test_three_distinct_locations_are_all_queried(mock_config, db, dialogue_memory, model, carryover):
    mock_config.llm_chat_model = model
    if carryover:
        dialogue_memory.add_message('user', 'Earlier weather request')
        dialogue_memory.record_tool_turn([
            {'role': 'user', 'tool_name': 'getWeather', 'content': 'Earlier weather result'},
            {'role': 'user', 'tool_name': 'getWeather', 'content': 'Another earlier result'},
        ])
        dialogue_memory.add_message('assistant', 'Earlier weather summary')
    locations = []
    def outcome(db, cfg, tool_name, tool_args, **kwargs):
        locations.append(tool_args['location'])
        return ToolExecutionResult(success=True, reply_text=f"{tool_args['location']}: 15 C")
    responses = [_call({'location': place}) for place in ('London', 'Paris', 'Ankara')]
    responses.append({'message': {'content': 'All three cities are 15 C.'}})
    _run(mock_config, db, dialogue_memory, responses, outcome)
    assert locations == ['London', 'Paris', 'Ankara']


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
@pytest.mark.parametrize('success', [False, True])
def test_identical_operations_reuse_success_or_failure(mock_config, db, dialogue_memory, model, success):
    mock_config.llm_chat_model = model
    locations = []
    def outcome(db, cfg, tool_name, tool_args, **kwargs):
        locations.append(tool_args['location'])
        return ToolExecutionResult(success=success, reply_text='London: 15 C' if success else None,
                                   error_message=None if success else 'Weather unavailable')
    responses = [_call({'location': 'London'}), _call({'location': 'London'}),
                 {'message': {'content': 'Weather checked.' if success else 'Weather unavailable.'}}]
    _run(mock_config, db, dialogue_memory, responses, outcome)
    assert locations == ['London']


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
def test_distinct_operations_still_stop_at_configured_turn_budget(mock_config, db, dialogue_memory, model):
    mock_config.llm_chat_model = model
    mock_config.agentic_max_turns = 4
    locations = []
    def outcome(db, cfg, tool_name, tool_args, **kwargs):
        locations.append(tool_args['location'])
        return ToolExecutionResult(success=True, reply_text='Weather checked.')
    responses = [_call({'location': f'City {n}'}) for n in range(mock_config.agentic_max_turns + 1)]
    reply = _run(mock_config, db, dialogue_memory, responses, outcome)
    assert locations == [f'City {n}' for n in range(mock_config.agentic_max_turns)]
    assert 'incomplete' in reply


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
def test_corrected_prerequisite_allows_later_distinct_operation(mock_config, db, dialogue_memory, model):
    mock_config.llm_chat_model = model
    operations = []
    def outcome(db, cfg, tool_name, tool_args, **kwargs):
        operations.append((tool_args['operation'], tool_args['path']))
        if tool_args['path'] == 'report.txt':
            return ToolExecutionResult(success=False, reply_text='Use ./report.txt to read the report.')
        return ToolExecutionResult(success=True, reply_text='Read report contents.' if tool_args['operation'] == 'read' else 'Deleted report.')
    def file_call(operation):
        return {'message': {'content': '', 'tool_calls': [{'function': {'name': 'localFiles',
                'arguments': {'operation': operation, 'path': './report.txt'}}}]}}
    responses = iter([file_call('read'), file_call('delete'), {'message': {'content': 'Report read and deleted.'}}])
    plan = ["localFiles operation='read' path='report.txt'", "localFiles operation='delete' path='report.txt'", 'Reply to user.']
    # Native mode starts with a model call; text mode uses the concrete plan.
    if model == 'gpt-oss:20b':
        initial = {'message': {'content': '', 'tool_calls': [{'function': {'name': 'localFiles',
                   'arguments': {'operation': 'read', 'path': 'report.txt'}}}]}}
        responses = iter([initial, file_call('read'), file_call('delete'), {'message': {'content': 'Report read and deleted.'}}])
    with patch.object(engine, 'run_tool_with_retries', side_effect=outcome), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=plan), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(responses)):
        reply = engine.run_reply_engine(db, mock_config, None,
                'Read report.txt, only delete it if reading succeeds.', dialogue_memory)
    assert operations == [('read', 'report.txt'), ('read', './report.txt'), ('delete', './report.txt')]
    assert 'read and deleted' in reply
