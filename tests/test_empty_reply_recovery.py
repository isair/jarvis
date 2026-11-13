"""A valid empty model turn has one bounded opportunity to recover."""
import json
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.fixture(params=['gemma4:e2b', 'gpt-oss:20b'])
def reply_config(mock_config, request):
    mock_config.llm_chat_model = request.param
    return mock_config


def _reply(cfg, db, memory, responses):
    answers = iter(responses)
    operations = []
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append((tool_name, tool_args))
        return ToolExecutionResult(success=True, reply_text='Ankara: 24 C, clear.')
    with patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(answers)), \
            patch.object(engine, 'digest_loop_for_max_turns', return_value=None), \
            patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(db, cfg, None, 'Weather in Ankara?', memory)
    return reply, operations


@pytest.mark.parametrize('blank', ['', ' \n\t'])
def test_valid_blank_turn_can_recover_an_answer(reply_config, db, dialogue_memory, blank):
    reply, operations = _reply(reply_config, db, dialogue_memory, [
        {'message': {'role': 'assistant', 'content': blank}},
        {'message': {'content': 'Recovered answer.'}},
    ])
    assert reply == 'Recovered answer.'
    assert not operations


def test_valid_blank_turn_can_recover_a_tool_operation(reply_config, db, dialogue_memory):
    call = {'id': 'weather', 'function': {'name': 'getWeather', 'arguments': {'location': 'Ankara'}}}
    request = ({'message': {'content': 'tool_calls: ' + json.dumps([call])}}
               if reply_config.llm_chat_model == 'gemma4:e2b'
               else {'message': {'content': '', 'tool_calls': [call]}})
    reply, operations = _reply(reply_config, db, dialogue_memory, [
        {'message': {'role': 'assistant', 'content': ''}}, request,
        {'message': {'content': 'Ankara is 24 C.'}},
    ])
    assert operations == [('getWeather', {'location': 'Ankara'})]
    assert reply == 'Ankara is 24 C.'


def test_a_second_empty_turn_cannot_reach_a_third_answer(reply_config, db, dialogue_memory):
    third = 'Third answer must not be delivered.'
    empty = {'message': {'role': 'assistant', 'content': ''}}
    reply, operations = _reply(reply_config, db, dialogue_memory, [empty, empty, {'message': {'content': third}}])
    assert reply and reply != third
    assert not operations


def test_recovery_cannot_exceed_the_configured_turn_budget(reply_config, db, dialogue_memory):
    reply_config.agentic_max_turns = 1
    recovered = 'Answer beyond the configured budget.'
    reply, operations = _reply(reply_config, db, dialogue_memory, [
        {'message': {'role': 'assistant', 'content': ''}},
        {'message': {'content': recovered}},
    ])
    assert reply and reply != recovered
    assert not operations


@pytest.mark.parametrize('unavailable', [None, {'error': 'unavailable'},
                                       {'message': {'role': 'tool', 'content': ''}},
                                       {'message': {'content': None}},
                                       {'message': {'content': '', 'tool_calls': [{'function': {'arguments': {}}}]}},
                                       {'message': {'content': '', 'tool_calls': 'invalid'}}])
def test_unavailable_or_invalid_responses_are_not_empty_successful_turns(reply_config, db, dialogue_memory, unavailable):
    recovered = 'Unavailable response must not lead to this answer.'
    reply, operations = _reply(reply_config, db, dialogue_memory, [unavailable, {'message': {'content': recovered}}])
    assert reply and reply != recovered
    assert not operations


def test_thinking_only_continuation_remains_available(reply_config, db, dialogue_memory):
    reply, operations = _reply(reply_config, db, dialogue_memory, [
        {'message': {'content': '', 'thinking': 'Considering the answer.'}},
        {'message': {'content': 'Considered answer.'}},
    ])
    assert reply == 'Considered answer.'
    assert not operations
