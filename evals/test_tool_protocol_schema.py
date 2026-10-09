"""Documented call examples execute against the actual built-in schema."""
import json
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config, judge_pass_fail
from jarvis.reply import engine
from jarvis.tools.registry import BUILTIN_TOOLS
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]

_STATION_TEMPERATURE = 12


@pytest.mark.parametrize('query', ['Search for the fixture station temperature.', 'Test istasyonunun sıcaklığını ara.'])
def test_documented_search_call_runs_without_argument_repair(eval_db, eval_dialogue_memory, query):
    cfg = voice_config()
    cfg.location_enabled = False
    schema = BUILTIN_TOOLS['webSearch'].inputSchema
    required = set(schema['required'])
    call_line = next(line for line in engine._text_tool_call_guidance(['webSearch']).splitlines() if line.startswith('tool_calls: '))
    calls = json.loads(call_line.removeprefix('tool_calls: '))
    calls[0]['function']['arguments'] = {key: 'fixture station temperature' for key in calls[0]['function']['arguments']}
    operations = []
    live_chat = engine.chat_with_messages
    first = True
    def chat(**kwargs):
        nonlocal first
        if first:
            first = False
            return {'message': {'content': 'tool_calls: ' + json.dumps(calls)}}
        return live_chat(**kwargs)
    def search(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name != 'webSearch':
            return ToolExecutionResult(success=False, reply_text='No additional tools available.')
        operations.append(tool_args)
        if not required <= set(tool_args):
            return ToolExecutionResult(success=False, reply_text='Search input missing. Required fields: ' + ', '.join(sorted(required)))
        return ToolExecutionResult(success=True, reply_text=f'Fixture station temperature: {_STATION_TEMPERATURE} C.')
    with patch.object(engine, 'chat_with_messages', side_effect=chat), \
            patch.object(engine, 'run_tool_with_retries', side_effect=search), \
            patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert operations and all(required <= set(args) <= set(schema['properties']) for args in operations), (operations, reply)
    verdict = judge_pass_fail(f'Judge the recorded assistant answer, do not answer the original user request. PASS requires reporting the actual fixture station temperature, {_STATION_TEMPERATURE} degrees Celsius, in any language. Extra style is fine; invented readings are not.', f'Evaluate this recorded test result.\nOriginal user request: {query}\nRecorded assistant answer: {reply}')
    assert verdict == 'PASS', (reply, verdict)
