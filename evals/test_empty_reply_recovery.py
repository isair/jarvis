"""A captured empty assistant envelope can recover a grounded local-model reply."""
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config, judge_pass_fail
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]
_TEMPERATURE = 15


@pytest.mark.parametrize('query', ['How is the weather here?', 'Burada hava nasıl?'])
def test_empty_assistant_response_recovers_an_actual_lookup(eval_db, eval_dialogue_memory, query):
    cfg = voice_config()
    cfg.location_enabled = True
    operations = []
    live_chat = engine.chat_with_messages
    first = True
    def chat(**kwargs):
        nonlocal first
        if first:
            first = False
            return {'message': {'role': 'assistant', 'content': ''}, 'done': True,
                    'done_reason': 'stop', 'eval_count': 7}
        return live_chat(**kwargs)
    def weather(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name != 'getWeather':
            return ToolExecutionResult(success=False, reply_text='No additional tools available.')
        operations.append(tool_args)
        return ToolExecutionResult(success=True, reply_text=f'London weather: {_TEMPERATURE} C, clear.')
    with patch.object(engine, 'chat_with_messages', side_effect=chat), \
            patch.object(engine, 'run_tool_with_retries', side_effect=weather), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'get_location_context_with_timezone', return_value=('Location: London, UK', None)), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert operations, f'No actual weather operation: {reply}'
    verdict = judge_pass_fail(
        f'PASS requires reporting the actual London reading '
        f'of {_TEMPERATURE} degrees Celsius and clear conditions in any language. '
        'Missing or invented readings require FAIL.',
        f'Original request: {query}\nRecorded answer: {reply}',
    )
    assert verdict == 'PASS', (reply, verdict)
