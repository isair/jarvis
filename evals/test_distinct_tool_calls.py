"""Live replies can finish distinct operations after two prior tool results."""
import json
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import call_judge_llm, voice_config
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query', [
    'Search separately for the weather in London, Paris and Ankara, then compare all three.',
    'Londra, Paris ve Ankara için hava durumunu ayrı ayrı ara, sonra üçünü karşılaştır.',
])
def test_live_model_finishes_third_distinct_search(eval_db, eval_dialogue_memory, query):
    cfg = voice_config()
    cfg.location_enabled = False
    checked = set()
    temperatures = {'London': 12, 'Paris': 18, 'Ankara': 24}
    def search(db, cfg, tool_name, tool_args, **kwargs):
        term = str((tool_args or {}).get('query', '')).casefold()
        for city, temperature in temperatures.items():
            if city.casefold() in term:
                checked.add(city)
                return ToolExecutionResult(success=True, reply_text=f'{city}: {temperature} C, clear.')
        return ToolExecutionResult(success=False, reply_text='The search requires a supported city.')
    plan = [f"webSearch query='{city} weather'" for city in temperatures] + ['Compare all three recorded results.']
    live_chat = engine.chat_with_messages
    first_requests = iter(('London', 'Paris'))
    def chat(**kwargs):
        city = next(first_requests, None)
        if city:
            return {'message': {'content': 'tool_calls: ' + json.dumps([{'function': {
                'name': 'webSearch', 'arguments': {'query': f'{city} weather'}}}])}}
        return live_chat(**kwargs)
    with patch.object(engine, 'run_tool_with_retries', side_effect=search), \
            patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
            patch.object(engine, 'plan_query', return_value=plan), \
            patch.object(engine, '_resolve_plan_step', return_value=None), \
            patch.object(engine, 'chat_with_messages', side_effect=chat), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert checked == set(temperatures), f'Missing actual searches: {checked}; reply={reply}'
    verdict = call_judge_llm(
        'Output only PASS or FAIL. PASS requires the reply to compare all three '
        'cities using their recorded temperatures correctly. Every named city '
        'must have its actual tool temperature, with no invented weather '
        'result. A list plus a warmer/cooler comparison is sufficient; ignore '
        'stylistic flourishes that make no new weather-data claim. '
        'Judge meaning in any language.',
        f'Query: {query}\nRecorded temperatures: {temperatures}\nReply: {reply}',
    )
    assert verdict and verdict.strip().upper() == 'PASS', f'Comparison not grounded: {reply}; judge={verdict}'
