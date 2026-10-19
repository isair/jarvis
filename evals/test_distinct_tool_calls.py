"""Live replies can finish distinct operations after two prior tool results."""
import json
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import judge_pass_fail, voice_config
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult
from jarvis.tools.registry import BUILTIN_TOOLS

_SEARCH_ARGUMENT = BUILTIN_TOOLS["webSearch"].inputSchema["required"][0]

pytestmark = [pytest.mark.eval, requires_judge_llm]


_COMPARISON_VERDICT_RULES = (
    'Judge the recorded assistant answer, do not answer the original request. '
    'PASS requires explicitly reporting '
    'every city with its actual recorded temperature. Side-by-side readings '
    'count as a comparison; ranking or warmer/cooler wording is not required. '
    'Judge the grounding of these measured readings. Opinions, humour and '
    'qualitative weather comments are outside this check unless they contradict '
    'the recorded numeric readings. Never guess missing readings or accept '
    'swapped values. Judge meaning in any language.'
)

def _record_fixture_search(temperatures, checked, tool_name, tool_args):
    if tool_name != 'webSearch':
        return ToolExecutionResult(success=False, reply_text='No additional tools available.')
    term = str((tool_args or {}).get(_SEARCH_ARGUMENT, '')).casefold()
    for city, temperature in temperatures.items():
        if city.casefold() in term:
            checked.add(city)
            return ToolExecutionResult(success=True, reply_text=f'{city}: {temperature} C, clear.')
    return ToolExecutionResult(success=False, reply_text='The search requires a supported city.')


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
        return _record_fixture_search(temperatures, checked, tool_name, tool_args)
    plan = [f"webSearch {_SEARCH_ARGUMENT}='{city} weather'" for city in temperatures] + ['Compare all three recorded results.']
    live_chat = engine.chat_with_messages
    first_requests = iter(('London', 'Paris'))
    def chat(**kwargs):
        city = next(first_requests, None)
        if city:
            return {'message': {'content': 'tool_calls: ' + json.dumps([{'function': {
                'name': 'webSearch', 'arguments': {_SEARCH_ARGUMENT: f'{city} weather'}}}])}}
        return live_chat(**kwargs)
    with patch.object(engine, 'run_tool_with_retries', side_effect=search), \
            patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
            patch.object(engine, 'plan_query', return_value=plan), \
            patch.object(engine, '_resolve_plan_step', return_value=None), \
            patch.object(engine, 'chat_with_messages', side_effect=chat), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert checked == set(temperatures), f'Missing actual searches: {checked}; reply={reply}'
    verdict = judge_pass_fail(
        _COMPARISON_VERDICT_RULES,
        f'Query: {query}\nRecorded temperatures: {temperatures}\nReply: {reply}',
    )
    assert verdict == 'PASS', f'Comparison not grounded: {reply}; judge={verdict}'


@pytest.mark.parametrize('variant,expected', [('complete', 'PASS'), ('with-comment', 'PASS'), ('missing', 'FAIL'), ('swapped', 'FAIL')])
def test_comparison_judge_requires_complete_correct_readings(variant, expected):
    temperatures = {'London': 12, 'Paris': 18, 'Ankara': 24}
    rows = list(temperatures.items())
    if variant == 'missing':
        rows = rows[:-1]
    elif variant == 'swapped':
        rows = [(city, rows[(i + 1) % len(rows)][1]) for i, (city, _) in enumerate(rows)]
    reply = ' | '.join(f'{city}: {temperature} C, clear' for city, temperature in rows)
    if variant == 'with-comment':
        reply += '. It seems quite consistent across the three locations today.'
    verdict = judge_pass_fail(_COMPARISON_VERDICT_RULES, f'Query: Compare all three cities.\nRecorded temperatures: {temperatures}\nReply: {reply}')
    assert verdict == expected, (reply, verdict)
