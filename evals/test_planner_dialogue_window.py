"""Actual planning retains dialogue entities across native tool traffic."""
from unittest.mock import patch

import pytest
from evals.helpers import ToolCallCapture, voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply import engine
from jarvis.reply.prompts.model_variants import ModelSize
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('entity, aliases, query', [
    ('Ursula Le Guin', ('Ursula Le Guin', 'Ursula K. Le Guin'), 'Which books did she write?'),
    ('夏目漱石', ('夏目漱石', 'Natsume Sōseki', 'Natsume Soseki'), '彼の本を検索して'),
    ('Orhan Pamuk', ('Orhan Pamuk',), 'Onun kitaplarını ara'),
])
def test_real_planner_dispatches_the_dialogue_entity(eval_db, eval_dialogue_memory, entity, aliases, query):
    cfg = voice_config()
    cfg.location_enabled = False
    cfg.planner_timeout_sec = 60.0
    eval_dialogue_memory.add_message('user', f'We are discussing {entity}.')
    traffic = []
    for index in range(5):
        traffic.extend([
            {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': str(index), 'type': 'function',
                'function': {'name': 'webSearch', 'arguments': {'search_query': 'reference material'}},
            }]},
            {'role': 'tool', 'tool_name': 'webSearch', 'tool_call_id': str(index),
             'content': 'Reference material.', 'tool_failed': False},
        ])
    eval_dialogue_memory.record_tool_turn(traffic)
    eval_dialogue_memory.add_message('assistant', 'What would you like to find about them?')
    capture = ToolCallCapture()

    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args)
        return ToolExecutionResult(success=True, reply_text='Local evaluation result.')

    # Exercise the selected backend's real planner/resolver and engine dispatch.
    # Force direct execution for every backend; synthesis and external search
    # are isolated so this case measures the planning context boundary.
    with patch.object(engine, 'select_tools', return_value=['webSearch', 'stop']), \
         patch.object(engine, 'detect_model_size', return_value=ModelSize.SMALL), \
         patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
         patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'Evaluation complete.'}}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert reply == 'Evaluation complete.'
    searches = [call['args'].get('search_query', '') for call in capture.calls if call['name'] == 'webSearch']
    assert any(alias.casefold() in search.casefold() for search in searches for alias in aliases), f'📚 Named dialogue entity was lost: {searches}'
