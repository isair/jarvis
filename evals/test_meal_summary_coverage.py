"""Replies distinguish partial recorded estimates from total consumption."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import judge_pass_fail, voice_config
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query', [
    "How many calories are recorded today, and does that total cover every meal?",
    'Bugün kaç kalori kaydedilmiş, bu toplam her öğünü kapsıyor mu?',
])
def test_partial_recorded_calories_are_not_complete_intake(eval_db, eval_dialogue_memory, query):
    now = datetime.now(timezone.utc).isoformat()
    eval_db.insert_meal(now, 'eval', 'Known meal', calories_kcal=150)
    eval_db.insert_meal(now, 'eval', 'Unknown meal')
    cfg = voice_config()
    cfg.planner_timeout_sec = cfg.llm_chat_timeout_sec = 60
    # Isolate result interpretation from the separately exercised date resolver.
    with patch.object(engine, 'select_tools', return_value=['fetchMeals', 'stop']), \
            patch.object(engine, 'plan_query', return_value=['fetchMeals']), \
            patch.object(engine, '_resolve_plan_step', return_value=('fetchMeals', {})), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    verdict = judge_pass_fail(
        'PASS requires reporting 150 recorded kcal and explaining that this is partial: '
        'one of the two meals has no calorie estimate, so actual total consumption is unknown. '
        'FAIL if 150 is presented as a complete total or the unknown meal is assigned zero. '
        'Judge meaning in any language.',
        f'Question: {query}\nReply: {reply}',
    )
    assert verdict == 'PASS', (reply, verdict)
