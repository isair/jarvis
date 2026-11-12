"""Replies distinguish partial recorded estimates from total consumption."""
from datetime import datetime, timezone
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import judge_pass_fail, voice_config
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


def coverage_verdict(query, reply):
    return judge_pass_fail(
        'PASS requires reporting 150 recorded kcal and explaining that this is partial: '
        'one of the two meals has no calorie estimate, so actual total consumption is unknown. '
        'Equivalent descriptions such as half the meals or one of two meals are valid; '
        'exact wording is not required. All three claims must be present. '
        'FAIL if 150 is presented as a complete total or the unknown meal is assigned zero. '
        'Judge meaning in any language.',
        f'Question: {query}\nReply: {reply}',
    )


@pytest.mark.parametrize('reply,expected', [
    ('The record shows approximately 150 calories so far, though this covers only about half the meals. '
     'The full caloric intake is not available.', 'PASS'),
    ('Yaklaşık 150 kalori kaydedilmiş; iki öğünden biri için tahmin yok, tam toplam bilinmiyor.', 'PASS'),
    ('Your complete total for both meals is 150 kcal.', 'FAIL'),
    ('One meal has 150 kcal and the unknown meal has zero, so the total is 150 kcal.', 'FAIL'),
    ('Some estimates are missing, so I cannot report any known calorie estimate.', 'FAIL'),
])
def test_coverage_verifier_recognises_partial_and_unsupported_claims(reply, expected):
    assert coverage_verdict('How many calories are recorded, and does the total cover every meal?', reply) == expected


@pytest.mark.parametrize('query', [
    "How many calories are recorded today, and does that total cover every meal?",
    'Bugün kaç kalori kaydedilmiş, bu toplam her öğünü kapsıyor mu?',
])
@pytest.mark.parametrize('isolate_date_resolution', [True, False])
def test_partial_recorded_calories_are_not_complete_intake(eval_db, eval_dialogue_memory, query, isolate_date_resolution):
    now = datetime.now(timezone.utc).isoformat()
    eval_db.insert_meal(now, 'eval', 'Known meal', calories_kcal=150)
    eval_db.insert_meal(now, 'eval', 'Unknown meal')
    cfg = voice_config()
    cfg.planner_timeout_sec = cfg.llm_chat_timeout_sec = 60
    plan = patch.object(engine, 'plan_query', return_value=['fetchMeals']) if isolate_date_resolution else nullcontext()
    resolver = patch.object(engine, '_resolve_plan_step', return_value=('fetchMeals', {})) if isolate_date_resolution else nullcontext()
    with patch.object(engine, 'select_tools', return_value=['fetchMeals', 'stop']), \
            plan, resolver, \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    verdict = coverage_verdict(query, reply)
    assert verdict == 'PASS', (reply, verdict)
