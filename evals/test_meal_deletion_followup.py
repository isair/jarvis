"""Real reply generation deletes the named meal through direct or fallback calls."""
from datetime import datetime, timedelta, timezone
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('resolver_available', [True, False])
def test_delete_recent_named_meal(eval_db, eval_dialogue_memory, resolver_available):
    now = datetime.now(timezone.utc)
    keep_before = eval_db.insert_meal(now.isoformat(), 'eval', 'Other meal')
    meal_id = eval_db.insert_meal(now.isoformat(), 'eval', 'Big Mac')
    keep_after = eval_db.insert_meal(now.isoformat(), 'eval', 'Soup')
    cfg = voice_config()
    cfg.location_enabled = False
    cfg.planner_timeout_sec = 60
    cfg.llm_chat_timeout_sec = 60
    cfg.tool_result_digest_enabled = False
    eval_dialogue_memory.add_message('user', 'I just ate a Big Mac.')
    eval_dialogue_memory.add_message('assistant', 'I have recorded the Big Mac, about 550 calories.')
    # The resolver can fail open. Measure the stored records after the complete
    # reply rather than requiring a particular intermediate plan or tool call.
    fallback = nullcontext() if resolver_available else patch.object(engine, '_resolve_plan_step', return_value=None)
    with patch.object(engine, 'select_tools', return_value=['deleteMeal', 'fetchMeals', 'stop']), fallback:
        reply = engine.run_reply_engine(eval_db, cfg, None, 'Delete that meal.', eval_dialogue_memory)
    assert reply, '🗑️ Deletion must produce a reply'
    rows = eval_db.get_meals_between((now - timedelta(minutes=1)).isoformat(),
                                    (now + timedelta(minutes=1)).isoformat())
    assert [row['id'] for row in rows] == [keep_before, keep_after], f'🗑️ Incorrect records after deleting #{meal_id}: {[(r["id"], r["description"]) for r in rows]}'
