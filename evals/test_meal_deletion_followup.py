"""A real planner resolves a meal follow-up to a deletable record."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.planner import plan_query, resolve_next_tool_call
from jarvis.tools.builtin.nutrition.delete_meal import DeleteMealTool

pytestmark = [pytest.mark.eval, requires_judge_llm]


def test_delete_recent_named_meal(eval_db):
    now = datetime.now(timezone.utc)
    meal_id = eval_db.insert_meal(now.isoformat(), 'eval', 'Big Mac')
    keep = eval_db.insert_meal(now.isoformat(), 'eval', 'Soup')
    tool = DeleteMealTool()
    cfg = voice_config()
    dialogue = 'user: I just ate a Big Mac.\nassistant: I have recorded the Big Mac, about 550 calories.'
    plan = plan_query(cfg, 'Delete that meal.', dialogue,
                      [(tool.name, tool.description)], timeout_sec=60)
    step = next((step for step in plan if tool.name in step), None)
    assert step, f'🗑️ Missing deletion step: {plan}'
    schema = [{'type': 'function', 'function': {'name': tool.name,
               'description': tool.description, 'parameters': tool.inputSchema}}]
    call = resolve_next_tool_call(cfg, step, [], schema, timeout_sec=60)
    assert call, f'🗑️ Unresolved meal deletion: {step}'
    result = tool.run(call[1], SimpleNamespace(db=eval_db, user_print=lambda *args: None))
    assert result.success, f'🗑️ Meal deletion failed: {call}, {result.reply_text}'
    rows = eval_db.get_meals_between((now - timedelta(minutes=1)).isoformat(),
                                    (now + timedelta(minutes=1)).isoformat())
    assert [row['id'] for row in rows] == [keep], f'🗑️ Incorrect remaining records after deleting #{meal_id}'
