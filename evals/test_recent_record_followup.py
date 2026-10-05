"""A follow-up deletes the logged record even when its label is duplicated."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query, correction', [
    ('I ate a Big Mac.', 'Delete that meal actually.'),
    ('Mercimek çorbası içtim.', 'Aslında o öğünü sil.'),
])
@pytest.mark.parametrize('resolver_available', [True, False])
def test_delete_logged_record_with_duplicate_label(eval_db, eval_dialogue_memory, query, correction, resolver_available):
    cfg = voice_config()
    cfg.location_enabled = False
    cfg.tool_result_digest_enabled = True
    cfg.planner_timeout_sec = cfg.llm_chat_timeout_sec = 60
    cfg.llm_digest_timeout_sec = 60
    now = datetime.now(timezone.utc)
    description = 'Big Mac' if 'Big Mac' in query else 'mercimek çorbası'
    older = eval_db.insert_meal((now - timedelta(hours=1)).isoformat(), 'eval', description)
    def digest(**kwargs):
        if kwargs['tool_name'] == 'logMeal':
            return 'The tool reports that the meal was logged with estimated nutrition.'
        return kwargs['tool_result']
    with patch.object(engine, 'select_tools', return_value=['logMeal', 'deleteMeal', 'fetchMeals', 'stop']), \
         patch.object(engine, 'digest_tool_result_for_query', side_effect=digest):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
        assert reply
        meals = eval_db.get_meals_between((now - timedelta(hours=2)).isoformat(),
                                         (now + timedelta(hours=1)).isoformat())
        logged = [r['id'] for r in meals if r['id'] != older]
        assert len(logged) == 1, f'🥗 Meal was not logged once: {len(logged)}'
        recent_id = logged[0]
        saved = next(row for row in meals if row['id'] == recent_id)
        # Match the stored label even when extraction normalises its spelling.
        eval_db.conn.execute('UPDATE meals SET description = ? WHERE id = ?',
                             (saved['description'], older))
        eval_db.conn.commit()
        # Ensure a database-wide "latest record" guess would delete the wrong meal.
        keep = eval_db.insert_meal(now.isoformat(), 'eval', 'Unrelated fixture meal')
        fallback = nullcontext() if resolver_available else patch.object(engine, '_resolve_plan_step', return_value=None)
        with fallback:
            reply = engine.run_reply_engine(eval_db, cfg, None, correction, eval_dialogue_memory)
        assert reply
    rows = eval_db.get_meals_between((now - timedelta(hours=2)).isoformat(),
                                    (now + timedelta(hours=1)).isoformat())
    assert [r['id'] for r in rows] == [older, keep], f'🗑️ Deletion did not target logged #{recent_id}: {[(r["id"], r["description"]) for r in rows]}'


@pytest.mark.parametrize('kind, identity, label, query', [
    ('note', 'note-k19', 'Shopping', 'Delete that note.'),
    ('appointment', 29, 'Dentist', 'O randevuyu sil.'),
])
def test_planner_uses_recorded_identity_for_other_resources(kind, identity, label, query):
    from jarvis.reply.planner import plan_query, resolve_next_tool_call, tool_steps_of
    cfg = voice_config()
    cfg.planner_timeout_sec = 60
    tool = 'deleteResource'
    schema = {'type':'object','properties':{'id':{'type':'string' if isinstance(identity,str) else 'integer'}},'required':['id']}
    dialogue = engine._planner_resource_context([{'role':'tool','tool_failed':False,
        'resource_references':[{'kind':kind,'id':identity,'label':label}]}])
    plan = plan_query(cfg, query, dialogue, [(tool, 'Delete the specified recorded resource using its id.')])
    steps = tool_steps_of(plan)
    assert steps
    tools = [{'type':'function','function':{'name':tool,'description':'Delete a recorded resource','parameters':schema}}]
    call = resolve_next_tool_call(cfg, steps[0], [], tools)
    assert call == (tool, {'id':identity}), f'🧾 Identity not retained: {plan}, {call}'
