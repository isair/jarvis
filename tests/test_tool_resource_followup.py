"""Successful tool writes retain exact record identity through summaries."""
import json
from unittest.mock import patch

import pytest
from jarvis.reply import engine
from jarvis.tools.builtin.nutrition import log_meal
from jarvis.tools.base import ToolContext

pytestmark = pytest.mark.unit


def test_logged_record_reference_is_grounded_in_saved_row(db, mock_config, monkeypatch):
    monkeypatch.setattr(log_meal, 'meal_recording_requested', lambda *args: True)
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: json.dumps({'description':'Oats'}) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    ctx = ToolContext(db, mock_config, '', '', 'I ate oats', 0, lambda text: None)
    result = log_meal.LogMealTool().run({}, ctx)
    assert result.success
    refs = result.resource_references
    assert len(refs) == 1, '🥗 A successful write must expose its record identity'
    row = db.conn.execute('SELECT id, description FROM meals').fetchone()
    assert refs[0]['id'] == row['id']
    assert refs[0]['label'] == row['description']


def test_digest_does_not_erase_followup_identity(db, mock_config, dialogue_memory, monkeypatch):
    mock_config.location_enabled = False
    mock_config.tool_result_digest_enabled = True
    now = '2026-10-05T00:00:00+00:00'
    keep = db.insert_meal(now, 'fixture', 'Oats')
    monkeypatch.setattr(log_meal, 'meal_recording_requested', lambda *args: True)
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: json.dumps({'description':'Oats'}) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    def plan(**kwargs):
        if kwargs['query'] == 'Log oats':
            return ["logMeal meal='Oats'", 'Reply to the user.']
        # The planner fixture follows available recorded identity, never DB ordering.
        records = dialogue_memory.get_recent_turns_with_tools()
        refs = [r for m in records for r in m.get('resource_references', [])]
        if refs and str(refs[-1]['id']) in kwargs['dialogue_context']:
            return [f"deleteMeal id={refs[-1]['id']}", 'Reply to the user.']
        return ["deleteMeal meal_description='Oats'", 'Reply to the user.']
    def reply_chat(cfg, messages, **kwargs):
        assert not any(m.get('tool_calls') for m in messages), 'Text-tool history must use the text protocol'
        if messages[-1].get('tool_name') == 'deleteMeal':
            assert 'Recorded tool resources' in messages[0]['content']
        return {'message': {'content': 'Done.'}}
    with patch.object(engine, 'select_tools', return_value=['logMeal','deleteMeal','stop']), \
         patch.object(engine, 'plan_query', side_effect=plan), \
         patch('jarvis.reply.planner.call_llm_direct', side_effect=AssertionError('Concrete IDs need no model')),  \
         patch.object(engine, 'digest_tool_result_for_query', return_value='The meal contains estimated nutrition.'), \
         patch.object(engine, 'chat_with_messages', side_effect=reply_chat):
        assert engine.run_reply_engine(db, mock_config, None, 'Log oats', dialogue_memory)
        assert engine.run_reply_engine(db, mock_config, None, 'Delete that', dialogue_memory)
    assert [r['id'] for r in db.conn.execute('SELECT id FROM meals')] == [keep]


@pytest.mark.parametrize('kind, identity, label', [
    ('appointment', 'record-c71b', 'Dentist'),
    ('note', 27, '買い物'),
])
def test_references_survive_digest_for_arbitrary_resource_types(mock_config, kind, identity, label):
    from jarvis.tools.types import ToolExecutionResult
    ref = {'kind':kind, 'id':identity, 'label':label}
    result = ToolExecutionResult(True, 'Long result prose.', resource_references=(ref,))
    mock_config.tool_result_digest_enabled = True
    with patch.object(engine, 'digest_tool_result_for_query', return_value='A shortened result.'):
        content, records = engine._tool_result_content(mock_config, 'Record it', 'createResource', result)
    assert records == [ref]
    assert json.dumps([ref], ensure_ascii=False) in content
    assert 'A shortened result.' in content


def test_failed_and_unconfirmed_results_cannot_supply_resource_identity(mock_config):
    from jarvis.tools.types import ToolExecutionResult
    ref = {'kind':'note','id':27,'label':'Fixture'}
    mock_config.tool_result_digest_enabled = False
    content, records = engine._tool_result_content(mock_config, 'Create it', 'createNote',
        ToolExecutionResult(False, 'Not created.', resource_references=(ref,)))
    assert not records
    assert content == 'Not created.'
    for status in (True, None):
        assert engine._planner_resource_context([{'role':'tool','tool_failed':status,'resource_references':[ref]}]) == ''


def test_reference_carryover_scrubs_secrets_and_respects_conversation_reset(dialogue_memory):
    secret = 'sk-abcd1234567890abcdef'
    ref = {'kind':'note','id':27,'label':f'apikey: {secret}'}
    dialogue_memory.record_tool_turn([{'role':'tool','content':'Created.',
        'tool_failed':False,'resource_references':[ref]}])
    recent = dialogue_memory.get_recent_turns_with_tools(per_entry_chars=1)
    context = engine._planner_resource_context(recent)
    assert secret not in context
    assert '27' in context
    assert secret in ref['label'], 'The caller-owned record must remain unchanged'
    dialogue_memory.clear_tool_carryover()
    assert engine._planner_resource_context(dialogue_memory.get_recent_turns_with_tools()) == ''


def test_reference_budget_preserves_complete_newest_ids():
    refs = [{'kind':'note','id':f'id-{index}','label':'x' * 500}
            for index in range(engine._RESOURCE_REFERENCE_LIMIT * 3)]
    recent = [{'role':'tool','content':'Created.','tool_failed':False,'resource_references':refs}]
    context = engine._planner_resource_context(recent)
    records = json.loads(context.split('\n', 1)[1])
    assert [r['id'] for r in records] == [r['id'] for r in refs[-engine._RESOURCE_REFERENCE_LIMIT:]]
    assert all(len(r['label']) == engine._RESOURCE_LABEL_CHAR_LIMIT for r in records)


def test_native_tool_execution_preserves_identity_for_followup(db, mock_config, dialogue_memory, monkeypatch):
    from jarvis.reply.prompts.model_variants import ModelSize
    mock_config.location_enabled = False
    mock_config.tool_result_digest_enabled = True
    older = db.insert_meal('2026-10-05T00:00:00+00:00', 'fixture', 'Oats')
    monkeypatch.setattr(log_meal, 'meal_recording_requested', lambda *args: True)
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: json.dumps({'description':'Oats'}) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    def chat(cfg, messages, **kwargs):
        if messages[-1]['role'] == 'tool':
            return {'message':{'content':'Done.'}}
        if messages[-1]['content'] == 'Log oats':
            name, args = 'logMeal', {'meal':'Oats'}
        else:
            resource_rows = [r for m in messages for r in m.get('resource_references', [])]
            assert resource_rows
            ref = resource_rows[-1]
            assert any(json.dumps([ref],ensure_ascii=False) in m.get('content','') for m in messages)
            name, args = 'deleteMeal', {'id':ref['id']}
        return {'message':{'content':'','tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':args}}]}}
    with patch.object(engine, 'select_tools', return_value=['logMeal','deleteMeal','stop']), \
         patch.object(engine, 'detect_model_size', return_value=ModelSize.LARGE), \
         patch.object(engine, 'plan_query', return_value=[]), \
         patch.object(engine, 'digest_tool_result_for_query', return_value='Estimated nutrition.'), \
         patch.object(engine, 'chat_with_messages', side_effect=chat):
        assert engine.run_reply_engine(db, mock_config, None, 'Log oats', dialogue_memory)
        assert engine.run_reply_engine(db, mock_config, None, 'Delete that', dialogue_memory)
    assert [r['id'] for r in db.conn.execute('SELECT id FROM meals')] == [older]


@pytest.mark.parametrize('reference_key', ['id', 'meal_description'])
def test_delete_meal_ignores_unused_null_reference(db, mock_config, reference_key):
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.nutrition.delete_meal import DeleteMealTool
    older = db.insert_meal('2026-10-05T00:00:00+00:00', 'fixture', 'Keep')
    target = db.insert_meal('2026-10-05T00:01:00+00:00', 'fixture', 'Unique fixture')
    reference = {'id': None, 'meal_description': None}
    reference[reference_key] = target if reference_key == 'id' else 'Unique fixture'
    context = ToolContext(db, mock_config, '', '', '', 0, lambda _: None)
    assert DeleteMealTool().run(reference, context).success
    assert [r['id'] for r in db.conn.execute('SELECT id FROM meals')] == [older]
