"""Meal deletion resolves unique descriptions without guessing a record."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory.db import Database
from jarvis.tools.builtin.nutrition.delete_meal import DeleteMealTool
from jarvis.tools.builtin.nutrition.fetch_meals import FetchMealsTool

pytestmark = pytest.mark.unit


@pytest.fixture
def context():
    db = Database(':memory:', sqlite_vss_path=None)
    yield SimpleNamespace(db=db, user_print=lambda *args: None)
    db.close()


def insert(context, description):
    return context.db.insert_meal(datetime.now(timezone.utc).isoformat(), 'test', description)


def rows(context):
    now = datetime.now(timezone.utc)
    return context.db.get_meals_between((now - timedelta(hours=1)).isoformat(),
                                        (now + timedelta(hours=1)).isoformat())


@pytest.mark.parametrize('description', ['Big Mac', 'mercimek çorbası', "oats'); DELETE FROM meals; --"])
def test_unique_description_deletes_only_that_meal(context, description):
    insert(context, description)
    keep = insert(context, 'other meal')
    result = DeleteMealTool().run({'id': description}, context)
    assert result.success, result.reply_text
    assert [row['id'] for row in rows(context)] == [keep]


@pytest.mark.parametrize('reference', ['Big', 'that meal', '', None, True, 1.9, pytest.param('1' * 5000, id='oversized-decimal')])
def test_unresolved_or_invalid_reference_preserves_meals(context, reference):
    keep = insert(context, 'Big Mac')
    result = DeleteMealTool().run({'id': reference}, context)
    assert not result.success
    assert [row['id'] for row in rows(context)] == [keep]


def test_duplicate_description_requires_an_id(context):
    first = insert(context, 'Big Mac')
    second = insert(context, 'Big Mac')
    result = DeleteMealTool().run({'id': 'Big Mac'}, context)
    assert not result.success
    assert 'ID' in result.reply_text
    assert [row['id'] for row in rows(context)] == [first, second]
    assert DeleteMealTool().run({'id': second}, context).success
    assert [row['id'] for row in rows(context)] == [first]


def test_fetch_meals_exposes_ids_for_disambiguation(context):
    mid = insert(context, 'Big Mac')
    result = FetchMealsTool().run({}, context)
    assert result.success
    assert f'#{mid}' in result.reply_text
    assert 'Big Mac' in result.reply_text


def test_explicit_description_argument_deletes_the_named_meal(context):
    keep = insert(context, 'Soup')
    insert(context, 'Big Mac')
    result = DeleteMealTool().run({'meal_description': 'Big Mac'}, context)
    assert result.success
    assert [row['id'] for row in rows(context)] == [keep]


def test_conflicting_references_preserve_every_record(context):
    soup = insert(context, 'Soup')
    burger = insert(context, 'Big Mac')
    result = DeleteMealTool().run({'id': soup, 'meal_description': 'Big Mac'}, context)
    assert not result.success
    assert [row['id'] for row in rows(context)] == [soup, burger]


def test_numeric_description_is_not_a_record_id(context):
    insert(context, '7')
    keep = insert(context, 'Soup')
    assert DeleteMealTool().run({'meal_description': '7'}, context).success
    assert [row['id'] for row in rows(context)] == [keep]


@pytest.mark.parametrize('reference', [1, True, {'name': 'Big Mac'}])
def test_description_requires_text(context, reference):
    keep = insert(context, 'Big Mac')
    assert not DeleteMealTool().run({'meal_description': reference}, context).success
    assert [row['id'] for row in rows(context)] == [keep]


def test_literal_description_step_deletes_without_model_resolution(context, monkeypatch):
    from jarvis.reply import planner
    keep = insert(context, 'Soup')
    insert(context, 'Big Mac')
    tool = DeleteMealTool()
    schema = [{'type': 'function', 'function': {'name': tool.name,
               'description': tool.description, 'parameters': tool.inputSchema}}]
    def unavailable(**kwargs):
        raise TimeoutError('Synthetic local model unavailable')
    monkeypatch.setattr(planner, 'call_llm_direct', unavailable)
    call = planner.resolve_next_tool_call(SimpleNamespace(llm_chat_model='synthetic'),
                                         "deleteMeal meal_description='Big Mac'", [], schema)
    assert call is not None
    assert tool.run(call[1], context).success
    assert [row['id'] for row in rows(context)] == [keep]
