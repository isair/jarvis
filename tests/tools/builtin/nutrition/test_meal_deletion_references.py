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


@pytest.mark.parametrize('reference', ['Big', 'that meal', '', None, True, 1.9])
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
