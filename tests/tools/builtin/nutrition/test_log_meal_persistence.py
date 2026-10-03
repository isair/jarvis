"""A saved meal remains successful when optional coaching is unavailable."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory.db import Database
from jarvis.tools.builtin.nutrition import log_meal

pytestmark = pytest.mark.unit


@pytest.fixture
def meal_context():
    db = Database(':memory:', sqlite_vss_path=None)
    cfg = SimpleNamespace(llm_chat_model='synthetic-model', llm_chat_timeout_sec=8,
                          llm_thinking_enabled=False, use_stdin=True)
    yield SimpleNamespace(db=db, cfg=cfg, redacted_text='I had eggs',
                          max_retries=2, user_print=lambda *args: None)
    db.close()


def saved_meals(db):
    now = datetime.now(timezone.utc)
    return db.get_meals_between((now - timedelta(minutes=1)).isoformat(),
                                (now + timedelta(minutes=1)).isoformat())


@pytest.mark.parametrize('coaching', [TimeoutError('synthetic timeout'), '', None])
def test_saved_meal_is_confirmed_once_without_coaching(monkeypatch, meal_context, coaching):
    meal = {'description': 'eggs', 'calories_kcal': 150, 'protein_g': 12}
    def direct(**kwargs):
        if kwargs['system_prompt'] == log_meal.NUTRITION_SYS:
            return json.dumps(meal)
        if isinstance(coaching, Exception):
            raise coaching
        return coaching
    monkeypatch.setattr(log_meal, 'call_llm_direct', direct)

    result = log_meal.LogMealTool().run({}, meal_context)

    assert result.success, result.reply_text
    rows = saved_meals(meal_context.db)
    assert len(rows) == 1, 'Coaching must not repeat a committed meal'
    assert rows[0]['description'] == meal['description']
    assert f'Logged meal #{rows[0]["id"]}' in result.reply_text
    assert meal['description'] in result.reply_text
    assert 'Follow-ups:' not in result.reply_text


@pytest.mark.parametrize('macro', ['not a number', 'NaN', 'Infinity'])
def test_invalid_optional_macro_does_not_duplicate_a_saved_meal(monkeypatch, meal_context, macro):
    meal = {'description': 'eggs', 'calories_kcal': macro, 'protein_g': 12}
    monkeypatch.setattr(log_meal, 'call_llm_direct',
                        lambda **kwargs: json.dumps(meal) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else 'Drink water.')
    result = log_meal.LogMealTool().run({}, meal_context)
    rows = saved_meals(meal_context.db)
    assert result.success, result.reply_text
    assert len(rows) == 1
    assert rows[0]['calories_kcal'] is None
    assert rows[0]['protein_g'] == meal['protein_g']
    assert 'kcal' not in result.reply_text


@pytest.mark.parametrize('answer', ['[]', 'null', 'true', '"eggs"'])
def test_non_object_extraction_is_a_graceful_failure(monkeypatch, meal_context, answer):
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: answer)
    assert log_meal.extract_and_log_meal(meal_context.db, meal_context.cfg,
                                        meal_context.redacted_text, 'stdin') is None
    assert not saved_meals(meal_context.db)


def test_failed_write_can_retry_before_a_meal_is_saved(monkeypatch, meal_context):
    meal = {'description': 'eggs', 'calories_kcal': 150}
    monkeypatch.setattr(log_meal, 'call_llm_direct',
                        lambda **kwargs: json.dumps(meal) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else 'Drink water.')
    insert = meal_context.db.insert_meal
    unavailable = True
    def insert_after_recovery(**fields):
        nonlocal unavailable
        if unavailable:
            unavailable = False
            raise RuntimeError('synthetic write failure before commit')
        return insert(**fields)
    monkeypatch.setattr(meal_context.db, 'insert_meal', insert_after_recovery)
    result = log_meal.LogMealTool().run({}, meal_context)
    assert result.success, result.reply_text
    assert len(saved_meals(meal_context.db)) == 1
    assert 'Follow-ups: Drink water.' in result.reply_text


def test_missing_description_confirmation_matches_the_saved_meal(monkeypatch, meal_context):
    meal = {'calories_kcal': 150}
    monkeypatch.setattr(log_meal, 'call_llm_direct',
                        lambda **kwargs: json.dumps(meal) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    result = log_meal.LogMealTool().run({}, meal_context)
    rows = saved_meals(meal_context.db)
    assert result.success and len(rows) == 1
    assert f'Logged meal #{rows[0]["id"]}: {rows[0]["description"]}:' in result.reply_text
