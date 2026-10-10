"""A saved meal remains successful when optional coaching is unavailable."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory.db import Database
from jarvis.tools.builtin.nutrition import log_meal

pytestmark = pytest.mark.unit


@pytest.fixture
def meal_context(monkeypatch):
    monkeypatch.setattr(log_meal, 'meal_recording_requested', lambda *args: True)
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


@pytest.mark.parametrize('macro', ['not a number', 'NaN', 'Infinity', True, False])
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
                                        meal_context.redacted_text, 'stdin', request_text=meal_context.redacted_text) is None
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


@pytest.mark.parametrize('description', [None, '', '   ', True, 42, {'unexpected': 'value'}, ['eggs']])
def test_unavailable_description_uses_the_same_record_label(monkeypatch, meal_context, description):
    def answer(meal):
        monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs:
                            json.dumps(meal) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    answer({'calories_kcal': 150})
    baseline = log_meal.LogMealTool().run({}, meal_context)
    assert baseline.success
    default_label = baseline.resource_references[0]['label']

    answer({'description': description, 'calories_kcal': 150})
    result = log_meal.LogMealTool().run({}, meal_context)
    rows = saved_meals(meal_context.db)
    assert result.success and len(rows) == 2
    assert rows[-1]['description'] == default_label
    assert result.resource_references[0]['label'] == default_label
    assert f'Logged meal #{rows[-1]["id"]}: {default_label}:' in result.reply_text


def test_recorded_description_has_no_surrounding_whitespace(monkeypatch, meal_context):
    label = 'eggs with toast'
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs:
                        json.dumps({'description': '  ' + label + '  ', 'calories_kcal': 150})
                        if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    result = log_meal.LogMealTool().run({}, meal_context)
    rows = saved_meals(meal_context.db)
    assert result.success and len(rows) == 1
    assert rows[0]['description'] == label
    assert result.resource_references[0]['label'] == label


@pytest.mark.parametrize('answer', ['NONE', ' none ', '```text\nNONE\n```', '```json\nNONE\n```'])
def test_valid_no_meal_decision_cannot_retry_into_a_written_record(monkeypatch, meal_context, answer):
    responses = iter([answer, json.dumps({'description': 'invented meal', 'calories_kcal': 150}), 'Drink water.'])
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: next(responses))
    meal_context.redacted_text = 'A general food question, not a consumed meal.'
    result = log_meal.LogMealTool().run({}, meal_context)
    assert not result.success, result.reply_text
    assert not saved_meals(meal_context.db), 'A valid no-meal decision must not become an intake record'
    assert not result.resource_references


def test_malformed_extraction_can_recover_to_one_valid_record(monkeypatch, meal_context):
    meal = {'description': 'eggs', 'calories_kcal': 150}
    responses = iter(['{invalid', json.dumps(meal), 'Drink water.'])
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: next(responses))
    result = log_meal.LogMealTool().run({}, meal_context)
    assert result.success, result.reply_text
    rows = saved_meals(meal_context.db)
    assert len(rows) == 1
    assert rows[0]['description'] == meal['description']


@pytest.mark.parametrize('estimates', [{}, {'calories_kcal': 'NaN', 'protein_g': 'not a number'}])
def test_record_without_numeric_estimates_has_an_honest_confirmation(monkeypatch, meal_context, estimates):
    meal = {'description': 'eggs with toast', **estimates}
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs:
                        json.dumps(meal) if kwargs['system_prompt'] == log_meal.NUTRITION_SYS else '')
    result = log_meal.LogMealTool().run({}, meal_context)
    rows = saved_meals(meal_context.db)
    assert result.success
    assert len(rows) == 1
    assert rows[0]['calories_kcal'] is None
    assert rows[0]['protein_g'] is None
    assert 'nutrition estimates unavailable' in result.reply_text
    assert 'approximate macros logged' not in result.reply_text
