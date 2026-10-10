"""Only a validated user meal decision may reach persistence."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from jarvis.memory.db import Database
from jarvis.tools.builtin.nutrition import log_meal

pytestmark = pytest.mark.unit


@pytest.fixture
def context():
    db = Database(':memory:', sqlite_vss_path=None)
    cfg = SimpleNamespace(llm_chat_model='chat-model', fast_model='fast-model',
                          llm_chat_timeout_sec=8, llm_thinking_enabled=False, use_stdin=True)
    yield SimpleNamespace(db=db, cfg=cfg, redacted_text='Should I eat eggs?',
                          max_retries=2, user_print=lambda *args: None)
    db.close()


def meals(context):
    now = datetime.now(timezone.utc)
    return context.db.get_meals_between((now - timedelta(minutes=1)).isoformat(),
                                       (now + timedelta(minutes=1)).isoformat())


@pytest.mark.parametrize('decision', [
    {'record': False}, {}, {'record': 'false', 'meal': 'eggs'},
    {'record': True, 'meal': ''}, {'record': True, 'meal': None},
    {'record': False, 'meal': 'eggs'}, {'record': True, 'meal': 'eggs', 'extra': 'untrusted'},
])
def test_negative_or_invalid_request_decision_cannot_write_a_meal(monkeypatch, context, decision):
    answers = iter([json.dumps(decision), json.dumps({'description': 'invented eggs', 'calories_kcal': 150}), 'Drink water.'])
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: next(answers))
    result = log_meal.LogMealTool().run({'meal': 'eggs'}, context)
    assert not result.success
    assert not meals(context), 'A negative or unknown source decision must not become a meal record'
    assert not result.resource_references
    assert bool(result.error_message) is (decision != {'record': False})


def test_actual_request_guides_approved_estimation_and_storage(monkeypatch, context):
    context.redacted_text = 'I ate eggs; my partner ate a Big Mac.'
    def direct(**kwargs):
        if kwargs['chat_model'] == context.cfg.fast_model:
            return json.dumps({'record': True})
        if kwargs['system_prompt'] == log_meal.NUTRITION_SYS:
            source = kwargs['user_content'].split('<<<BEGIN UNTRUSTED USER TEXT>>>\n', 1)[1].split('\n<<<END UNTRUSTED USER TEXT>>>', 1)[0]
            data = json.loads(source)
            return json.dumps({'description': 'eggs' if data['user_request'] == context.redacted_text else 'eggs and Big Mac', 'calories_kcal': 150})
        return 'Drink water.'
    monkeypatch.setattr(log_meal, 'call_llm_direct', direct)
    result = log_meal.LogMealTool().run({}, context)
    assert result.success, result.reply_text
    rows = meals(context)
    assert len(rows) == 1 and rows[0]['description'] == 'eggs'
    assert result.resource_references[0]['id'] == rows[0]['id']


@pytest.mark.parametrize('field', ['request', 'description'])
def test_oversized_source_cannot_hide_a_denial_and_create_a_meal(monkeypatch, context, field):
    text = 'eggs ' * 300 + 'I did not eat this, do not record it.'
    context.redacted_text = text if field == 'request' else ''
    answers = iter(['{"record":true}', '{"description":"eggs"}', 'Drink water.'])
    monkeypatch.setattr(log_meal, 'call_llm_direct', lambda **kwargs: next(answers))
    result = log_meal.LogMealTool().run({'meal': text if field == 'description' else 'eggs'}, context)
    assert not result.success
    assert result.error_message
    assert not meals(context)
    assert not result.resource_references


@pytest.mark.parametrize('raw', [None, '', 'not JSON', '[]', '{"record":0}', RuntimeError('offline backend')])
def test_unavailable_recording_decision_returns_an_honest_failure(monkeypatch, context, raw):
    def direct(**kwargs):
        if isinstance(raw, Exception):
            raise raw
        return raw
    monkeypatch.setattr(log_meal, 'call_llm_direct', direct)
    result = log_meal.LogMealTool().run({'meal': 'eggs'}, context)
    assert not result.success
    assert result.error_message
    assert not meals(context)
    assert not result.resource_references
