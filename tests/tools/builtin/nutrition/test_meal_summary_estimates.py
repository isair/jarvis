"""Meal summaries distinguish missing estimates from measured zero amounts."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jarvis.tools.builtin.nutrition.fetch_meals import FetchMealsTool

pytestmark = pytest.mark.unit


def fetch(db):
    return FetchMealsTool().run(None, SimpleNamespace(db=db, user_print=lambda text: None))


@pytest.mark.parametrize('unavailable', [None, float('inf'), -1, 'not a number'])
@pytest.mark.parametrize('field,label,unit', [
    ('calories_kcal', 'kcal', ' kcal'), ('protein_g', 'protein', 'g P'),
    ('carbs_g', 'carbs', 'g C'), ('fat_g', 'fat', 'g F'),
])
def test_partial_estimates_are_labelled_without_fabricated_zero(db, unavailable, field, label, unit):
    now = datetime.now(timezone.utc).isoformat()
    db.insert_meal(now, 'fixture', 'Known meal', **{field: 150})
    db.insert_meal(now, 'fixture', 'Unknown meal', **{field: unavailable})
    result = fetch(db)
    assert result.success
    assert f'150{unit} (1/2 meals estimated; full {label} total unavailable)' in result.reply_text
    unknown = next(line for line in result.reply_text.splitlines() if 'Unknown meal' in line)
    assert f'{label} unavailable' in unknown
    assert f'0{unit}' not in unknown


def test_absent_estimates_cannot_claim_zero_intake(db):
    db.insert_meal(datetime.now(timezone.utc).isoformat(), 'fixture', 'Unknown meal')
    result = fetch(db)
    assert result.success
    assert 'Total kcal unavailable' in result.reply_text
    assert '~0 kcal' not in result.reply_text


def test_explicit_zero_and_empty_range_keep_zero_totals(db):
    empty = fetch(db)
    assert empty.success and 'Total ~0 kcal' in empty.reply_text
    db.insert_meal(datetime.now(timezone.utc).isoformat(), 'fixture', 'Zero estimate', calories_kcal=0)
    result = fetch(db)
    assert result.success and 'Total ~0 kcal' in result.reply_text
    assert '(1/1 meals estimated)' not in result.reply_text


def test_unrepresentable_legacy_total_does_not_crash_retrieval(db):
    now = datetime.now(timezone.utc).isoformat()
    for description in ['First legacy meal', 'Second legacy meal']:
        db.insert_meal(now, 'fixture', description, calories_kcal=1e308)
    result = fetch(db)
    assert result.success
    assert 'Total kcal unavailable' in result.reply_text
    assert 'First legacy meal' in result.reply_text and 'Second legacy meal' in result.reply_text
