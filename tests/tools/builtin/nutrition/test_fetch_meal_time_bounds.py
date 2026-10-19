"""Equivalent UTC meal ranges return the same records at inclusive boundaries."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.tools.builtin.nutrition.fetch_meals import FetchMealsTool

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('format_kind', ['canonical', 'zulu', 'space', 'offset', 'naive'])
def test_equivalent_ranges_return_boundary_records(db, format_kind):
    start = datetime(2030, 1, 15, 12, tzinfo=timezone.utc)
    end = start + timedelta(seconds=1)
    expected = ['At start', 'Inside', 'At end']
    entries = [
        (start - timedelta(seconds=1), 'Before'),
        (start, expected[0]),
        (start + timedelta(milliseconds=500), expected[1]),
        (end, expected[2]),
        (end + timedelta(seconds=1), 'After'),
    ]
    for instant, description in entries:
        db.insert_meal(instant.isoformat(), 'fixture', description)

    def encode(instant):
        if format_kind == 'zulu':
            return instant.isoformat().replace('+00:00', 'Z')
        if format_kind == 'space':
            return instant.isoformat(sep=' ')
        if format_kind == 'offset':
            return instant.astimezone(timezone(timedelta(hours=1))).isoformat()
        if format_kind == 'naive':
            return instant.replace(tzinfo=None).isoformat()
        return instant.isoformat()

    context = SimpleNamespace(db=db, user_print=lambda text: None)
    result = FetchMealsTool().run({'since_utc': encode(start), 'until_utc': encode(end)}, context)
    assert result.success
    assert all(description in result.reply_text for description in expected)
    assert 'Before' not in result.reply_text
    assert 'After' not in result.reply_text


@pytest.mark.parametrize('args', [
    {'since_utc': 'not-a-time'},
    {'until_utc': 'not-a-time'},
    {'since_utc': 42},
    {'until_utc': False},
    {'since_utc': '0001-01-01T00:00:00+01:00'},
    {'until_utc': '9999-12-31T23:59:59-01:00'},
    {'until_utc': '0001-01-01T00:00:00Z'},
    [],
    'invalid',
    False,
    {'since_utc': '2030-01-16T12:00:00Z', 'until_utc': '2030-01-15T12:00:00Z'},
])
def test_invalid_ranges_report_failure_instead_of_empty_intake(db, args):
    context = SimpleNamespace(db=db, user_print=lambda text: None)
    result = FetchMealsTool().run(args, context)
    assert not result.success
    assert result.error_message
    assert not result.reply_text or 'Meals: 0' not in result.reply_text


def test_whole_second_end_includes_explicit_zero_fraction(db):
    instant = datetime(2030, 1, 15, 12, tzinfo=timezone.utc)
    db.insert_meal(instant.isoformat(), 'fixture', 'Automatic precision')
    db.insert_meal(instant.isoformat(timespec='microseconds'), 'fixture', 'Explicit precision')
    db.insert_meal((instant + timedelta(microseconds=1)).isoformat(), 'fixture', 'Later')
    result = FetchMealsTool().run(
        {'since_utc': instant.isoformat(), 'until_utc': instant.isoformat()},
        SimpleNamespace(db=db, user_print=lambda text: None),
    )
    assert result.success
    assert 'Automatic precision' in result.reply_text
    assert 'Explicit precision' in result.reply_text
    assert 'Later' not in result.reply_text


@pytest.mark.parametrize('args', [None, {}, {'since_utc': '', 'until_utc': ''}])
def test_missing_bounds_select_last_day(db, args):
    now = datetime.now(timezone.utc)
    db.insert_meal((now - timedelta(hours=1)).isoformat(), 'fixture', 'Recent meal')
    db.insert_meal((now - timedelta(days=2)).isoformat(), 'fixture', 'Older meal')
    result = FetchMealsTool().run(args, SimpleNamespace(db=db, user_print=lambda text: None))
    assert result.success
    assert 'Recent meal' in result.reply_text
    assert 'Older meal' not in result.reply_text


def test_missing_start_is_one_day_before_explicit_end(db):
    end = datetime(2030, 1, 15, 12, tzinfo=timezone.utc)
    db.insert_meal((end - timedelta(hours=1)).isoformat(), 'fixture', 'In requested day')
    db.insert_meal((end - timedelta(days=2)).isoformat(), 'fixture', 'Before requested day')
    result = FetchMealsTool().run(
        {'until_utc': end.isoformat()}, SimpleNamespace(db=db, user_print=lambda text: None),
    )
    assert result.success
    assert 'In requested day' in result.reply_text
    assert 'Before requested day' not in result.reply_text
