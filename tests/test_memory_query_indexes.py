"""Date lookups use bounded index ranges on fresh and reopened diaries."""

from datetime import datetime, timedelta, timezone

import pytest

from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


def query_plan(connection, sql, parameters=()):
    return [row[3].upper() for row in connection.execute(
        'EXPLAIN QUERY PLAN ' + sql, parameters
    )]


def assert_range_search(plan, table):
    assert any(f'SEARCH {table.upper()} USING INDEX' in step for step in plan), plan
    assert not any('USE TEMP B-TREE' in step for step in plan), plan


@pytest.mark.parametrize('existing', [False, True])
def test_meal_range_is_indexed_and_preserves_rows_on_reopen(tmp_path, existing):
    path = tmp_path / 'diary.db'
    diary = Database(str(path))
    try:
        late = diary.insert_meal('2026-10-09T12:00:00Z', 'jarvis', 'Lunch')
        early = diary.insert_meal('2026-10-09T08:00:00Z', 'jarvis', 'Breakfast')
        diary.insert_meal('2026-10-08T08:00:00Z', 'jarvis', 'Yesterday')
        if existing:
            # A diary from before the date index still has its stored records.
            indexes = diary.conn.execute("PRAGMA index_list('meals')").fetchall()
            for index in indexes:
                diary.conn.execute('DROP INDEX "' + index['name'].replace('"', '""') + '"')
            diary.conn.commit()
            diary.close()
            diary = Database(str(path))
        bounds = ('2026-10-09T08:00:00Z', '2026-10-09T12:00:00Z')
        assert [row['id'] for row in diary.get_meals_between(*bounds)] == [early, late]
        assert_range_search(query_plan(diary.conn,
            'SELECT * FROM meals WHERE ts_utc >= ? AND ts_utc <= ? ORDER BY ts_utc ASC',
            bounds), 'meals')
        diary.close()
        diary = Database(str(path))
        assert [row['id'] for row in diary.get_meals_between(*bounds)] == [early, late]
        assert diary.conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        diary.close()


def test_summary_date_paths_use_existing_unique_index(db):
    today = datetime.now(timezone.utc).date()
    old = (today - timedelta(days=30)).isoformat()
    recent = (today - timedelta(days=1)).isoformat()
    db.upsert_conversation_summary(old, 'Old record')
    newest = db.upsert_conversation_summary(today.isoformat(), 'Current record')
    previous = db.upsert_conversation_summary(recent, 'Recent record')
    assert [row['id'] for row in db.get_recent_conversation_summaries(7)] == [newest, previous]
    assert db.get_conversation_summary(today.isoformat())['id'] == newest
    for sql, params in [
        ('SELECT * FROM conversation_summaries WHERE date_utc >= ? ORDER BY date_utc DESC', (recent,)),
        ('SELECT * FROM conversation_summaries WHERE date_utc = ? AND source_app = ?', (recent, 'jarvis')),
    ]:
        assert_range_search(query_plan(db.conn, sql, params), 'conversation_summaries')
    indexes = db.conn.execute("PRAGMA index_list('conversation_summaries')").fetchall()
    date_only = [index for index in indexes if [row['name'] for row in
        db.conn.execute('PRAGMA index_info("' + index['name'].replace('"', '""') + '")')]
        == ['date_utc']]
    assert not date_only, 'The unique (date_utc, source_app) index already serves date queries'
