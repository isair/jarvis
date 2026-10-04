"""Updating diary text retains its semantic references and full-text index."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from jarvis.memory.db import Database, _SCHEMA_SQL
from jarvis.utils.fast_vector_store import FAISSVectorStore
from jarvis.utils.vector_store import PythonVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'faiss'])
def diary(request, tmp_path, monkeypatch):
    kind = PythonVectorStore if request.param == 'python' else FAISSVectorStore
    def make_store(path, dimension):
        return kind(path) if kind is PythonVectorStore else kind(path, 2)
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store', make_store)
    db = Database(str(tmp_path / 'diary.db'))
    yield db
    db.close()


def assert_fts_consistent(db):
    db.conn.execute("INSERT INTO summaries_fts(summaries_fts, rank) VALUES('integrity-check', 1)")
    db.conn.commit()


def test_updated_summary_retains_semantic_reference_after_reopening(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'The user likes cycling.', 'sports')
    diary.upsert_summary_embedding(ident, [1., 0.])
    stamp = '2026-01-01T12:30:00+00:00'
    updated = diary.upsert_conversation_summary(
        '2026-01-01', 'The user likes cycling and prefers Celsius.', 'sports, temperature', ts_utc=stamp,
    )
    assert updated == ident
    assert diary.get_conversation_summary('2026-01-01')['ts_utc'] == stamp
    reopened = Database(diary.db_path)
    try:
        for current in (diary, reopened):
            hits = current.search_hybrid('bicycle', json.dumps([1., 0.]))
            assert hits and hits[0]['id'] == ident and 'Celsius' in hits[0]['text']
    finally:
        reopened.close()


def test_repeated_updates_keep_fts_consistent(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'cycling')
    for term in ('temperature', 'coffee', 'passport'):
        assert diary.upsert_conversation_summary('2026-01-01', term) == ident
        assert_fts_consistent(diary)
        assert [row['id'] for row in diary.search_hybrid(term, None)] == [ident]
        assert diary.search_hybrid('cycling', None) == []


def test_distinct_days_and_sources_retain_independent_identities(diary):
    first = diary.upsert_conversation_summary('2026-01-01', 'cycling', source_app='voice')
    second = diary.upsert_conversation_summary('2026-01-01', 'coffee', source_app='chat')
    third = diary.upsert_conversation_summary('2026-01-02', 'passport', source_app='voice')
    assert len({first, second, third}) == 3
    assert diary.upsert_conversation_summary('2026-01-01', 'temperature', source_app='voice') == first
    assert diary.get_conversation_summary('2026-01-01', 'chat')['id'] == second
    assert diary.get_conversation_summary('2026-01-02', 'voice')['id'] == third


def test_failed_text_update_releases_writer_and_preserves_previous_memory(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'cycling')
    diary.upsert_summary_embedding(ident, [1., 0.])
    diary.conn.executescript("""CREATE TRIGGER reject_diary BEFORE INSERT ON conversation_summaries
        WHEN new.summary = 'rejected' BEGIN SELECT RAISE(ABORT, 'diary rejected'); END;""")
    with pytest.raises(sqlite3.IntegrityError):
        diary.upsert_conversation_summary('2026-01-01', 'rejected')
    assert diary.get_conversation_summary('2026-01-01')['summary'] == 'cycling'
    assert diary.search_hybrid('bicycle', '[1, 0]')[0]['id'] == ident
    with sqlite3.connect(diary.db_path, timeout=0) as conn:
        conn.execute('DROP TRIGGER reject_diary')
    assert diary.upsert_conversation_summary('2026-01-01', 'coffee') == ident
    assert_fts_consistent(diary)


def test_two_active_database_owners_keep_one_summary_identity(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'cycling')
    other = Database(diary.db_path)
    ready = Barrier(2)
    def update(db, term):
        ready.wait(timeout=2)
        return db.upsert_conversation_summary('2026-01-01', term)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(update, diary, 'coffee'), pool.submit(update, other, 'passport')]
            assert [future.result(timeout=5) for future in futures] == [ident, ident]
        assert len(diary.get_all_conversation_summaries()) == 1
        assert_fts_consistent(diary)
    finally:
        other.close()


def test_existing_diary_full_text_entries_are_rebuilt_on_open(tmp_path):
    path = str(tmp_path / 'legacy.db')
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        for text in ('cycling', 'coffee'):
            conn.execute("INSERT OR REPLACE INTO conversation_summaries "
                         "(date_utc, ts_utc, summary, source_app) VALUES (?, ?, ?, ?)",
                         ('2026-01-01', '2026-01-01T12:00:00+00:00', text, 'jarvis'))
    db = Database(path)
    try:
        assert_fts_consistent(db)
        assert 'coffee' in db.search_hybrid('coffee', None)[0]['text']
        assert db.search_hybrid('cycling', None) == []
    finally:
        db.close()


def test_summary_update_preserves_foreign_key_embedding_mapping(diary):
    diary.conn.execute('PRAGMA foreign_keys=ON')
    diary.conn.executescript('''
        CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, vec TEXT);
        CREATE TABLE summary_vec (
            summary_id INTEGER PRIMARY KEY REFERENCES conversation_summaries(id) ON DELETE CASCADE,
            emb_id INTEGER
        );
    ''')
    diary.is_vss_enabled = True
    ident = diary.upsert_conversation_summary('2026-01-01', 'cycling')
    embedding = diary.upsert_summary_embedding(ident, [1., 0.])
    diary.upsert_conversation_summary('2026-01-01', 'cycling and coffee')
    with sqlite3.connect(diary.db_path) as conn:
        assert conn.execute('SELECT summary_id, emb_id FROM summary_vec').fetchall() == [(ident, embedding)]


def test_completed_rebuild_is_not_repeated_on_reopen(tmp_path, monkeypatch):
    path = str(tmp_path / 'diary.db')
    db = Database(path)
    db.upsert_conversation_summary('2026-01-01', 'cycling')
    db.close()
    connect = sqlite3.connect
    def guarded_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_authorizer(lambda action, table, *rest:
                            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == 'summaries_fts'
                            else sqlite3.SQLITE_OK)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', guarded_connect)
    reopened = Database(path)
    try:
        assert 'cycling' in reopened.search_hybrid('cycling', None)[0]['text']
    finally:
        reopened.close()


def test_failed_rebuild_releases_writer_and_can_retry(tmp_path, monkeypatch):
    path = str(tmp_path / 'diary.db')
    connect = sqlite3.connect
    def guarded_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_authorizer(lambda action, table, *rest:
                            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == 'summaries_fts'
                            else sqlite3.SQLITE_OK)
        return conn
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, 'connect', guarded_connect)
        with pytest.raises(sqlite3.DatabaseError):
            Database(path)
    with connect(path, timeout=0) as conn:
        assert conn.execute('SELECT count(*) FROM diary_index_migrations').fetchone()[0] == 0
        conn.execute("INSERT INTO conversation_summaries(date_utc,ts_utc,summary,source_app) "
                     "VALUES ('2026-01-01','2026-01-01T12:00:00+00:00','cycling','jarvis')")
    reopened = Database(path)
    try:
        assert_fts_consistent(reopened)
        assert 'cycling' in reopened.search_hybrid('cycling', None)[0]['text']
    finally:
        reopened.close()


def test_concurrent_first_owners_complete_the_rebuild(tmp_path):
    path = str(tmp_path / 'diary.db')
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        for text in ('cycling', 'coffee'):
            conn.execute("INSERT OR REPLACE INTO conversation_summaries "
                         "(date_utc, ts_utc, summary, source_app) VALUES (?, ?, ?, ?)",
                         ('2026-01-01', '2026-01-01T12:00:00+00:00', text, 'jarvis'))
    ready = Barrier(2)
    def open_diary():
        ready.wait(timeout=2)
        return Database(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(open_diary) for _ in range(2)]
        owners = [future.result(timeout=5) for future in futures]
    try:
        for db in owners:
            assert_fts_consistent(db)
            assert 'coffee' in db.search_hybrid('coffee', None)[0]['text']
    finally:
        for db in owners:
            db.close()
