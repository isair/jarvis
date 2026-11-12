"""A rejected vector write preserves both live and persisted candidates."""
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jarvis.utils.fast_vector_store import FAISSVectorStore
from jarvis.utils.vector_store import PythonVectorStore
from jarvis.memory import conversation
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'faiss'])
def store(request, tmp_path):
    path = str(tmp_path / 'vectors.db')
    return PythonVectorStore(path) if request.param == 'python' else FAISSVectorStore(path, 2)


def reopen(store):
    return (PythonVectorStore(store.db_path) if isinstance(store, PythonVectorStore)
            else FAISSVectorStore(store.db_path, store.dimension))


@pytest.mark.parametrize('summary_id', [1, 2], ids=['replace', 'append'])
def test_rejected_write_preserves_memory_and_releases_writer_for_retry(store, summary_id):
    store.add_vector(1, [1., 0.])
    table = 'python_vector_store' if isinstance(store, PythonVectorStore) else 'faiss_vector_store'
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(f"CREATE TRIGGER reject_embedding BEFORE INSERT ON {table} "
                     "BEGIN SELECT RAISE(ABORT, 'embedding rejected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_vector(summary_id, [0., 1.])
    for current in (store, reopen(store)):
        assert current.search([1., 0.]) == [pytest.approx((1, 0.))]
    # Removing the constraint and retrying exercises rollback and lock release.
    with sqlite3.connect(store.db_path, timeout=0) as conn:
        conn.execute('DROP TRIGGER reject_embedding')
    store.add_vector(summary_id, [0., 1.])
    for current in (store, reopen(store)):
        assert current.search([0., 1.])[0] == pytest.approx((summary_id, 0.))


@pytest.mark.parametrize('operation', ['save', 'rewrite', 'topics'])
def test_diary_refresh_failure_retains_text_and_previous_vector(store, monkeypatch, operation):
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store', lambda path, dimension: store)
    db = Database(store.db_path)
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        original = 'The user prefers Celsius.'
        if operation == 'rewrite':
            original += ' The assistant could not help.'
        ident = db.upsert_conversation_summary(today, original, 'temperature')
        store.add_vector(ident, [1., 0.])
        table = 'python_vector_store' if isinstance(store, PythonVectorStore) else 'faiss_vector_store'
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(f"CREATE TRIGGER reject_embedding BEFORE INSERT ON {table} "
                         "BEGIN SELECT RAISE(ABORT, 'embedding rejected'); END")
        cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
        monkeypatch.setattr(conversation, '_embed_text', lambda *args, **kwargs: [0., 1.])
        if operation == 'save':
            monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs:
                                'SUMMARY: The user prefers Celsius.\nTOPICS: temperature')
            assert conversation.update_daily_conversation_summary(db, ['User: I prefer Celsius.'], cfg)
        else:
            reply = 'The user prefers Celsius.' if operation == 'rewrite' else '{"temperature": "preferences"}'
            monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs: reply)
            refresh = (conversation.rewrite_all_diary_summaries if operation == 'rewrite'
                       else conversation.optimise_diary_topics)
            assert list(refresh(db, cfg))[0]['embedding_refreshed'] is False
        assert any('Celsius' in row['text'] for row in db.search_hybrid('Celsius', None))
        for current in (store, reopen(store)):
            assert current.search([1., 0.])[0] == pytest.approx((ident, 0.))
        with sqlite3.connect(store.db_path, timeout=0) as conn:
            conn.execute('CREATE TABLE writer_probe (value TEXT)')
    finally:
        db.close()
