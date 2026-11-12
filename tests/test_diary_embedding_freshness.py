"""A delayed embedding cannot overwrite a newer diary refresh."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation
from jarvis.memory.db import Database
from jarvis.utils.fast_vector_store import get_faiss_vector_store
from jarvis.utils.vector_store import get_python_vector_store
from jarvis.utils.vector_store import PythonVectorStore
from jarvis.utils.fast_vector_store import FAISSVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(params=['python', 'faiss', 'vss'])
def owners(request, tmp_path, monkeypatch):
    factory = get_faiss_vector_store if request.param == 'faiss' else get_python_vector_store
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                        lambda path, dimension: factory(path, 2) if request.param == 'faiss' else factory(path))
    first = Database(str(tmp_path / 'diary.db'))
    if request.param == 'vss':
        first.conn.executescript('''
            CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, vec TEXT);
            CREATE TABLE summary_vec (summary_id INTEGER PRIMARY KEY, emb_id INTEGER);
        ''')
    second = Database(first.db_path)
    first.is_vss_enabled = second.is_vss_enabled = request.param == 'vss'
    yield first, second
    first.close()
    second.close()


def assert_current_vector(db, ident):
    if db.is_vss_enabled:
        row = db.conn.execute('SELECT e.vec FROM embeddings e JOIN summary_vec s ON e.rowid=s.emb_id '
                              'WHERE s.summary_id=?', (ident,)).fetchone()
        assert row and json.loads(row['vec']) == [1., 0.]
    else:
        assert db._python_vector_store.search([1., 0.], top_k=1)[0][0] == ident
        stored = db._python_vector_store
        fresh = (PythonVectorStore(db.db_path) if isinstance(stored, PythonVectorStore)
                 else FAISSVectorStore(db.db_path, stored.dimension))
        assert fresh.search([1., 0.], top_k=1)[0][0] == ident


@pytest.mark.parametrize('operation', ['save', 'rewrite', 'topics'])
def test_delayed_refresh_keeps_newer_vector_across_owners_and_reopen(owners, monkeypatch, operation):
    first, second = owners
    today = datetime.now(timezone.utc).date().isoformat()
    original = 'The user likes coffee.'
    if operation == 'rewrite':
        original += ' The assistant could not help.'
    ident = first.upsert_conversation_summary(today, original, 'drinks')
    first.upsert_summary_embedding(ident, [0., 1.], first.get_summary_embedding_text(ident))
    competitor = first.upsert_conversation_summary('2027-01-01', 'Unrelated cycling.', 'sports')
    first.upsert_summary_embedding(competitor, [0.6, 0.8], first.get_summary_embedding_text(competitor))
    old_reply = {'save': 'SUMMARY: The user likes coffee.\nTOPICS: drinks',
                 'rewrite': 'The user likes coffee.', 'topics': '{"drinks": "beverages"}'}[operation]
    replies = iter([old_reply, 'SUMMARY: The user renews a passport.\nTOPICS: travel'])
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs: next(replies, 'Unrelated cycling.'))
    entered, release = Event(), Event()
    def embed(text, *args, **kwargs):
        if 'coffee' in text:
            entered.set()
            assert release.wait(timeout=5)
            return [0., 1.]
        return [1., 0.]
    monkeypatch.setattr(conversation, '_embed_text', embed)
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
    def old_refresh():
        if operation == 'save':
            return conversation.update_daily_conversation_summary(first, ['User: I like coffee.'], cfg)
        refresh = (conversation.rewrite_all_diary_summaries if operation == 'rewrite'
                   else conversation.optimise_diary_topics)
        return list(refresh(first, cfg))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(old_refresh)
        try:
            assert entered.wait(timeout=2)
            current = conversation.update_daily_conversation_summary(second, ['User: I renew a passport.'], cfg)
            assert current == ident
            assert_current_vector(second, ident)
        finally:
            release.set()
        result = pending.result(timeout=5)
    assert 'passport' in second.get_conversation_summary(today)['summary']
    assert_current_vector(second, ident)
    if operation != 'save':
        assert next(event for event in result if event['date_utc'] == today)['embedding_refreshed'] is False
    reopened = Database(first.db_path)
    reopened.is_vss_enabled = first.is_vss_enabled
    try:
        assert_current_vector(reopened, ident)
    finally:
        reopened.close()


@pytest.mark.parametrize('backend', ['python', 'faiss'])
def test_content_change_at_vector_write_boundary_rejects_stale_refresh(tmp_path, monkeypatch, backend):
    factory = get_python_vector_store if backend == 'python' else get_faiss_vector_store
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                        lambda path, dimension: factory(path) if backend == 'python' else factory(path, 2))
    first = Database(str(tmp_path / 'diary.db'))
    second = Database(first.db_path)
    try:
        ident = first.upsert_conversation_summary('2026-01-01', 'coffee', 'drinks')
        source = first.get_summary_embedding_text(ident)
        first.upsert_summary_embedding(ident, [0., 1.], source)
        write = first._python_vector_store.add_summary_vector
        def racing_write(summary_id, vector, source_text):
            second.upsert_conversation_summary('2026-01-01', 'passport', 'travel')
            assert write(ident, [1., 0.], second.get_summary_embedding_text(ident))
            return write(summary_id, vector, source_text)
        monkeypatch.setattr(first._python_vector_store, 'add_summary_vector', racing_write)
        assert first.upsert_summary_embedding(ident, [0., 1.], source) is None
        assert first._python_vector_store.search([1., 0.])[0] == pytest.approx((ident, 0.))
    finally:
        first.close()
        second.close()


def test_deleted_summary_rejects_a_delayed_refresh(owners):
    first, second = owners
    ident = first.upsert_conversation_summary('2026-01-01', 'coffee')
    source = first.get_summary_embedding_text(ident)
    first.upsert_summary_embedding(ident, [0., 1.], source)
    second.conn.execute('DELETE FROM conversation_summaries WHERE id=?', (ident,))
    second.conn.commit()
    assert first.upsert_summary_embedding(ident, [1., 0.], source) is None


def test_missing_source_cannot_bypass_the_guard(owners):
    first, _ = owners
    ident = first.upsert_conversation_summary('2026-01-01', 'coffee')
    with pytest.raises(ValueError):
        first.upsert_summary_embedding(ident, [1., 0.], None)


def test_python_in_memory_diary_rejects_superseded_text(monkeypatch):
    monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store',
                        lambda path, dimension: get_python_vector_store(path))
    db = Database(':memory:')
    try:
        ident = db.upsert_conversation_summary('2026-01-01', 'coffee')
        source = db.get_summary_embedding_text(ident)
        assert db.upsert_summary_embedding(ident, [0., 1.], source) == ident
        db.upsert_conversation_summary('2026-01-01', 'passport')
        current = db.get_summary_embedding_text(ident)
        assert db.upsert_summary_embedding(ident, [1., 0.], current) == ident
        assert db.upsert_summary_embedding(ident, [0., 1.], source) is None
        assert db._python_vector_store.search([1., 0.])[0] == pytest.approx((ident, 0.))
    finally:
        db.close()


@pytest.mark.parametrize('kind', [PythonVectorStore, FAISSVectorStore])
def test_guarded_store_api_requires_a_persistent_diary_source(kind):
    store = kind(':memory:') if kind is PythonVectorStore else kind(':memory:', 2)
    with pytest.raises(ValueError):
        store.add_summary_vector(1, [1., 0.], 'coffee ')
