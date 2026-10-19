"""Committed diary text remains usable when its optional index fails."""
from datetime import datetime, timezone
import sqlite3
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.fixture
def diary(tmp_path):
    db = Database(str(tmp_path / 'diary.db'), sqlite_vss_path=None)
    # Ordinary tables exercise the VSS write transaction without an extension.
    db.conn.executescript('''
        CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, vec TEXT);
        CREATE TABLE summary_vec (summary_id INTEGER PRIMARY KEY, emb_id INTEGER);
        CREATE TRIGGER reject_index BEFORE INSERT ON summary_vec BEGIN
            SELECT RAISE(ABORT, 'synthetic index failure');
        END;
    ''')
    db.is_vss_enabled = True
    yield db
    db.close()


def assert_readable_and_unlocked(diary, ident):
    with sqlite3.connect(diary.db_path, timeout=0.01) as reader:
        row = reader.execute('SELECT summary FROM conversation_summaries WHERE id=?', (ident,)).fetchone()
        assert row and 'Celsius' in row[0]
        assert reader.execute('SELECT count(*) FROM embeddings').fetchone()[0] == 0
        # An independent writer can proceed after the optional index failure.
        reader.execute('CREATE TABLE IF NOT EXISTS writer_probe (value TEXT)')
        reader.execute("INSERT INTO writer_probe VALUES ('available')")
    hits = diary.search_hybrid('Celsius', None)
    assert any(row['id'] == ident for row in hits)


def test_failed_vector_mapping_rolls_back_only_the_index_write(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'The user prefers Celsius.', 'preferences')
    with pytest.raises(sqlite3.IntegrityError):
        diary.upsert_summary_embedding(ident, [0.1, 0.2], diary.get_summary_embedding_text(ident))
    assert_readable_and_unlocked(diary, ident)


@pytest.mark.parametrize('owned_flush', [False, True], ids=['direct', 'dialogue'])
@pytest.mark.parametrize('failure', ['storage', 'backend'])
def test_optional_index_failure_still_confirms_the_diary(diary, monkeypatch, owned_flush, failure):
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
    memory = conversation.DialogueMemory()
    memory.add_message('user', 'I prefer Celsius.')
    late = 'My dog is called Pip.'
    def embed(*args, **kwargs):
        memory.add_message('user', late)
        if failure == 'backend':
            raise TimeoutError('synthetic embedding timeout')
        return [0.1, 0.2]
    monkeypatch.setattr(conversation, 'get_embedding_backend', lambda cfg: SimpleNamespace(embed=embed))
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs:
                        'SUMMARY: The user prefers Celsius.\nTOPICS: preferences, temperature')
    graph_summaries = []
    def graph(**kwargs):
        graph_summaries.append(kwargs['summary'])
        return SimpleNamespace(stored=[], skipped=0)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue', graph)
    if owned_flush:
        ident = conversation.update_diary_from_dialogue_memory(diary, memory, cfg, force=True)
    else:
        ident = conversation.update_daily_conversation_summary(diary, memory.get_pending_chunks(), cfg)
    assert ident is not None
    assert_readable_and_unlocked(diary, ident)
    if owned_flush:
        assert memory.get_pending_chunks() == [f'User: {late}']
        assert graph_summaries == ['The user prefers Celsius.']


def test_failed_text_write_keeps_the_previous_diary_and_pending_messages(diary, monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    ident = diary.upsert_conversation_summary(today, 'The user prefers Celsius.', 'preferences')
    diary.conn.executescript('''CREATE TRIGGER reject_diary BEFORE INSERT ON conversation_summaries
        BEGIN SELECT RAISE(ABORT, 'synthetic diary failure'); END;''')
    memory = conversation.DialogueMemory()
    memory.add_message('user', 'My dog is called Pip.')
    pending = memory.get_pending_chunks()
    cfg = SimpleNamespace(llm_chat_model='synthetic', embedding_model='synthetic')
    monkeypatch.setattr(conversation, '_direct_llm', lambda *args, **kwargs:
                        'SUMMARY: The user prefers Celsius and has a dog called Pip.\nTOPICS: pets, preferences')
    assert conversation.update_diary_from_dialogue_memory(diary, memory, cfg, force=True) is None
    assert memory.get_pending_chunks() == pending
    with sqlite3.connect(diary.db_path) as reader:
        assert reader.execute('SELECT summary FROM conversation_summaries WHERE id=?', (ident,)).fetchone()[0] == 'The user prefers Celsius.'


def test_failed_refresh_preserves_the_previous_vector_mapping(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'The user prefers Celsius.', 'preferences')
    diary.conn.execute('DROP TRIGGER reject_index')
    previous = diary.upsert_summary_embedding(ident, [0.5, 0.6], diary.get_summary_embedding_text(ident))
    diary.conn.executescript('''CREATE TRIGGER reject_index BEFORE INSERT ON summary_vec
        BEGIN SELECT RAISE(ABORT, 'synthetic refresh failure'); END;''')
    with pytest.raises(sqlite3.IntegrityError):
        diary.upsert_summary_embedding(ident, [0.1, 0.2], diary.get_summary_embedding_text(ident))
    with sqlite3.connect(diary.db_path, timeout=0.01) as reader:
        assert reader.execute('SELECT emb_id FROM summary_vec WHERE summary_id=?', (ident,)).fetchone()[0] == previous
        assert reader.execute('SELECT count(*) FROM embeddings').fetchone()[0] == 1
        reader.execute("INSERT INTO meals(ts_utc, source_app, description) VALUES ('2026-01-01', 'test', 'meal')")


def test_successful_vector_write_is_visible_after_reopening(diary):
    ident = diary.upsert_conversation_summary('2026-01-01', 'The user prefers Celsius.', 'preferences')
    diary.conn.execute('DROP TRIGGER reject_index')
    emb_id = diary.upsert_summary_embedding(ident, [0.1, 0.2], diary.get_summary_embedding_text(ident))
    with sqlite3.connect(diary.db_path) as reader:
        assert reader.execute('SELECT emb_id FROM summary_vec WHERE summary_id=?', (ident,)).fetchone()[0] == emb_id
        assert reader.execute('SELECT vec FROM embeddings WHERE rowid=?', (emb_id,)).fetchone()[0] == '[0.1, 0.2]'
