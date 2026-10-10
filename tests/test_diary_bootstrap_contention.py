"""Opening an initialised diary must not contend with an active save."""

from contextlib import closing
import sqlite3
import pytest
from jarvis.memory.db import Database

@pytest.mark.unit
@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_initialised_diary_can_open_while_another_owner_saves(tmp_path, monkeypatch, finish):
    path = str(tmp_path / 'diary.db')
    with closing(Database(path)) as initial:
        ident = initial.upsert_conversation_summary('2026-01-01', 'Cycling by the canal')
    connect = sqlite3.connect
    with closing(connect(path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        writer.execute('UPDATE conversation_summaries SET summary=? WHERE id=?', ('Uncommitted replacement', ident))
        def fast_connect(*args, **kwargs):
            kwargs['timeout'] = 0.01
            return connect(*args, **kwargs)
        monkeypatch.setattr(sqlite3, 'connect', fast_connect)
        with closing(Database(path)) as reopened:
            assert reopened.get_conversation_summary_by_id(ident)['summary'] == 'Cycling by the canal'
            assert 'Cycling' in reopened.search_hybrid('Cycling', None)[0]['text']
            getattr(writer, finish)()
            expected = 'Uncommitted replacement' if finish == 'commit' else 'Cycling by the canal'
            assert reopened.get_conversation_summary_by_id(ident)['summary'] == expected
            keyword = 'replacement' if finish == 'commit' else 'Cycling'
            assert reopened.search_hybrid(keyword, None)[0]['id'] == ident
            assert reopened.upsert_conversation_summary('2026-01-02', 'Coffee with a friend')
