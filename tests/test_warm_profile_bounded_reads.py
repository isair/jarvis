"""Graph text remains usable under SQLite's per-value allocation ceiling."""
from contextlib import closing
import sqlite3

import pytest

from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.graph_ops import build_warm_profile

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


def _expected(text, budget):
    text = text.strip()
    return text if len(text) <= budget else text[:max(0, budget-1)].rstrip()+'…'


@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16le', 'UTF-16be'])
@pytest.mark.parametrize('branch', ['user', 'directives'])
def test_large_notes_survive_a_sqlite_allocation_ceiling(tmp_path, encoding, branch):
    path = str(tmp_path/'graph.db')
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA encoding='+repr(encoding))
        connection.execute('CREATE TABLE encoding_anchor(value TEXT)')
    ceiling = 16*1024
    note = 'café🦉\0remaining instruction\n'+'x'*(ceiling*8)
    with closing(GraphMemoryStore(path)) as store:
        store.conn.execute('UPDATE memory_nodes SET data=? WHERE id=?', (note, branch))
        store.conn.commit()
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, ceiling)
        for budget in (1, 2, 5, 32):
            profile = build_warm_profile(store, user_max_chars=budget, directives_max_chars=budget)
            assert profile[branch] == _expected(note, budget)
        assert not store.conn.in_transaction


def test_profile_does_not_load_descendant_payloads_after_its_budget_fills(tmp_path):
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        first = 'A complete first note.'
        store.update_node('user', data=first)
        store.create_node('Large child', 'Owned', 'x'*(128*1024), 'user')
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16*1024)
        assert build_warm_profile(store, user_max_chars=len(first))['user'] == first


@pytest.mark.parametrize('budget', [0, -1])
def test_no_output_budget_does_not_materialise_stored_notes(tmp_path, budget):
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        store.update_node('user', data='x'*(128*1024))
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16*1024)
        assert build_warm_profile(store, user_max_chars=budget, directives_max_chars=budget) == {
            'user': '', 'directives': '',
        }


@pytest.mark.parametrize('text', [
    ' \t\nLeading and trailing spaces. \n',
    '\u2003café🦉\0Unicode boundaries.\u2003',
    'First line\nSecond line\n',
    'x'*4095+'🦉after a chunk boundary',
    ' '*100000+'Useful text.'+' '*100000,
    ' \n\t'*40000,
])
@pytest.mark.parametrize('budget', [1, 5, 32])
def test_profile_keeps_its_existing_text_and_whitespace_contract(tmp_path, text, budget):
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        store.update_node('user', data=text)
        assert build_warm_profile(store, user_max_chars=budget)['user'] == _expected(text, budget)


def test_profile_read_preserves_the_callers_pending_edit(tmp_path):
    path = str(tmp_path/'graph.db')
    with closing(GraphMemoryStore(path)) as store:
        original = store.get_node('user').data
        store.conn.execute("UPDATE memory_nodes SET data='Pending caller edit' WHERE id='user'")
        assert build_warm_profile(store)['user'] == 'Pending caller edit'
        assert store.conn.in_transaction
        with sqlite3.connect(path) as observer:
            assert observer.execute("SELECT data FROM memory_nodes WHERE id='user'").fetchone()[0] == original
        store.conn.rollback()
        assert store.get_node('user').data == original


def test_large_child_text_does_not_materialise_during_sibling_ordering(tmp_path):
    note = 'Useful child note.\n'+'x'*(128*1024)
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        store.create_node('Large child', 'Owned', note, 'user')
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16*1024)
        assert build_warm_profile(store, user_max_chars=32)['user'] == _expected(note, 32)


def test_streamed_text_stays_bound_to_its_selected_node_during_rowid_reuse(tmp_path):
    path = str(tmp_path/'graph.db')
    with closing(GraphMemoryStore(path)) as reader:
        reader.conn.execute('PRAGMA journal_mode=WAL')
        original = reader.create_node('Selected', 'Owned', 'Selected personal note.', 'user')
        replacements = []
        def replace_after_selection(cursor, values):
            row = sqlite3.Row(cursor, values)
            if cursor.description[0][0] == 'rowid' and not replacements:
                with closing(GraphMemoryStore(path)) as writer:
                    writer.delete_node(original.id)
                    replacement = writer.create_node('Unrelated', 'Owned', 'Unrelated world note.', 'world')
                    reused = writer.conn.execute(
                        'SELECT rowid FROM memory_nodes WHERE id=?', (replacement.id,),
                    ).fetchone()[0]
                    assert reused == row[0], 'The owned interleaving must exercise real identity reuse.'
                    replacements.append(replacement.id)
            return row
        reader.conn.row_factory = replace_after_selection
        with reader.open_node_text(original.id) as chunks:
            assert ''.join(chunks) == original.data
        assert replacements, 'The concurrent owned replacement must run during selection.'
        assert not reader.conn.in_transaction


@pytest.mark.parametrize('outer_transaction', [False, True])
def test_abandoned_text_stream_releases_only_its_own_snapshot(tmp_path, outer_transaction):
    path = str(tmp_path/'graph.db')
    with closing(GraphMemoryStore(path)) as reader:
        if outer_transaction:
            reader.conn.execute("UPDATE memory_nodes SET data='Pending caller edit' WHERE id='user'")
        with pytest.raises(RuntimeError, match='Owned consumer stopped'):
            with reader.open_node_text('user') as chunks:
                list(chunks)
                raise RuntimeError('Owned consumer stopped')
        assert reader.conn.in_transaction is outer_transaction
        if outer_transaction:
            assert reader.get_node('user').data == 'Pending caller edit'
            with sqlite3.connect(path) as observer:
                assert observer.execute("SELECT data FROM memory_nodes WHERE id='user'").fetchone()[0] == ''
            reader.conn.rollback()
        with sqlite3.connect(path, timeout=0.1) as writer:
            writer.execute("UPDATE memory_nodes SET data='Subsequent edit' WHERE id='user'")
        assert reader.get_node('user').data == 'Subsequent edit'


@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16le', 'UTF-16be'])
def test_decoder_retains_a_character_split_between_blob_chunks(tmp_path, encoding):
    path = str(tmp_path/'graph.db')
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA encoding='+repr(encoding))
        connection.execute('CREATE TABLE encoding_anchor(value TEXT)')
    padding = 4095 if encoding == 'UTF-8' else 2047
    text = ' '*padding+'🦉café\0complete'
    with closing(GraphMemoryStore(path)) as store:
        store.update_node('user', data=text)
        for budget in (1, 5, 32):
            assert build_warm_profile(store, user_max_chars=budget)['user'] == _expected(text, budget)


def test_profile_streams_can_share_an_existing_readonly_transaction(tmp_path):
    path = str(tmp_path/'graph.db')
    with closing(GraphMemoryStore(path)) as writer:
        writer.update_node('user', data='A saved note.')
    with closing(GraphMemoryStore(path, read_only=True)) as reader:
        reader.conn.execute('BEGIN')
        assert build_warm_profile(reader)['user'] == 'A saved note.'
        assert reader.conn.in_transaction
        assert reader.conn.total_changes == 0
        reader.conn.rollback()
        with reader.open_node_text('missing') as chunks:
            assert chunks is None
        assert not reader.conn.in_transaction
