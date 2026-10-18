"""Profile collection stops when no more whole lines can be delivered."""
from contextlib import closing
import threading
import sqlite3
from unittest.mock import patch

import pytest

from jarvis.memory.db import Database
from jarvis.memory.facts import FactStore, build_fact_warm_profile
from jarvis.memory.graph import GraphMemoryStore
from jarvis.reply import engine
from jarvis.reply.execution import ExecutionControl, ExecutionCancelled, ExecutionDeadlineExceeded

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def profile_path(tmp_path):
    path = str(tmp_path / 'profile.db')
    with closing(FactStore(path)):
        pass
    with closing(GraphMemoryStore(path)) as graph:
        graph.append_to_node('user', 'One complete note\nAnother complete note')
    return path


def test_full_profile_does_not_read_descendants(profile_path, monkeypatch):
    attempts = []
    original = GraphMemoryStore.get_child_ids

    def guarded_children(graph, node_id, **kwargs):
        if node_id == 'user':
            attempts.append('read beyond full profile')
            raise AssertionError('No output budget remains for descendants')
        return original(graph, node_id, **kwargs)

    monkeypatch.setattr(GraphMemoryStore, 'get_child_ids', guarded_children)
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None, user_max_chars=len('One complete note'))
    assert profile['legacy'].split('\n', 1)[1] == 'One complete note'
    assert attempts == [], 'A swallowed read violation crossed the profile budget'


def test_zero_legacy_budget_does_not_open_graph(profile_path, monkeypatch):
    attempts = []

    def denied_graph(*args, **kwargs):
        attempts.append('opened graph with no output budget')
        raise AssertionError('No legacy output budget')

    monkeypatch.setattr('jarvis.memory.graph.GraphMemoryStore', denied_graph)
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None, user_max_chars=0)
    assert profile['legacy'] == ''
    assert attempts == []


def test_profile_keeps_complete_lines_that_fit(profile_path):
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None, user_max_chars=len('One complete note'))
    assert profile['legacy'] == 'Unverified legacy memory (reference only):\nOne complete note'


@pytest.mark.parametrize('reason', ['cancel', 'deadline'])
def test_reply_can_abandon_a_blocked_profile_without_output(
    profile_path, mock_config, dialogue_memory, reason,
):
    mock_config.db_path = profile_path
    mock_config.location_enabled = False
    mock_config.mcps = {}
    mock_config.embedding_model = ''
    mock_config.memory_enrichment_source = 'none'
    mock_config.agentic_preparation = 'staged'
    mock_config.agentic_query_timeout_sec = 1 if reason == 'deadline' else 10
    cancelled, entered, release, finished = (threading.Event() for _ in range(4))
    profile_workers, outcomes = [], []

    def blocked_profile(*args, **kwargs):
        profile_workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5), 'Owned profile probe was not released'
        return {'user': '', 'directives': '', 'legacy': ''}

    def query():
        try:
            with closing(Database(profile_path, sqlite_vss_path=None)) as db:
                outcomes.append(engine.run_reply_engine(
                    db, mock_config, None, 'Check the saved profile.', dialogue_memory,
                    cancel_event=cancelled,
                ))
        finally:
            finished.set()

    thread = threading.Thread(target=query, name='owned-profile-query')
    with patch('jarvis.memory.facts.build_fact_warm_profile', side_effect=blocked_profile), \
         patch.object(engine, 'plan_query', return_value=[]), \
         patch.object(engine, 'select_tools', return_value=[]), \
         patch.object(engine, 'extract_search_params_for_memory', return_value={}), \
         patch.object(engine, 'chat_with_messages', side_effect=AssertionError('No model call after profile abandonment')):
        try:
            thread.start()
            assert entered.wait(3), 'Query did not reach the owned profile reader'
            if reason == 'cancel':
                cancelled.set()
            assert finished.wait(2), 'Query remained blocked on an abandoned profile read'
            assert outcomes == [None] if reason == 'cancel' else outcomes and isinstance(outcomes[0], str)
            assert dialogue_memory.get_recent_messages() == []
        finally:
            release.set()
            thread.join(3)
            for worker in profile_workers:
                worker.join(3)
            assert not thread.is_alive()
            assert all(not worker.is_alive() for worker in profile_workers)


@pytest.mark.parametrize('reason', ['cancel', 'deadline'])
def test_profile_stops_reading_and_closes_its_graph_after_abandonment(profile_path, monkeypatch, reason):
    cancelled = threading.Event()
    clock = [0.0]
    control = ExecutionControl(cancelled, timeout_sec=10, clock=lambda: clock[0])
    connections, violations = [], []
    initialise, read_text, get_children = (GraphMemoryStore.__init__, GraphMemoryStore.read_node_text,
                                          GraphMemoryStore.get_child_ids)

    def owned_graph(graph, *args, **kwargs):
        initialise(graph, *args, **kwargs)
        connections.append(graph.conn)

    def signal_abandonment(graph, node_id, **kwargs):
        node = read_text(graph, node_id, **kwargs)
        if node_id == 'user':
            if reason == 'cancel':
                cancelled.set()
            else:
                clock[0] = control.deadline
        return node

    def guarded_children(graph, node_id, **kwargs):
        if cancelled.is_set() or clock[0] >= control.deadline:
            violations.append('graph read after abandonment')
            raise AssertionError('Query no longer active')
        return get_children(graph, node_id, **kwargs)

    monkeypatch.setattr(GraphMemoryStore, '__init__', owned_graph)
    monkeypatch.setattr(GraphMemoryStore, 'read_node_text', signal_abandonment)
    monkeypatch.setattr(GraphMemoryStore, 'get_child_ids', guarded_children)
    expected = ExecutionCancelled if reason == 'cancel' else ExecutionDeadlineExceeded
    with pytest.raises(expected):
        build_fact_warm_profile(profile_path, check_active=control.check)
    assert violations == []
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')


@pytest.mark.parametrize('prefix', ['Useful note', ''])
def test_unusable_remainder_cannot_scan_the_whole_graph(profile_path, monkeypatch, prefix):
    from jarvis.memory import facts
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_NODES', 8, raising=False)
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_CHARS', 10000 if prefix else 128, raising=False)
    with closing(GraphMemoryStore(profile_path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data=? WHERE id='user'", (prefix,))
        for index in range(80):
            graph.create_node(f'Note {index}', '', parent_id='user', data='A note too long to fit')
        graph.conn.commit()
    checks = []
    def active():
        checks.append(None)
        if len(checks) > 100:
            raise ExecutionDeadlineExceeded('Optional profile scanning used the reply allowance')
    profile = build_fact_warm_profile(profile_path, check_active=active,
                                      user_max_chars=len(prefix) + 2 if prefix else 2)
    assert profile['legacy'] == ('Unverified legacy memory (reference only):\n' + prefix if prefix else '')


def test_profile_bounds_database_text_delivery_for_a_wide_branch(profile_path, monkeypatch):
    from jarvis.memory import facts
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_NODES', 8, raising=False)
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_CHARS', 128, raising=False)
    with closing(GraphMemoryStore(profile_path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data='' WHERE id='user'")
        for index in range(40):
            graph.create_node(f'Note {index}', '', parent_id='user', data='Useful note\n' + 'x' * 10000)
        graph.conn.commit()
    delivered = []
    initialise = GraphMemoryStore.__init__
    def observed_graph(graph, *args, **kwargs):
        initialise(graph, *args, **kwargs)
        def row_factory(cursor, values):
            for column, value in zip(cursor.description, values):
                if isinstance(value, str) and column[0] not in {'id', 'parent_id'}:
                    delivered.append(len(value))
            return sqlite3.Row(cursor, values)
        graph.conn.row_factory = row_factory
    monkeypatch.setattr(GraphMemoryStore, '__init__', observed_graph)
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None, user_max_chars=len('Useful note'))
    assert profile['legacy'].endswith('\nUseful note')
    assert sum(delivered) <= facts.LEGACY_PROFILE_MAX_CHARS + facts.LEGACY_PROFILE_MAX_NODES


def test_partial_line_at_scan_limit_never_becomes_a_saved_note(profile_path, monkeypatch):
    from jarvis.memory import facts
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_CHARS', 20, raising=False)
    with closing(GraphMemoryStore(profile_path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data=? WHERE id='user'", ('Complete note\nNever truncate this instruction',))
        graph.conn.commit()
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None)
    assert profile['legacy'] == 'Unverified legacy memory (reference only):\nComplete note'


@pytest.mark.parametrize('reason', ['allowance', 'cancel', 'deadline'])
def test_sqlite_scan_interrupt_preserves_profile_or_propagates_query_stop(profile_path, monkeypatch, reason):
    from jarvis.memory import facts
    with closing(GraphMemoryStore(profile_path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data='Useful note' WHERE id='user'")
        for index in range(1000):
            graph.create_node(f'Child {index}', '', parent_id='user', data='Other note')
        graph.conn.commit()
    monkeypatch.setattr(facts, 'LEGACY_PROFILE_MAX_READ_CHECKPOINTS', 16)
    connections = []
    started_child_query = threading.Event()
    initialise = GraphMemoryStore.__init__
    def owned_graph(graph, *args, **kwargs):
        initialise(graph, *args, **kwargs)
        connections.append(graph.conn)
        def trace(statement):
            if statement.startswith('SELECT id FROM memory_nodes WHERE parent_id'):
                started_child_query.set()
        graph.conn.set_trace_callback(trace)
    monkeypatch.setattr(GraphMemoryStore, '__init__', owned_graph)
    observed = []
    cancelled = threading.Event()
    clock = [0.0]
    control = ExecutionControl(cancelled, timeout_sec=10, clock=lambda: clock[0])
    def active():
        if started_child_query.is_set():
            observed.append('Activity check after child query began')
            if reason == 'cancel':
                cancelled.set()
            if reason == 'deadline':
                clock[0] = control.deadline
        control.check()
    if reason == 'allowance':
        profile = build_fact_warm_profile(profile_path, check_active=active)
        assert profile['legacy'] == 'Unverified legacy memory (reference only):\nUseful note'
    else:
        expected = ExecutionCancelled if reason == 'cancel' else ExecutionDeadlineExceeded
        with pytest.raises(expected):
            build_fact_warm_profile(profile_path, check_active=active)
    assert observed, 'The owned cancellation boundary must follow the start of a child query'
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')


def test_profile_does_not_cut_a_saved_line_at_embedded_nul(profile_path):
    note = 'Useful prefix\0remaining instruction'
    with closing(GraphMemoryStore(profile_path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data=? WHERE id='user'", (note,))
        graph.conn.commit()
    profile = build_fact_warm_profile(profile_path, check_active=lambda: None)
    assert profile['legacy'] == 'Unverified legacy memory (reference only):\n' + note


@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16le', 'UTF-16be'])
def test_bounded_reader_preserves_unicode_and_nul_under_allocation_ceiling(tmp_path, encoding):
    path = str(tmp_path / 'encoded.db')
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA encoding=' + repr(encoding))
        connection.execute('CREATE TABLE encoding_anchor(value TEXT)')
    note = 'café🦉\0complete note\n' + 'x' * 100000
    with closing(GraphMemoryStore(path)) as graph:
        graph.conn.execute("UPDATE memory_nodes SET data=? WHERE id='user'", (note,))
        graph.conn.commit()
        graph.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65536)
        for cap in (1, 2, 4, 5, 6, 32):
            assert graph.read_node_text('user', max_chars=cap, check_active=lambda: None) == note[:cap]


def test_blob_text_stays_bound_to_node_during_viewer_replacement(profile_path):
    with closing(GraphMemoryStore(profile_path)) as reader:
        reader.conn.execute('PRAGMA journal_mode=WAL')
        original = reader.create_node('Owned user note', '', 'Original user note', 'user')
        replacements = []
        def replace_after_identity_lookup(cursor, values):
            row = sqlite3.Row(cursor, values)
            if cursor.description[0][0] == 'rowid' and not replacements:
                with closing(GraphMemoryStore(profile_path)) as writer:
                    writer.delete_node(original.id)
                    replacement = writer.create_node('World replacement', '', 'Unrelated world note', 'world')
                    reused = writer.conn.execute('SELECT rowid FROM memory_nodes WHERE id=?', (replacement.id,)).fetchone()[0]
                    assert reused == row[0], 'The owned fixture must exercise actual rowid reuse'
                    replacements.append(replacement.id)
            return row
        reader.conn.row_factory = replace_after_identity_lookup
        result = reader.read_node_text(original.id, max_chars=128, check_active=lambda: None)
        assert replacements
        assert result == 'Original user note'
        assert not reader.conn.in_transaction


@pytest.mark.parametrize('outer_transaction', [False, True])
def test_cancelled_node_snapshot_releases_only_its_own_transaction(profile_path, outer_transaction):
    with closing(GraphMemoryStore(profile_path)) as reader:
        if outer_transaction:
            reader.conn.execute("UPDATE memory_nodes SET data='Uncommitted owned text' WHERE id='user'")
        cancelled = threading.Event()
        control = ExecutionControl(cancelled, timeout_sec=10)
        def active():
            if reader.conn.in_transaction:
                cancelled.set()
            control.check()
        with pytest.raises(ExecutionCancelled):
            reader.read_node_text('user', max_chars=128, check_active=active)
        assert reader.conn.in_transaction is outer_transaction
        if outer_transaction:
            reader.conn.rollback()
        reader.conn.execute("UPDATE memory_nodes SET data='Subsequent write works' WHERE id='user'")
        reader.conn.commit()
        assert reader.get_node('user').data == 'Subsequent write works'
