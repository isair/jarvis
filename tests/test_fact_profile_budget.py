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
    original = GraphMemoryStore.get_children

    def guarded_children(graph, node_id):
        if node_id == 'user':
            attempts.append('read beyond full profile')
            raise AssertionError('No output budget remains for descendants')
        return original(graph, node_id)

    monkeypatch.setattr(GraphMemoryStore, 'get_children', guarded_children)
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
    initialise, get_node, get_children = (GraphMemoryStore.__init__, GraphMemoryStore.get_node,
                                         GraphMemoryStore.get_children)

    def owned_graph(graph, *args, **kwargs):
        initialise(graph, *args, **kwargs)
        connections.append(graph.conn)

    def signal_abandonment(graph, node_id):
        node = get_node(graph, node_id)
        if node_id == 'user':
            if reason == 'cancel':
                cancelled.set()
            else:
                clock[0] = control.deadline
        return node

    def guarded_children(graph, node_id):
        if cancelled.is_set() or clock[0] >= control.deadline:
            violations.append('graph read after abandonment')
            raise AssertionError('Query no longer active')
        return get_children(graph, node_id)

    monkeypatch.setattr(GraphMemoryStore, '__init__', owned_graph)
    monkeypatch.setattr(GraphMemoryStore, 'get_node', signal_abandonment)
    monkeypatch.setattr(GraphMemoryStore, 'get_children', guarded_children)
    expected = ExecutionCancelled if reason == 'cancel' else ExecutionDeadlineExceeded
    with pytest.raises(expected):
        build_fact_warm_profile(profile_path, check_active=control.check)
    assert violations == []
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')
