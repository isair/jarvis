"""Cancelling a request prevents subsequent preparation work."""
from threading import Event

import pytest
from jarvis.reply import engine
from jarvis.memory import conversation

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('stop_during,failed_stage', [
    (None, False), ('discovery', False), ('router_cache', False), ('router', False), ('planner', False),
    ('profile', False), ('keywords', False), ('diary', False), ('graph', False), ('digest', False),
    ('discovery', True), ('planner', True), ('keywords', True),
    ('diary', True), ('digest', True),
])
def test_preparation_stops_after_the_inflight_stage(monkeypatch, mock_config, db, dialogue_memory, tmp_path, stop_during, failed_stage):
    cancelled = Event()
    forbidden = tmp_path / 'work-started-after-cancellation'
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = True
    mock_config.memory_digest_enabled = True
    mock_config.tool_result_digest_enabled = False
    mock_config.tts_engine = 'none'

    def stage(name, value):
        if cancelled.is_set():
            forbidden.write_text(name)
        if name == stop_during:
            cancelled.set()
            if failed_stage:
                raise RuntimeError('Owned cancelled preparation failure')
        return value

    if stop_during in {'profile', 'graph'}:
        from jarvis.memory import graph, graph_ops
        mock_config.db_path = str(tmp_path / 'graph.db')
        store = graph.GraphMemoryStore(mock_config.db_path)
        try:
            note = store.create_node('Gardening', 'Plant preferences', data='The user grows orchids.', parent_id='user')
        finally:
            store.close()
        if stop_during == 'profile':
            monkeypatch.setattr(graph_ops, 'build_warm_profile', lambda store: stage('profile', {'user': '', 'directives': ''}))
        else:
            mock_config.memory_enrichment_source = 'graph'
            monkeypatch.setattr(graph.GraphMemoryStore, 'search_nodes', lambda *a, **kw: stage('graph', [note]))

    if stop_during == 'router_cache':
        monkeypatch.setattr(dialogue_memory, 'hot_cache_get', lambda key: stage('router_cache', None))

    if stop_during == 'discovery':
        from jarvis.tools import registry
        mock_config.mcps = {'owned': {}}
        monkeypatch.setattr(registry, 'is_mcp_cache_initialized', lambda: True)
        monkeypatch.setattr(registry, 'refresh_mcp_tools', lambda **kw: stage('discovery', ({}, {})))
        monkeypatch.setattr(registry, 'get_cached_mcp_tools', lambda: {})

    monkeypatch.setattr(engine, 'select_tools', lambda *a, **kw: stage('router', ['getWeather']))
    monkeypatch.setattr(engine, 'plan_query', lambda *a, **kw: stage('planner', ["searchMemory topic='preferences'", 'Reply to the user.']))
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *a, **kw: stage('keywords', {'keywords': ['orchids'], 'questions': ['Which flowers does the user grow?']}))
    monkeypatch.setattr(conversation, 'search_conversation_memory_by_keywords', lambda *a, **kw: stage('diary', ['The user grows orchids.']))
    monkeypatch.setattr(engine, 'digest_memory_for_query', lambda *a, **kw: stage('digest', 'The user grows orchids.'))
    monkeypatch.setattr(engine, 'chat_with_messages', lambda *a, **kw: stage('reply', {'message': {'content': 'Ready.'}}))
    result = engine.run_reply_engine(db, mock_config, None, 'Find the preferences I told you about before and summarise them.', dialogue_memory, cancel_event=cancelled)
    if stop_during is None:
        assert result == 'Ready.'
        assert dialogue_memory.has_recent_messages()
    else:
        assert result is None
        assert not dialogue_memory.has_recent_messages()
    assert not forbidden.exists(), forbidden.read_text() if forbidden.exists() else ''


def test_cancelling_a_running_task_stops_preparation_before_planning(
    monkeypatch, mock_config, db, dialogue_memory, tmp_path,
):
    from jarvis.tasks import TaskManager, TaskStatus
    started, release = Event(), Event()
    forbidden = tmp_path / 'planning-started-after-cancellation'
    mock_config.planner_enabled = True

    def route(**kwargs):
        started.set()
        assert release.wait(5), 'Owned router was not released'
        return ['getWeather']

    def plan(**kwargs):
        forbidden.write_text('A cancelled task started planning')
        return ['Reply to the user.']

    monkeypatch.setattr(engine, 'select_tools', route)
    monkeypatch.setattr(engine, 'plan_query', plan)
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *a, **kw: {})
    manager = TaskManager(db, mock_config, dialogue_memory)
    try:
        task_id = manager.submit('Check the forecast and summarise the outlook.')
        assert started.wait(3)
        assert manager.cancel(task_id)
        release.set()
        manager.wait_for_idle(timeout=3)
        task = manager.get(task_id)
        assert task.status is TaskStatus.CANCELLED
        assert task.result is None
        assert not forbidden.exists()
        assert not dialogue_memory.has_recent_messages()
    finally:
        release.set()
        manager.shutdown()
