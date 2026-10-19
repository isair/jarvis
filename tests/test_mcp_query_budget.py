"""MCP catalogue waits respect the actual reply's cancellation and deadline."""
import threading
from contextlib import closing

import pytest

from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools import registry

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def budget_config(tmp_path, mock_config):
    cfg = mock_config
    cfg.db_path = str(tmp_path / 'owned-memory.db')
    cfg.mcps = {name: {'command': 'never-started-owned-fake'} for name in ('first', 'later')}
    cfg.agentic_query_timeout_sec = 3
    cfg.agentic_preparation = 'staged'
    cfg.llm_chat_model = 'test-large'
    cfg.location_enabled = False
    cfg.memory_enrichment_source = 'none'
    cfg.memory_digest_enabled = False
    cfg.planner_enabled = False
    return cfg


@pytest.mark.parametrize('stage', ['initialized', 'refresh', 'cache', 'servers', 'publication'])
@pytest.mark.parametrize('outcome', ['cancel', 'deadline', 'success'])
def test_mcp_catalogue_wait_obeys_query_lifetime(budget_config, monkeypatch, stage, outcome):
    cfg = budget_config
    if stage == 'publication':
        cfg.mcps = {'first': cfg.mcps['first']}
    cfg.agentic_query_timeout_sec = 0.08 if outcome == 'deadline' else 3
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    cancelled = threading.Event()
    helpers = []
    result = {}
    model_actions = []
    server_calls = []
    memory = DialogueMemory()

    def catalogue(*args, **kwargs):
        helpers.append(threading.current_thread())
        entered.set()
        assert release.wait(3), 'Owned catalogue gate was not released'
        if stage == 'initialized':
            return True
        return ({}, {}) if stage == 'refresh' else {}

    monkeypatch.setattr(registry, 'is_mcp_cache_initialized', catalogue if stage == 'initialized' else lambda **kwargs: True)
    if stage in {'servers', 'publication'}:
        monkeypatch.setattr(registry, '_mcp_config_cache', cfg.mcps)
        monkeypatch.setattr(registry, '_mcp_tools_cache', {'keep': registry.ToolSpec('keep', 'Saved catalogue')})

        class OwnedClient:
            def __init__(self, config):
                pass

            def list_tools(self, server_name):
                server_calls.append(server_name)
                if server_name == 'first':
                    catalogue()
                return [{'name': server_name + '_tool'}]

        monkeypatch.setattr(registry, 'MCPClient', OwnedClient)
    else:
        monkeypatch.setattr(registry, 'refresh_mcp_tools', catalogue if stage == 'refresh' else lambda **kwargs: ({}, {}))
        monkeypatch.setattr(registry, 'get_cached_mcp_tools', catalogue if stage == 'cache' else lambda **kwargs: {})
    monkeypatch.setattr(engine, '_build_enrichment_context_hint', lambda *args: 'Owned context')
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: [])

    def select(**kwargs):
        model_actions.append('route')
        return []

    def chat(**kwargs):
        model_actions.append('reply')
        return {'message': {'role': 'assistant', 'content': 'Owned request completed.'}}

    monkeypatch.setattr(engine, 'select_tools', select)
    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db:
        def request():
            try:
                result['reply'] = engine.run_reply_engine(
                    db, cfg, None, 'Answer the current request.', memory,
                    cancel_event=cancelled, quiet=True,
                )
            except BaseException as error:
                result['error'] = error
            finally:
                done.set()

        caller = threading.Thread(target=request, name='owned-budget-reply')
        try:
            caller.start()
            assert entered.wait(1)
            if outcome == 'success':
                release.set()
            elif outcome == 'cancel':
                cancelled.set()
            assert done.wait(2 if outcome == 'success' else 0.5), 'Reply remained blocked in MCP catalogue work'
            assert 'error' not in result
            if outcome == 'success':
                assert result['reply'] == 'Owned request completed.'
                assert model_actions == ['route', 'reply']
                assert memory.get_recent_messages()
            else:
                assert not model_actions, 'Abandoned discovery continued model work'
                assert memory.get_recent_messages() == []
                assert result['reply'] is None if outcome == 'cancel' else result['reply'] != 'Owned request completed.'
        finally:
            release.set()
            caller.join(3)
            assert not caller.is_alive(), 'Owned reply caller did not drain'
            for helper in helpers:
                helper.join(3)
                assert not helper.is_alive(), 'Owned catalogue helper did not drain'
        if stage in {'servers', 'publication'}:
            assert server_calls == (list(cfg.mcps) if outcome == 'success' else [next(iter(cfg.mcps))]), 'Abandoned discovery started another server'
            if outcome != 'success':
                assert set(registry.get_cached_mcp_tools()) == {'keep'}, 'Abandoned discovery replaced the saved catalogue'


@pytest.mark.parametrize('stage', ['initialized', 'refresh', 'cache'])
@pytest.mark.parametrize('outcome', ['cancel', 'deadline'])
def test_abandoned_query_releases_its_cache_lock_waiter(budget_config, monkeypatch, stage, outcome):
    cfg = budget_config
    cfg.agentic_query_timeout_sec = 0.08 if outcome == 'deadline' else 3
    admitted = threading.Event()
    done = threading.Event()
    cancelled = threading.Event()
    helpers = []
    result = {}
    target = {'initialized': 'is_mcp_cache_initialized',
              'refresh': 'refresh_mcp_tools', 'cache': 'get_cached_mcp_tools'}[stage]
    operation = getattr(registry, target)

    if stage != 'initialized':
        monkeypatch.setattr(registry, 'is_mcp_cache_initialized', lambda **kwargs: stage == 'refresh')

    def tracked(*args, **kwargs):
        helpers.append(threading.current_thread())
        admitted.set()
        return operation(*args, **kwargs)

    monkeypatch.setattr(registry, target, tracked)
    memory = DialogueMemory()
    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db:
        def request():
            try:
                result['reply'] = engine.run_reply_engine(
                    db, cfg, None, 'Answer the current request.', memory,
                    cancel_event=cancelled, quiet=True,
                )
            except BaseException as error:
                result['error'] = error
            finally:
                done.set()

        caller = threading.Thread(target=request, name='owned-cache-wait-reply')
        registry._mcp_tools_cache_lock.acquire()
        try:
            caller.start()
            assert admitted.wait(1)
            if outcome == 'cancel':
                cancelled.set()
            assert done.wait(0.5), 'An abandoned reply remained blocked on the cache lock'
            assert 'error' not in result
            assert memory.get_recent_messages() == []
            for helper in helpers:
                helper.join(0.5)
                assert not helper.is_alive(), 'Abandoned catalogue work still waits for another operation to unlock the cache'
        finally:
            registry._mcp_tools_cache_lock.release()
            caller.join(3)
            assert not caller.is_alive()
            for helper in helpers:
                helper.join(3)
                assert not helper.is_alive()
