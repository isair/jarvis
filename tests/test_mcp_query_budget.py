"""MCP catalogue waits respect the actual reply's cancellation and deadline."""
import threading
from contextlib import closing

import pytest

from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools import registry

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('stage', ['refresh', 'cache'])
@pytest.mark.parametrize('outcome', ['cancel', 'deadline', 'success'])
def test_mcp_catalogue_wait_obeys_query_lifetime(tmp_path, mock_config, monkeypatch, stage, outcome):
    cfg = mock_config
    cfg.db_path = str(tmp_path / 'owned-memory.db')
    cfg.mcps = {'owned-public-probe': {'command': 'never-started-owned-fake'}}
    cfg.agentic_query_timeout_sec = 0.08 if outcome == 'deadline' else 3
    cfg.agentic_preparation = 'staged'
    cfg.llm_chat_model = 'test-large'
    cfg.location_enabled = False
    cfg.memory_enrichment_source = 'none'
    cfg.memory_digest_enabled = False
    cfg.planner_enabled = False
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    cancelled = threading.Event()
    helpers = []
    result = {}
    model_actions = []
    memory = DialogueMemory()

    def catalogue(*args, **kwargs):
        helpers.append(threading.current_thread())
        entered.set()
        assert release.wait(3), 'Owned catalogue gate was not released'
        return ({}, {}) if stage == 'refresh' else {}

    monkeypatch.setattr(registry, 'is_mcp_cache_initialized', lambda: True)
    monkeypatch.setattr(registry, 'refresh_mcp_tools', catalogue if stage == 'refresh' else lambda **kwargs: ({}, {}))
    monkeypatch.setattr(registry, 'get_cached_mcp_tools', catalogue if stage == 'cache' else lambda: {})
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
            assert done.wait(0.5), 'Reply remained blocked in MCP catalogue work'
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
