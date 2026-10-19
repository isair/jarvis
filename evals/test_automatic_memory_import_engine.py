"""Automatic source changes remain usable through the real reply backend."""
from contextlib import closing
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.memory import auto_import
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.imports import ImportChoice, MemorySource, commit_import, read_source
from jarvis.reply import engine

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('scenario', ['city', 'protected', 'directive'])
def test_source_refresh_reaches_the_shared_conversation(tmp_path, scenario):
    source = tmp_path/'source.md'
    db_path = str(tmp_path/'graph.db')
    directive = scenario == 'directive'
    source.write_text('Start every reply with Ahoy.' if directive else 'I live in London.')
    preview = read_source(MemorySource('Claude', 'Owned memory', 'personal', source))
    destination = 'directives' if directive else 'user'
    choices = [ImportChoice(item.id, item.text, destination, None) for item in preview.items]
    with closing(GraphMemoryStore(db_path)) as store:
        result = commit_import(store, preview, choices)
        consent = auto_import.record_success(store, preview, choices, result)
        consent = auto_import.decide(store, consent.source_id, True)

    cfg = voice_config()
    cfg.db_path = db_path
    cfg.embedding_model = ''
    cfg.memory_enrichment_source = 'none'
    memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    # Isolate the memory-delivery path from optional planning and discovery.
    # Chat uses the actual provider backend, including its context settings.
    with patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'select_tools', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={}), \
            closing(Database(db_path, sqlite_vss_path=None)) as db:
        before = engine.run_reply_engine(
            db, cfg, None, 'What is two plus two?' if directive else 'What city do I live in?', memory,
        )
        with closing(GraphMemoryStore(db_path)) as store:
            if scenario == 'protected':
                store.update_node(result.node_ids[0], data='I live in Oxford.')
            source.write_text('Start every reply with Hello.' if directive else 'I live in Bath.')
            auto_import.sync_source(
                store, consent.source_id, consent.generation, read_source(preview.source),
            )
        after = engine.run_reply_engine(
            db, cfg, None,
            'What is three plus three?' if directive else 'What is my current home city?', memory,
        )

    assert before and after, (before, after)
    if directive:
        assert before.lower().startswith('ahoy'), before
        assert after.lower().startswith('hello'), after
    else:
        assert 'london' in before.lower(), before
        expected = 'oxford' if scenario == 'protected' else 'bath'
        assert expected in after.lower() and 'london' not in after.lower(), after
