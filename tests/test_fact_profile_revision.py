"""A write during profile construction is visible on the next conversation turn."""
from contextlib import closing
from unittest.mock import patch

import pytest

from jarvis.memory.db import Database
from jarvis.memory.facts import FactStore, build_fact_warm_profile
from jarvis.memory.graph import GraphMemoryStore
from jarvis.reply import engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('mutation', ['correct', 'retract', 'directive'])
def test_concurrent_profile_write_is_not_hidden_by_cache(tmp_path, mock_config, dialogue_memory, mutation):
    path = str(tmp_path / 'profile.db')
    old_text = 'Use long responses' if mutation == 'directive' else 'The user lives in Bristol'
    new_text = 'Keep replies concise' if mutation == 'directive' else 'The user lives in Bath'
    with closing(FactStore(path)) as store:
        old, _ = store.add_fact(
            old_text, kind='directive' if mutation == 'directive' else 'user', owner='user',
            source_ref='owned-first', source_type='dialogue', source_role='user',
            source_channel='text', source_text=old_text, evidence=old_text,
            observed_at='2026-01-01T10:00:00+00:00',
        )
    with closing(GraphMemoryStore(path)):
        pass
    mock_config.db_path = path
    mock_config.location_enabled = False
    mock_config.embedding_model = ''
    mock_config.memory_enrichment_source = 'none'
    mock_config.agentic_preparation = 'staged'
    mock_config.mcps = {}
    written = []
    systems = []

    def profile_with_concurrent_write(*args, **kwargs):
        snapshot = build_fact_warm_profile(*args, **kwargs)
        if not written:
            with closing(FactStore(path)) as writer:
                if mutation == 'retract':
                    writer.retract_fact(old['id'], evidence='Remove that saved assertion')
                else:
                    writer.correct_fact(
                        old['id'], text=new_text, source_ref='owned-correction',
                        source_role='user', source_channel='text', source_text=new_text,
                        evidence=new_text, observed_at='2026-01-02T10:00:00+00:00',
                    )
            written.append(True)
        return snapshot

    def chat(*args, **kwargs):
        systems.append(kwargs['messages'][0]['content'])
        return {'message': {'role': 'assistant', 'content': 'Saved profile checked.'}}

    with patch('jarvis.memory.facts.build_fact_warm_profile', side_effect=profile_with_concurrent_write), \
         patch.object(engine, 'plan_query', return_value=[]), \
         patch.object(engine, 'select_tools', return_value=[]), \
         patch.object(engine, 'extract_search_params_for_memory', return_value={}), \
         patch.object(engine, 'chat_with_messages', side_effect=chat), \
         closing(Database(path, sqlite_vss_path=None)) as db:
        for query in ('Check my saved profile.', 'Check my saved profile again.'):
            assert engine.run_reply_engine(db, mock_config, None, query, dialogue_memory) == 'Saved profile checked.'
    assert old_text in systems[0]
    assert old_text not in systems[-1]
    if mutation != 'retract':
        assert new_text in systems[-1]
