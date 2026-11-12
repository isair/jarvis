"""Real diary and graph inference retain facts when a flush crosses midnight."""
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from conftest import requires_judge_llm
from helpers import judge_pass_fail, voice_config
from jarvis.memory import conversation
from jarvis.memory.db import Database

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('text', 'streaming'), [
    ('I own Cafe Aurora. It opens at 06:35 every weekday.', False),
    ('Aurora adlı kafenin sahibiyim. Kafe hafta içi her gün sabah 06:35 saatinde açılıyor.', True),
])
def test_midnight_diary_facts_reach_the_persistent_graph(graph_store, monkeypatch, text, streaming):
    before = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
    clock = [before]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]
    monkeypatch.setattr(conversation, 'datetime', Clock)
    generate = conversation.generate_conversation_summary
    def crossing_midnight(*args, **kwargs):
        result = generate(*args, **kwargs)
        clock[0] = before + timedelta(seconds=2)
        return result
    monkeypatch.setattr(conversation, 'generate_conversation_summary', crossing_midnight)
    cfg = voice_config()
    cfg.embedding_model = None
    cfg.db_path = graph_store.db_path
    memory = conversation.DialogueMemory()
    memory.add_message('user', text)
    with closing(Database(cfg.db_path, sqlite_vss_path=None)) as db:
        ident = conversation.update_diary_from_dialogue_memory(
            db, memory, cfg, force=True, timeout_sec=60,
            on_token=(lambda token: None) if streaming else None,
        )
        saved = db.get_conversation_summary(before.date().isoformat(), 'jarvis')
        assert saved and ident == saved['id'] and 'aurora' in saved['summary'].casefold(), saved
        facts = '\n'.join(node.data for node in graph_store.get_all_nodes() if node.data)
        assert 'aurora' in facts.casefold(), 'Committed diary facts never reached the graph'
        assert judge_pass_fail(
            'PASS only if the stored facts identify Cafe Aurora opening at 06:35 in the morning '
            'on weekdays. Equivalent words or time expressions in any language are accepted; '
            'a different opening time or a missing fact is FAIL.', facts,
        ) == 'PASS', facts
