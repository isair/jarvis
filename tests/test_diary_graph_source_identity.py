"""Graph extraction uses the identity and date of the saved diary entry."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.memory import conversation, graph_ops
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('transition', ['same-day', 'midnight-empty', 'midnight-existing'])
@pytest.mark.parametrize('streaming', [False, True])
def test_graph_uses_saved_diary_row_across_midnight(tmp_path, mock_config, dialogue_memory, monkeypatch, transition, streaming):
    before = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
    after = before if transition == 'same-day' else before + timedelta(seconds=2)
    clock = [before]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]
    monkeypatch.setattr(conversation, 'datetime', Clock)
    mock_config.embedding_model = None
    summary = 'The user owns Cafe Aurora, which opens at 06:35 on weekdays.'
    observed = []
    with closing(Database(str(tmp_path / 'synthetic.db'), sqlite_vss_path=None)) as db:
        if transition == 'midnight-existing':
            db.upsert_conversation_summary(after.date().isoformat(), 'An unrelated next-day entry.', 'other', 'jarvis')
        def generate(*args, **kwargs):
            clock[0] = after
            return summary, 'cafe, opening'
        monkeypatch.setattr(conversation, 'generate_conversation_summary', generate)
        def update_graph(**kwargs):
            observed.append((kwargs['summary'], kwargs['date_utc']))
            return SimpleNamespace(stored=[], skipped=0)
        monkeypatch.setattr(graph_ops, 'update_graph_from_dialogue', update_graph)
        dialogue_memory.add_message('user', summary)
        ident = conversation.update_diary_from_dialogue_memory(
            db, dialogue_memory, mock_config, force=True,
            on_token=(lambda token: None) if streaming else None,
        )
        saved = db.get_conversation_summary(before.date().isoformat(), 'jarvis')
        assert ident == saved['id'] and saved['summary'] == summary
        assert dialogue_memory.get_pending_chunks() == []
        assert observed == [(summary, saved['date_utc'])]
        if transition == 'midnight-existing':
            assert db.get_conversation_summary(after.date().isoformat(), 'jarvis')['summary'] == 'An unrelated next-day entry.'
