"""Fact review preserves the diary reference date supplied to extraction."""
from datetime import date, timedelta
from contextlib import closing
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.memory import graph_ops
from jarvis.memory.graph import GraphMemoryStore

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('source_day', ['2026-10-11', '2027-12-31', '2028-02-29'])
@pytest.mark.parametrize('language', ['en', 'tr'])
def test_review_distinguishes_dated_plan_from_invented_date(monkeypatch, tmp_path, source_day, language):
    day = date.fromisoformat(source_day)
    event_day = day + timedelta(days=1)
    wrong_day = event_day + timedelta(days=1)
    facts = [
        ('user', f'The user plans to visit a museum on {event_day.isoformat()}.'),
        ('user', f'The user plans to visit a museum on {wrong_day.isoformat()}.'),
        ('user', 'The user works in the office three days per week.'),
    ]
    summary = {
        'en': 'The user plans to visit a museum tomorrow. The user works in the office three days per week.',
        'tr': 'Kullanıcı yarın bir müzeyi ziyaret etmeyi planlıyor. Haftada üç gün ofiste çalışıyor.',
    }[language]
    infer = graph_ops.call_llm_direct

    def extract_candidates(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps([{'branch': 'USER', 'fact': fact} for _, fact in facts])
        return infer(**kwargs)

    monkeypatch.setattr(graph_ops, 'call_llm_direct', extract_candidates)
    cfg = voice_config()
    with closing(GraphMemoryStore(str(tmp_path / 'source-date.db'))) as store:
        result = graph_ops.update_graph_from_dialogue(
            store, summary, cfg, cfg.llm_chat_model, timeout_sec=60, date_utc=source_day,
        )
        expected = [facts[0][1], facts[2][1]]
        assert [fact for fact, _ in result.stored] == expected
        assert store.get_node('user').data.splitlines() == expected
