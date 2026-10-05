"""Graph review rejects invented relationships before durable fact storage."""
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('summary', 'branch', 'fact', 'supported'), [
    ('The user asked to translate "I live in Bristol".', 'USER', 'The user lives in Bristol.', False),
    ('Kullanıcı "Ankara şehrinde yaşıyorum" cümlesini çevirmemi istedi.', 'USER', 'The user lives in Ankara.', False),
    ('The user\'s brother lives in Bristol.', 'USER', 'The user lives in Bristol.', False),
    ('The user used to live in Bristol and currently lives in York.', 'USER', 'The user lives in Bristol.', False),
    ('The user requested a vegetarian dinner.', 'USER', 'The user is vegetarian.', False),
    ('The user said "I live in Bristol" and requested tomorrow\'s weather.', 'USER', 'The user lives in Bristol.', True),
    ('The user is vegetarian and requested dinner suggestions.', 'USER', 'The user is vegetarian.', True),
    ('The user told the assistant to reply briefly.', 'DIRECTIVES', 'Always reply briefly.', True),
])
def test_review_checks_candidate_relationship_against_source(monkeypatch, summary, branch, fact, supported):
    from jarvis.memory import graph_ops
    original = graph_ops.call_llm_direct
    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps([{'branch': branch, 'fact': fact}])
        return original(**kwargs)
    monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
    cfg = voice_config()
    result = graph_ops.extract_graph_memories(summary, cfg, cfg.llm_chat_model, timeout_sec=30)
    assert result == ([(branch.lower(), fact)] if supported else [])
