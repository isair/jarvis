"""Completed assistant tasks do not become durable personal knowledge."""
import json

import pytest

from conftest import requires_judge_llm
from helpers import voice_config

pytestmark = [pytest.mark.eval, requires_judge_llm]

CASES = [
    ('The user logged a Big Mac meal and then asked for that entry to be deleted. The assistant confirmed deletion.', 'The user requested and received confirmation that the meal entry was deleted.', False),
    ('The user cancelled a calendar booking and received confirmation.', 'The user cancelled a calendar booking and received confirmation.', False),
    ('Kullanıcı yemek kaydını sildirdi ve silindiği onaylandı.', 'The user had a meal entry deleted successfully.', False),
    ('La usuaria pidió abrir un sitio web y recibió confirmación de que se había abierto.', 'The user opened a website successfully with the assistant.', False),
    ('Der Benutzer liess eine Erinnerung ändern und erhielt eine Bestätigung.', 'The user changed a reminder successfully.', False),
    ('The user asked to log a Big Mac meal and then asked to delete it. No statement about actually consuming it was made.', 'The user ate a Big Mac today.', False),
    ('The user said they ate a Big Mac, then clarified that this was a test and they did not eat it.', 'The user ate a Big Mac today.', False),
    ('The user said they ate a Big Mac and deleted the entry for privacy, explicitly confirming that they did eat it.', 'The user ate a Big Mac today.', True),
    ('The user deleted a meal record and said they live in London.', 'The user lives in London.', True),
    ('The user deleted a reminder and instructed the assistant to always reply briefly.', 'Always reply briefly.', True),
]


@pytest.mark.parametrize(('summary', 'fact', 'retained'), CASES)
def test_completed_transactions_preserve_only_established_knowledge(
    monkeypatch, summary, fact, retained,
):
    from jarvis.memory import graph_ops

    original = graph_ops.call_llm_direct
    branch = 'DIRECTIVES' if fact.startswith('Always') else 'USER'

    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps([{'branch': branch, 'fact': fact}])
        return original(**kwargs)

    monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
    cfg = voice_config()
    result = graph_ops.extract_graph_memories(
        summary, cfg, cfg.llm_chat_model, timeout_sec=30,
    )
    assert result == ([(branch.lower(), fact)] if retained else [])


@pytest.mark.parametrize('summary', [case[0] for case in CASES[:5]])
def test_extraction_omits_completed_transactions(summary):
    from jarvis.memory import graph_ops

    cfg = voice_config()
    assert graph_ops.extract_graph_memories(
        summary, cfg, cfg.llm_chat_model, timeout_sec=30,
    ) == []


def test_mixed_candidates_retain_personal_knowledge(monkeypatch):
    from jarvis.memory import graph_ops

    summary = (
        'The user asked to delete a meal entry and received confirmation. '
        'They said they live in London.'
    )
    candidates = [
        {'branch': 'USER', 'fact': 'The user had a meal entry deleted successfully.'},
        {'branch': 'USER', 'fact': 'The user lives in London.'},
    ]
    original = graph_ops.call_llm_direct

    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps(candidates)
        return original(**kwargs)

    monkeypatch.setattr(graph_ops, 'call_llm_direct', infer)
    cfg = voice_config()
    assert graph_ops.extract_graph_memories(
        summary, cfg, cfg.llm_chat_model, timeout_sec=30,
    ) == [('user', candidates[1]['fact'])]
