"""Long pending conversations retain facts outside the final input window."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.memory.conversation import DialogueMemory, update_diary_from_dialogue_memory


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('facts', [
    ('My dog is called Pip.', 'My sister is called Mira.', 'I am travelling to Kyoto.'),
    ('Köpeğimin adı Pip.', 'Kız kardeşimin adı Mira.', 'Kyoto şehrine seyahat edeceğim.'),
])
def test_diary_retains_early_middle_and_late_user_facts(eval_db, facts, streaming, monkeypatch):
    cfg = voice_config()
    cfg.embedding_model = None
    filler = ['User: Thanks.', "Assistant: You're welcome."] * 6
    chunks = ([f'User: {facts[0]}'] + filler + [f'User: {facts[1]}']
              + filler + [f'User: {facts[2]}'])
    memory = DialogueMemory()
    for chunk in chunks:
        role, _, text = chunk.partition(': ')
        memory.add_message(role.lower(), text)
    monkeypatch.setattr('jarvis.memory.graph_ops.update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=0, skipped=0))
    tokens = []
    ident = update_diary_from_dialogue_memory(
        eval_db, memory, cfg, force=True, timeout_sec=60, on_token=tokens.append if streaming else None,
    )
    row = eval_db.get_conversation_summary(datetime.now(timezone.utc).date().isoformat(), 'jarvis')
    assert ident and row and row['topics'], 'Incomplete inference is not a saved diary'
    assert all(fact.casefold() in row['summary'].casefold() for fact in ('Pip', 'Mira', 'Kyoto')), row['summary']
    assert memory.get_pending_chunks() == []
    if streaming:
        visible_lines = [line.strip() for line in ''.join(tokens).splitlines() if line.strip()]
        assert visible_lines == [f"SUMMARY: {row['summary']}", f"TOPICS: {row['topics']}"]
