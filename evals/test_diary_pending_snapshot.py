"""Long pending conversations retain facts outside the final input window."""
from datetime import datetime, timezone

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.memory.conversation import update_daily_conversation_summary


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('facts', [
    ('My dog is called Pip.', 'My sister is called Mira.', 'I am travelling to Kyoto.'),
    ('Köpeğimin adı Pip.', 'Kız kardeşimin adı Mira.', 'Kyoto şehrine seyahat edeceğim.'),
])
def test_diary_retains_early_middle_and_late_user_facts(eval_db, facts, streaming):
    cfg = voice_config()
    cfg.embedding_model = None
    filler = ['User: Thanks.', "Assistant: You're welcome."] * 6
    chunks = ([f'User: {facts[0]}'] + filler + [f'User: {facts[1]}']
              + filler + [f'User: {facts[2]}'])
    tokens = []
    ident = update_daily_conversation_summary(
        eval_db, chunks, cfg, timeout_sec=60, on_token=tokens.append if streaming else None,
    )
    row = eval_db.get_conversation_summary(datetime.now(timezone.utc).date().isoformat(), 'jarvis')
    assert ident and row and row['topics'], 'Incomplete inference is not a saved diary'
    assert all(fact.casefold() in row['summary'].casefold() for fact in ('Pip', 'Mira', 'Kyoto')), row['summary']
    if streaming:
        visible_lines = [line.strip() for line in ''.join(tokens).splitlines() if line.strip()]
        assert visible_lines == [f"SUMMARY: {row['summary']}", f"TOPICS: {row['topics']}"]
