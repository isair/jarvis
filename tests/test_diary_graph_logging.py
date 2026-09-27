"""The diary flush reports useful fact extraction outcomes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.memory.fact_ops import FactIngestResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "result,expected",
    [
        (FactIngestResult(stored=2, skipped=1), "🧠 Facts: 2 learned, 1 skipped"),
        (FactIngestResult(stored=0, skipped=3), "🧠 Facts: 0 learned, 3 skipped"),
    ],
)
def test_fact_flush_reports_extraction(db, dialogue_memory, capsys, result, expected):
    from jarvis.memory.conversation import update_diary_from_dialogue_memory

    dialogue_memory.add_message("user", "I am learning about bats")
    with patch("jarvis.memory.conversation.generate_conversation_summary",
               return_value=("The user discussed bats", "bats")), patch(
                   "jarvis.memory.fact_ops.process_pending_fact_batches", return_value=result):
        cfg = SimpleNamespace(llm_chat_model="test", embedding_model="")
        assert update_diary_from_dialogue_memory(db, dialogue_memory, cfg, force=True)
    assert expected in capsys.readouterr().out


@pytest.mark.unit
def test_empty_extraction_is_quiet(db, dialogue_memory, capsys):
    from jarvis.memory.conversation import update_diary_from_dialogue_memory

    dialogue_memory.add_message("user", "Hello")
    with patch("jarvis.memory.conversation.generate_conversation_summary",
               return_value=("The user greeted Jarvis", "greeting")), patch(
                   "jarvis.memory.fact_ops.process_pending_fact_batches",
                   return_value=FactIngestResult()):
        cfg = SimpleNamespace(llm_chat_model="test", embedding_model="")
        assert update_diary_from_dialogue_memory(db, dialogue_memory, cfg, force=True)
    assert "🧠 Facts:" not in capsys.readouterr().out
