"""Phase 4 · Section B — legacy Knowledge Graph auto-write gate.

Root cause (verified against the live DB): the conversation→graph write inside
``update_diary_from_dialogue_memory`` ran on every diary flush, INDEPENDENT of
``conversation_learning_enabled``. It accumulated fabricated / hallucinated
"facts" into the User & Directives branches that ``build_warm_profile`` injects
into every system prompt.

The fix is a single default-FALSE gate ``legacy_knowledge_auto_write_enabled``.
These tests lock in:
  * gate OFF (default) → the graph writer is NEVER called, yet the diary summary
    is still produced and messages are still marked saved (conversation flow and
    diary are untouched);
  * gate ON → the writer runs exactly as before;
  * the parameter defaults to False (fail-closed for every existing caller).
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock

import pytest

from jarvis.memory.graph_ops import GraphUpdateResult
from jarvis.memory.conversation import update_diary_from_dialogue_memory


def _drive(db, dialogue_memory, *, gate, graph_mock):
    dialogue_memory.add_message("user", "remember bats are not blind")
    dialogue_memory.add_message("assistant", "Yes — they also use echolocation.")
    with patch(
        "jarvis.memory.conversation.generate_conversation_summary",
        return_value=("User asked about bats.", "bats"),
    ), patch(
        "jarvis.memory.graph_ops.update_graph_from_dialogue",
        graph_mock,
    ):
        kwargs = dict(
            db=db,
            dialogue_memory=dialogue_memory,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="test",
            ollama_embed_model="test",
            force=True,
            timeout_sec=5.0,
        )
        if gate is not None:
            kwargs["legacy_knowledge_auto_write_enabled"] = gate
        return update_diary_from_dialogue_memory(**kwargs)


@pytest.mark.unit
class TestLegacyKgGate:
    def test_param_defaults_to_false(self):
        sig = inspect.signature(update_diary_from_dialogue_memory)
        p = sig.parameters["legacy_knowledge_auto_write_enabled"]
        assert p.default is False, "gate param must be fail-closed by default"

    def test_gate_off_skips_graph_writer_but_keeps_diary(self, db, dialogue_memory, capsys):
        graph_mock = MagicMock(return_value=GraphUpdateResult(stored=[("x", "world")], skipped=0))
        # gate omitted entirely → relies on the False default.
        summary_id = _drive(db, dialogue_memory, gate=None, graph_mock=graph_mock)

        assert summary_id is not None, "diary summary must still be produced with gate OFF"
        graph_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "Knowledge graph: learned" not in out

    def test_gate_off_explicit_false_also_skips(self, db, dialogue_memory):
        graph_mock = MagicMock(return_value=GraphUpdateResult(stored=[], skipped=0))
        summary_id = _drive(db, dialogue_memory, gate=False, graph_mock=graph_mock)
        assert summary_id is not None
        graph_mock.assert_not_called()

    def test_gate_off_still_marks_messages_saved(self, db, dialogue_memory):
        # mark_saved_up_to runs BEFORE the gate, so pending chunks must clear
        # even though the graph writer is skipped — otherwise the same messages
        # would be re-summarised forever.
        graph_mock = MagicMock(return_value=GraphUpdateResult(stored=[], skipped=0))
        with patch.object(
            dialogue_memory, "mark_saved_up_to", wraps=dialogue_memory.mark_saved_up_to
        ) as spy:
            summary_id = _drive(db, dialogue_memory, gate=False, graph_mock=graph_mock)
        assert summary_id is not None
        spy.assert_called()  # saved-marker still advanced with gate OFF

    def test_gate_on_invokes_graph_writer(self, db, dialogue_memory, capsys):
        graph_mock = MagicMock(
            return_value=GraphUpdateResult(stored=[("Bats echolocate.", "world")], skipped=0)
        )
        summary_id = _drive(db, dialogue_memory, gate=True, graph_mock=graph_mock)
        assert summary_id is not None
        graph_mock.assert_called_once()
        out = capsys.readouterr().out
        assert "Knowledge graph: learned 1 new fact" in out
