"""
🧠 Diary → graph console-logging regression tests.

After #282 added duplicate-skip on the cumulative-summary re-flush path,
the `🧠 Knowledge graph: learned N new facts` line in
``update_diary_from_dialogue_memory`` went silent on every flush past the
first — every re-extraction routed to a node that already contained the
fact, ``stored`` came back empty, and the print was gated on a non-empty
list. From the user's perspective the memory pipeline looked dead.

These tests lock in the three CLI states (some new, all duplicates,
mixed) so the regression can't slip back in unnoticed.
"""

import time
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestKnowledgeGraphConsoleLogging:
    """Behavioural tests for the 🧠 console line emitted after a diary flush."""

    def _run_flush(self, db, dialogue_memory, graph_result):
        """Drive ``update_diary_from_dialogue_memory`` with a stubbed
        summariser and graph updater, returning whatever it printed.

        ``graph_result`` is the ``GraphUpdateResult`` the patched
        ``update_graph_from_dialogue`` should return.
        """
        from jarvis.memory.conversation import update_diary_from_dialogue_memory

        dialogue_memory.add_message("user", "I learned that bats are not blind")
        dialogue_memory.add_message("assistant", "Correct, they use echolocation in addition to sight.")

        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=("User asked about bats. Bats are not blind.", "bats, biology"),
        ), patch(
            "jarvis.memory.graph_ops.update_graph_from_dialogue",
            return_value=graph_result,
        ):
            return update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=dialogue_memory,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test",
                ollama_embed_model="test",
                force=True,
                timeout_sec=5.0,
            )

    def test_logs_count_when_new_facts_stored(self, db, dialogue_memory, capsys):
        """Mixed flush: 2 new + 1 duplicate prints the count and per-fact preview."""
        from jarvis.memory.graph_ops import GraphUpdateResult

        result = GraphUpdateResult(
            stored=[
                ("Bats use echolocation.", "world"),
                ("User is curious about bats.", "user"),
            ],
            skipped=1,
        )
        summary_id = self._run_flush(db, dialogue_memory, result)
        assert summary_id is not None

        out = capsys.readouterr().out
        assert "🧠 Knowledge graph: learned 2 new facts" in out
        assert "(1 duplicate skipped)" in out
        assert "Bats use echolocation. → world" in out
        assert "User is curious about bats. → user" in out

    def test_logs_singular_when_one_new_fact(self, db, dialogue_memory, capsys):
        """Pluralisation: a single new fact uses singular wording."""
        from jarvis.memory.graph_ops import GraphUpdateResult

        result = GraphUpdateResult(
            stored=[("Bats use echolocation.", "world")],
            skipped=0,
        )
        self._run_flush(db, dialogue_memory, result)

        out = capsys.readouterr().out
        assert "🧠 Knowledge graph: learned 1 new fact" in out
        # No trailing 's' on 'fact' and no "duplicates skipped" tail.
        assert "1 new facts" not in out
        assert "duplicate" not in out

    def test_logs_duplicates_when_only_skipped(self, db, dialogue_memory, capsys):
        """All-duplicate flush still prints a status line.

        This is the regression #282 introduced: extraction ran, the LLM
        produced facts, but every one was a duplicate so ``stored`` was
        empty and the previous gate suppressed the print entirely. The
        user lost their only signal that the memory pipeline was alive.
        """
        from jarvis.memory.graph_ops import GraphUpdateResult

        result = GraphUpdateResult(stored=[], skipped=3)
        self._run_flush(db, dialogue_memory, result)

        out = capsys.readouterr().out
        assert "🧠 Knowledge graph: nothing new" in out
        assert "(3 duplicates skipped)" in out

    def test_logs_singular_duplicate(self, db, dialogue_memory, capsys):
        """Pluralisation: a single duplicate uses singular wording."""
        from jarvis.memory.graph_ops import GraphUpdateResult

        result = GraphUpdateResult(stored=[], skipped=1)
        self._run_flush(db, dialogue_memory, result)

        out = capsys.readouterr().out
        assert "(1 duplicate skipped)" in out
        assert "1 duplicates" not in out

    def test_silent_when_extractor_returned_nothing(self, db, dialogue_memory, capsys):
        """Empty extraction (no facts and no duplicates) stays quiet.

        Distinct from the all-duplicate case: there's genuinely nothing
        to report, so we don't add console noise on every diary flush
        that didn't yield knowledge.
        """
        from jarvis.memory.graph_ops import GraphUpdateResult

        result = GraphUpdateResult(stored=[], skipped=0)
        self._run_flush(db, dialogue_memory, result)

        out = capsys.readouterr().out
        assert "🧠 Knowledge graph" not in out
