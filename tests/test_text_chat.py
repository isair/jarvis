"""Behaviour tests for the text chat session (``python -m jarvis.chat``).

The text chat is the keyboard-driven front door to the same reply engine
the voice listener drives. These tests assert what a user observes when
they type at it: what reaches the assistant, what ends the session, and
that the conversation survives to the diary on the way out.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import MockConfig

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


@pytest.fixture
def chat_cfg(tmp_path):
    """Config pointing at a throwaway on-disk database."""
    return MockConfig(db_path=str(tmp_path / "chat.db"), mcps={})


@pytest.fixture
def recorded_queries(monkeypatch):
    """Stand in for the reply engine and record what it was asked.

    The stub mirrors the real engine's contract: it appends the turn to
    dialogue memory and returns the reply text.
    """
    seen: list[dict] = []

    def _fake_engine(db, cfg, tts, text, dialogue_memory, language=None):
        seen.append({
            "text": text,
            "recent": list(dialogue_memory.get_recent_messages()) if dialogue_memory else [],
        })
        reply = f"answer to {text}"
        if dialogue_memory is not None:
            dialogue_memory.add_message("user", text)
            dialogue_memory.add_message("assistant", reply)
        return reply

    monkeypatch.setattr("jarvis.chat.cli.run_reply_engine", _fake_engine)
    return seen


@pytest.fixture
def captured_diary(monkeypatch):
    """Make the diary summariser deterministic and offline.

    Writes the pending chunks straight into the summaries table so tests
    can assert on the persisted diary row instead of on call counts.
    """
    def _fake_summariser(db, new_chunks, cfg, **kwargs):
        if not new_chunks:
            return None
        today = datetime.now(timezone.utc).date().isoformat()
        return db.upsert_conversation_summary(
            date_utc=today,
            summary="\n".join(new_chunks),
            source_app=kwargs.get("source_app", "jarvis"),
        )

    monkeypatch.setattr(
        "jarvis.memory.conversation.update_daily_conversation_summary",
        _fake_summariser,
    )

    # A successful summary is followed by knowledge-graph extraction,
    # which calls the configured LLM endpoint for real. On a dev machine
    # running Ollama that turns every test here into a live call with a
    # 30s budget — not what `unit` promises.
    monkeypatch.setattr(
        "jarvis.memory.graph_ops.update_graph_from_dialogue",
        lambda **kwargs: SimpleNamespace(stored=[], skipped=0),
    )


def _session(cfg, typed: str, **kwargs) -> int:
    from jarvis.chat.cli import run_chat_session
    return run_chat_session(cfg, stdin=io.StringIO(typed), **kwargs)


@pytest.mark.unit
def test_typed_line_reaches_the_assistant(chat_cfg, recorded_queries, captured_diary):
    _session(chat_cfg, "what is the weather\n/exit\n")

    assert [q["text"] for q in recorded_queries] == ["what is the weather"]


@pytest.mark.unit
def test_successive_turns_keep_conversation_context(chat_cfg, recorded_queries, captured_diary):
    _session(chat_cfg, "who am I\nand where do I live\n/exit\n")

    assert [q["text"] for q in recorded_queries] == ["who am I", "and where do I live"]
    # The second turn sees the first turn's exchange as prior context.
    assert recorded_queries[0]["recent"] == []
    assert [m["content"] for m in recorded_queries[1]["recent"]] == [
        "who am I",
        "answer to who am I",
    ]


@pytest.mark.unit
def test_blank_lines_are_not_sent_to_the_assistant(chat_cfg, recorded_queries, captured_diary):
    _session(chat_cfg, "\n   \nhello\n\n/exit\n")

    assert [q["text"] for q in recorded_queries] == ["hello"]


@pytest.mark.unit
def test_exit_command_stops_reading_input(chat_cfg, recorded_queries, captured_diary):
    code = _session(chat_cfg, "first\n/exit\nsecond\n")

    assert code == 0
    assert [q["text"] for q in recorded_queries] == ["first"]


@pytest.mark.unit
def test_quit_is_accepted_as_well(chat_cfg, recorded_queries, captured_diary):
    code = _session(chat_cfg, "first\n/quit\nsecond\n")

    assert code == 0
    assert [q["text"] for q in recorded_queries] == ["first"]


@pytest.mark.unit
def test_end_of_input_ends_the_session(chat_cfg, recorded_queries, captured_diary):
    code = _session(chat_cfg, "only line\n")

    assert code == 0
    assert [q["text"] for q in recorded_queries] == ["only line"]


@pytest.mark.unit
def test_assistant_failure_does_not_end_the_session(chat_cfg, monkeypatch, captured_diary, capsys):
    seen: list[str] = []

    def _flaky_engine(db, cfg, tts, text, dialogue_memory, language=None):
        seen.append(text)
        if len(seen) == 1:
            raise RuntimeError("model exploded")
        return "recovered"

    monkeypatch.setattr("jarvis.chat.cli.run_reply_engine", _flaky_engine)

    code = _session(chat_cfg, "first\nsecond\n/exit\n")

    assert code == 0
    assert seen == ["first", "second"]
    assert "model exploded" in capsys.readouterr().out


@pytest.mark.unit
def test_conversation_is_written_to_the_diary_on_exit(chat_cfg, recorded_queries, captured_diary):
    _session(chat_cfg, "remember I like porridge\n/exit\n")

    from jarvis.memory.db import Database
    db = Database(chat_cfg.db_path, None)
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        row = db.get_conversation_summary(today, "stdin")
        assert row is not None, "quitting should flush the conversation to the diary"
        assert "remember I like porridge" in row["summary"]
    finally:
        db.close()


@pytest.mark.unit
def test_reset_starts_a_fresh_conversation(chat_cfg, recorded_queries, captured_diary):
    _session(chat_cfg, "first topic\n/reset\nsecond topic\n/exit\n")

    assert [q["text"] for q in recorded_queries] == ["first topic", "second topic"]
    # After a reset the next turn carries no prior dialogue.
    assert recorded_queries[1]["recent"] == []


@pytest.mark.unit
def test_help_lists_commands_without_asking_the_assistant(chat_cfg, recorded_queries, captured_diary, capsys):
    _session(chat_cfg, "/help\n/exit\n")

    out = capsys.readouterr().out
    assert recorded_queries == []
    for command in ("/help", "/reset", "/exit"):
        assert command in out


@pytest.mark.unit
def test_one_shot_query_answers_and_exits(chat_cfg, recorded_queries, captured_diary):
    from jarvis.chat.cli import run_chat_session

    code = run_chat_session(chat_cfg, stdin=io.StringIO(""), one_shot="just this")

    assert code == 0
    assert [q["text"] for q in recorded_queries] == ["just this"]


@pytest.mark.unit
def test_one_shot_puts_only_the_answer_on_stdout(chat_cfg, recorded_queries, captured_diary, capsys):
    """README advertises this form for scripts, so `$(...)` must capture the answer.

    Startup notices, the engine's planning narration and the shutdown
    lines all belong on stderr; anything of theirs on stdout ends up
    inside the caller's variable.
    """
    from jarvis.chat.cli import run_chat_session

    run_chat_session(chat_cfg, stdin=io.StringIO(""), one_shot="what is it")

    captured = capsys.readouterr()
    assert captured.out.strip() == "answer to what is it"
    for noise in ("MCP", "Saving conversation", "Goodbye", "Jarvis text chat"):
        assert noise not in captured.out


@pytest.mark.unit
def test_missing_stdin_ends_the_session_cleanly(chat_cfg, recorded_queries, captured_diary, monkeypatch):
    """A detached launch (pythonw, closed fd 0) leaves sys.stdin as None."""
    from jarvis.chat import cli

    monkeypatch.setattr(cli.sys, "stdin", None)

    assert cli.run_chat_session(chat_cfg) == 0
    assert recorded_queries == []


@pytest.mark.unit
def test_session_attributes_memory_to_stdin_even_if_the_flag_is_unset(
    chat_cfg, recorded_queries, captured_diary,
):
    """The diary and logMeal both derive their label from cfg.use_stdin.

    run_chat_session is a public entry point, so it must not depend on
    main() having set the flag — otherwise one conversation is split
    across two attributions.
    """
    import dataclasses

    from jarvis.chat.cli import run_chat_session
    from jarvis.memory.db import Database

    cfg = dataclasses.replace(chat_cfg, use_stdin=False)
    run_chat_session(cfg, stdin=io.StringIO("remember this\n/exit\n"))

    db = Database(cfg.db_path, None)
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        assert db.get_conversation_summary(today, "stdin") is not None
    finally:
        db.close()


@pytest.mark.unit
def test_runs_on_a_machine_without_audio_or_gui_libraries():
    """The text chat is the headless path: importing it must not drag in
    the microphone, hotkey, or Qt stacks."""
    blocked = ["sounddevice", "faster_whisper", "webrtcvad", "pynput", "PyQt6", "pygame"]
    program = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in %r:\n"
        "            raise ImportError(name + ' is not installed')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import jarvis.chat.cli\n"
        "print('imported')\n" % (blocked,)
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
