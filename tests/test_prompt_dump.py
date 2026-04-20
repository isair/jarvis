"""
Unit tests for the opt-in prompt dump (src/jarvis/reply/prompt_dump.py).

The dump exists because PR #232's harness evals cannot reproduce the live
Possessor→"Under the Skin" confab. We need a way to capture the *exact*
messages array the field hits so a deterministic eval can replay it.

Tests focus on behaviours rather than internals:
  * OFF by default (no file when env var unset).
  * ON when env var is set — file lands in the expected directory with the
    full messages array round-tripped.
  * Failures during dump never propagate (diagnostics must not break replies).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.reply import prompt_dump


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect Path.home() so dumps land in a sandbox."""
    monkeypatch.setattr(prompt_dump.Path, "home", lambda: tmp_path)
    return tmp_path


class TestGating:
    def test_disabled_by_default(self, tmp_home, monkeypatch):
        monkeypatch.delenv("JARVIS_DUMP_PROMPTS", raising=False)
        result = prompt_dump.dump_reply_turn(
            session_id="abc12345",
            turn=1,
            query="hi",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools_schema=None,
            use_text_tools=False,
        )
        assert result is None
        # No files created.
        assert not (tmp_home / ".local" / "share" / "jarvis" / "prompts").exists()

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_enabled_by_truthy_env_values(self, tmp_home, monkeypatch, value):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", value)
        assert prompt_dump.is_enabled()

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_disabled_by_falsy_env_values(self, tmp_home, monkeypatch, value):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", value)
        assert not prompt_dump.is_enabled()


class TestDumpContents:
    def test_writes_full_payload(self, tmp_home, monkeypatch, capsys):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Tell me about Possessor"},
        ]
        tools = [{"type": "function", "function": {"name": "webSearch"}}]
        path = prompt_dump.dump_reply_turn(
            session_id="deadbeef",
            turn=2,
            query="Tell me about Possessor",
            model="gemma4:e2b",
            messages=messages,
            tools_schema=tools,
            use_text_tools=False,
            response={"message": {"content": "Under the Skin"}},
        )
        assert path is not None
        assert path.exists()
        assert path.parent == tmp_home / ".local" / "share" / "jarvis" / "prompts"
        assert "deadbeef" in path.name
        assert "t02" in path.name

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["session_id"] == "deadbeef"
        assert payload["turn"] == 2
        assert payload["query"] == "Tell me about Possessor"
        assert payload["model"] == "gemma4:e2b"
        assert payload["messages"] == messages
        assert payload["tools_schema"] == tools
        assert payload["use_text_tools"] is False
        assert payload["response"]["message"]["content"] == "Under the Skin"

        # User-visible line should mention the path so they know where to grab it.
        out = capsys.readouterr().out
        assert str(path) in out

    def test_session_ids_are_unique_per_call(self):
        ids = {prompt_dump.new_session_id() for _ in range(50)}
        assert len(ids) == 50

    def test_dump_failure_is_swallowed(self, tmp_home, monkeypatch):
        """A broken serialiser must not propagate — prompts dump is a
        diagnostic aid, never a hard dependency of reply generation."""
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")

        class Unserialisable:
            def __repr__(self):
                raise RuntimeError("nope")

        with patch("jarvis.reply.prompt_dump.json.dumps", side_effect=RuntimeError("boom")):
            result = prompt_dump.dump_reply_turn(
                session_id="abc",
                turn=1,
                query="q",
                model="m",
                messages=[],
                tools_schema=None,
                use_text_tools=False,
            )
        assert result is None  # swallowed, not raised
