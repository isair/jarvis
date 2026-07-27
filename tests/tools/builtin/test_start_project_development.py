"""Tests for the 'start development' trigger: resolving which plan note to
use (by name, sole active plan, or asking when several match), reading it,
dispatching to Antigravity via MCP, and only updating status after a
confirmed successful handoff. See project_intake.spec.md "Starting
development".
"""

import json
from unittest.mock import Mock, patch

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import project_intake as pi
from jarvis.tools.builtin.project_intake import (
    StartProjectDevelopmentTool,
    _extract_note_paths,
    _extract_read_content,
)


def _make_context(cfg, redacted_text=""):
    return ToolContext(
        db=Mock(),
        cfg=cfg,
        system_prompt="",
        original_prompt=redacted_text,
        redacted_text=redacted_text,
        max_retries=1,
        user_print=lambda *_: None,
    )


def _mock_client(responses):
    """responses: list of (isError, text) tuples returned in call order."""
    client = Mock()
    client.invoke_tool.side_effect = [
        {"isError": is_error, "text": text} for is_error, text in responses
    ]
    return client


class TestStartProjectDevelopment:
    def test_resolves_plan_by_name(self, mock_config):
        tool = StartProjectDevelopmentTool()
        note_content = "---\nstatus: active\nproject: loja-x\ntype: plan\n---\n## Status\nPlano criado, desenvolvimento ainda não iniciado."
        responses = [
            (False, "Projects/loja-x/website - 2026-07-01.md\nProjects/outro/outro.md"),  # search
            (False, note_content),  # read
            (False, "dispatched"),  # antigravity dispatch
            (False, "patched"),  # status patch
        ]
        with patch.object(pi, "MCPClient", return_value=_mock_client(responses)):
            result = tool.run({"input": "avança com o projeto loja-x"}, _make_context(mock_config))

        assert result.success is True
        assert "loja-x" in result.reply_text or "website" in result.reply_text.lower()

    def test_resolves_sole_active_plan_when_none_named(self, mock_config):
        tool = StartProjectDevelopmentTool()
        note_content = "---\nstatus: active\ntype: plan\n---\n## Status\nPlano criado."
        responses = [
            (False, "Projects/unico/unico.md"),
            (False, note_content),
            (False, "dispatched"),
            (False, "patched"),
        ]
        with patch.object(pi, "MCPClient", return_value=_mock_client(responses)):
            result = tool.run({"input": "manda isto para os agentes"}, _make_context(mock_config))

        assert result.success is True

    def test_asks_user_when_several_match(self, mock_config):
        tool = StartProjectDevelopmentTool()
        responses = [
            (False, "Projects/a/a.md\nProjects/b/b.md"),
        ]
        with patch.object(pi, "MCPClient", return_value=_mock_client(responses)):
            result = tool.run({"input": "vamos começar o desenvolvimento"}, _make_context(mock_config))

        assert result.success is True
        assert "qual" in result.reply_text.lower()

    def test_updates_status_only_after_confirmed_handoff(self, mock_config):
        tool = StartProjectDevelopmentTool()
        note_content = "---\nstatus: active\ntype: plan\n---\n## Status\nPlano criado."
        responses = [
            (False, "Projects/unico/unico.md"),
            (False, note_content),
            (True, "antigravity unreachable"),  # dispatch FAILS
        ]
        client = _mock_client(responses)
        with patch.object(pi, "MCPClient", return_value=client):
            result = tool.run({"input": "manda isto para os agentes"}, _make_context(mock_config))

        assert result.success is False
        # Only 3 calls made: search, read, dispatch — the status patch call
        # must NOT have been attempted since the handoff failed.
        assert client.invoke_tool.call_count == 3

    def test_no_plans_found_offers_fresh_intake(self, mock_config):
        tool = StartProjectDevelopmentTool()
        responses = [(False, "")]
        with patch.object(pi, "MCPClient", return_value=_mock_client(responses)):
            result = tool.run({"input": "vamos começar o desenvolvimento"}, _make_context(mock_config))

        assert result.success is True
        assert "não encontrei" in result.reply_text.lower()

    def test_uses_real_configured_server_and_tool_names(self, mock_config):
        """Verified live against the user's actual config.mcps: the Obsidian
        server is keyed "obsidian" (not "Jarvis Brain") and delegation goes
        through the "jarvis-router" server's run_antigravity tool (not a
        standalone "Antigravity" server / "run_task" tool)."""
        assert pi.OBSIDIAN_MCP_SERVER == "obsidian"
        assert pi.OBSIDIAN_WRITE_TOOL == "vault_write"
        assert pi.OBSIDIAN_SEARCH_TOOL == "search_simple"
        assert pi.OBSIDIAN_READ_TOOL == "vault_read"
        assert pi.OBSIDIAN_PATCH_TOOL == "vault_patch"
        assert pi.ANTIGRAVITY_MCP_SERVER == "jarvis-router"
        assert pi.ANTIGRAVITY_DISPATCH_TOOL == "run_antigravity"

    def test_dispatch_and_patch_use_real_tool_argument_shapes(self, mock_config):
        """run_antigravity's schema only accepts {"task": str} (no separate
        "instructions" field) and vault_patch requires targetType/target/
        operation rather than a raw full-file "content" overwrite —
        confirmed via a live list_tools call against the user's servers."""
        tool = StartProjectDevelopmentTool()
        note_json = json.dumps({"content": "---\ntype: plan\n---\n## Status\nPlano criado."})
        responses = [
            (False, "Projects/unico/unico.md"),
            (False, note_json),
            (False, "dispatched"),
            (False, "patched"),
        ]
        client = _mock_client(responses)
        with patch.object(pi, "MCPClient", return_value=client):
            result = tool.run({"input": "manda isto para os agentes"}, _make_context(mock_config))

        assert result.success is True

        dispatch_call = client.invoke_tool.call_args_list[2]
        assert dispatch_call.kwargs["server_name"] == "jarvis-router"
        assert dispatch_call.kwargs["tool_name"] == "run_antigravity"
        assert set(dispatch_call.kwargs["arguments"].keys()) == {"task"}
        assert "Plano criado" in dispatch_call.kwargs["arguments"]["task"]

        patch_call = client.invoke_tool.call_args_list[3]
        assert patch_call.kwargs["server_name"] == "obsidian"
        assert patch_call.kwargs["tool_name"] == "vault_patch"
        patch_args = patch_call.kwargs["arguments"]
        assert patch_args["targetType"] == "heading"
        assert patch_args["target"] == "Status"
        assert patch_args["operation"] == "replace"
        assert "Desenvolvimento iniciado" in patch_args["content"]


class TestExtractNotePaths:
    def test_parses_real_search_simple_json_shape(self):
        """search_simple returns a JSON array of {filename, score, matches}
        objects, not one path per line — verified via a live list_tools /
        call against the "obsidian" server."""
        payload = json.dumps([
            {"filename": "Projects/a/a.md", "score": -0.1, "matches": []},
            {"filename": "Projects/b/b.md", "score": -0.2, "matches": []},
        ])
        assert _extract_note_paths(payload) == ["Projects/a/a.md", "Projects/b/b.md"]

    def test_falls_back_to_line_parsing_for_plain_text(self):
        text = "Projects/a/a.md\nProjects/b/b.md"
        assert _extract_note_paths(text) == ["Projects/a/a.md", "Projects/b/b.md"]


class TestExtractReadContent:
    def test_unwraps_real_vault_read_json_shape(self):
        """A full-file vault_read (no targetType/target) returns a JSON
        object with a "content" key plus metadata, not raw markdown."""
        payload = json.dumps({"content": "# Title\nBody", "path": "x.md", "tags": []})
        assert _extract_read_content(payload) == "# Title\nBody"

    def test_falls_back_to_raw_text_when_not_json(self):
        assert _extract_read_content("# Title\nBody") == "# Title\nBody"
