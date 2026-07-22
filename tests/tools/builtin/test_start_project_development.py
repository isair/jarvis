"""Tests for the 'start development' trigger: resolving which plan note to
use (by name, sole active plan, or asking when several match), reading it,
dispatching to Antigravity via MCP, and only updating status after a
confirmed successful handoff. See project_intake.spec.md "Starting
development".
"""

from unittest.mock import Mock, patch

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import project_intake as pi
from jarvis.tools.builtin.project_intake import StartProjectDevelopmentTool


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
