"""Tests for MCP integration status."""

from __future__ import annotations

import json

import pytest

from jarvis.operator.mcp_status import (
    build_mcp_integrations_status,
    load_mcp_status,
    save_mcp_status,
    _status_path,
)


@pytest.mark.unit
class TestMcpStatus:
    def test_build_status_ready_and_error(self):
        mcps = {
            "whatsapp": {"command": "uvx", "args": ["whatsapp-mcp-server"]},
            "broken": {"command": "npx"},
        }
        tools = {
            "whatsapp__send_message": object(),
            "whatsapp__list_chats": object(),
        }
        errors = {"broken": "connection refused"}
        status = build_mcp_integrations_status(mcps, tools, errors)
        assert status["ready_count"] == 1
        assert status["servers"]["whatsapp"]["state"] == "ready"
        assert status["servers"]["whatsapp"]["tool_count"] == 2
        assert status["servers"]["whatsapp"]["comms"] is True
        assert status["servers"]["broken"]["state"] == "error"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        path = tmp_path / "mcp_status.json"
        monkeypatch.setattr("jarvis.operator.mcp_status._status_path", lambda: path)
        payload = {"servers": {"x": {"state": "ready"}}}
        save_mcp_status(payload)
        assert load_mcp_status()["servers"]["x"]["state"] == "ready"
