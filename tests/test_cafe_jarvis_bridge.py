"""Cafe-agent → Jarvis Sulainis bridge (email/WhatsApp)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_cafe_jarvis_bridge_email_queues_briefing():
    from desktop_app.cafe_jarvis_bridge import handle_cafe_jarvis_bridge

    with patch(
        "desktop_app.sulainis_api.queue_sulainis_action",
        return_value="stdin",
    ) as mock_q:
        out = handle_cafe_jarvis_bridge("email", "sync")
    assert out["ok"] is True
    assert out["delivery"] == "stdin"
    mock_q.assert_called_once()
    args, _ = mock_q.call_args
    assert args[0] == "briefing"


@pytest.mark.unit
def test_cafe_jarvis_bridge_route():
    from desktop_app.memory_viewer import app

    with patch(
        "desktop_app.cafe_jarvis_bridge.handle_cafe_jarvis_bridge",
        return_value={"ok": True, "summary": "queued", "delivery": "stdin"},
    ):
        client = app.test_client()
        resp = client.post(
            "/api/cafe-agent/jarvis-bridge",
            json={"channel": "whatsapp", "action": "sync"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True


@pytest.mark.unit
def test_cafe_jarvis_bridge_rejects_unknown_channel():
    from desktop_app.cafe_jarvis_bridge import handle_cafe_jarvis_bridge

    out = handle_cafe_jarvis_bridge("sms", "sync")
    assert out["ok"] is False
