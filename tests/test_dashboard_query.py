"""Dashboard typed query API (shell quick message)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_dashboard_query_returns_inbox_delivery():
    from desktop_app.memory_viewer import app

    with patch(
        "jarvis.text_input.deliver_text_query",
        return_value="inbox",
    ):
        client = app.test_client()
        resp = client.post(
            "/api/dashboard/query",
            json={"text": "hello jarvis"},
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["delivery"] == "inbox"
    assert "inbox" in data["message"].lower()
