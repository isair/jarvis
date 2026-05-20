"""Tests for typed-message delivery to Jarvis."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_submit_text_query_empty_returns_false(tmp_path, monkeypatch):
    from jarvis import text_input

    monkeypatch.setattr(text_input, "inbox_path", lambda: tmp_path / "inbox.jsonl")
    monkeypatch.setattr(text_input, "_listener", None)

    assert text_input.submit_text_query("") is False
    assert text_input.submit_text_query("   ") is False


@pytest.mark.unit
def test_deliver_text_query_prefers_external_hook(tmp_path, monkeypatch):
    from jarvis import text_input

    monkeypatch.setattr(text_input, "_listener", None)
    monkeypatch.setattr(text_input, "_external_delivery", lambda t, imgs: True)

    assert text_input.deliver_text_query("run queue") == "stdin"


@pytest.mark.unit
def test_submit_text_query_writes_inbox_when_no_listener(tmp_path, monkeypatch):
    from jarvis import text_input

    inbox = tmp_path / "inbox.jsonl"
    monkeypatch.setattr(text_input, "inbox_path", lambda: inbox)
    monkeypatch.setattr(text_input, "_listener", None)
    monkeypatch.setattr(text_input, "_external_delivery", None)

    assert text_input.submit_text_query("Sveiki, Jarvis") is True
    lines = inbox.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "Sveiki, Jarvis"


@pytest.mark.unit
def test_inbox_drained_into_listener_queue(tmp_path, monkeypatch):
    from jarvis import text_input

    inbox = tmp_path / "inbox.jsonl"
    inbox.write_text(json.dumps({"text": "first"}) + "\n", encoding="utf-8")
    listener = MagicMock()

    monkeypatch.setattr(text_input, "inbox_path", lambda: inbox)
    count = text_input._drain_inbox_file_into_listener(listener)

    assert count == 1
    listener.enqueue_text_query.assert_called_once_with("first", image_paths=[])
    assert not inbox.exists()


@pytest.mark.unit
def test_dashboard_query_endpoint_rejects_empty():
    from desktop_app.memory_viewer import app

    with app.test_client() as c:
        res = c.post("/api/dashboard/query", json={"text": "  "})
        assert res.status_code == 400


@pytest.mark.unit
def test_dashboard_query_endpoint_queues_message(tmp_path, monkeypatch):
    from desktop_app.memory_viewer import app
    from jarvis import text_input

    inbox = tmp_path / "inbox.jsonl"
    monkeypatch.setattr(text_input, "inbox_path", lambda: inbox)
    monkeypatch.setattr(text_input, "_listener", None)

    with app.test_client() as c:
        res = c.post("/api/dashboard/query", json={"text": "What is on my calendar?"})
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("ok") is True

    assert "calendar" in inbox.read_text(encoding="utf-8")
