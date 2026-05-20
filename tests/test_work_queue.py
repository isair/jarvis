"""Tests for operator work queue storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.operator import work_queue as wq


@pytest.mark.unit
class TestWorkQueue:
    def test_create_and_list(self, tmp_path, monkeypatch):
        path = tmp_path / "work_queue.json"
        monkeypatch.setattr(wq, "_queue_path", lambda: path)

        item = wq.create_item(title="Reply to email", item_type="comms_email_reply")
        assert item["id"].startswith("wi-")
        assert item["status"] == "open"

        items = wq.list_items(status="open")
        assert len(items) == 1
        assert items[0]["title"] == "Reply to email"

    def test_update_status(self, tmp_path, monkeypatch):
        path = tmp_path / "work_queue.json"
        monkeypatch.setattr(wq, "_queue_path", lambda: path)

        item = wq.create_item(title="Task one")
        updated = wq.update_item(item["id"], patch={"status": "done"})
        assert updated is not None
        assert updated["status"] == "done"
        assert wq.work_summary()["total_active"] == 0

    def test_summary_preview(self, tmp_path, monkeypatch):
        path = tmp_path / "work_queue.json"
        monkeypatch.setattr(wq, "_queue_path", lambda: path)

        wq.create_item(title="A", priority="high")
        wq.create_item(title="B")
        summary = wq.work_summary()
        assert summary["total_active"] == 2
        assert len(summary["preview"]) == 2
