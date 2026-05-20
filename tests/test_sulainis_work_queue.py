"""Integration tests for Sulainis task queue API and panel."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_work_queue_panel_lists_active_only(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)

    operator.work_queue.create_item(title="Open task")
    operator.work_queue.create_item(title="In progress", description="")
    done = operator.work_queue.create_item(title="Done task")
    operator.work_queue.update_item(done["id"], patch={"status": "done"})

    from desktop_app.sulainis_api import load_work_queue_panel

    panel = load_work_queue_panel(MagicMock(work_queue_enabled=True))
    assert panel["enabled"] is True
    titles = {i["title"] for i in panel["items"]}
    assert "Open task" in titles
    assert "Done task" not in titles


@pytest.mark.unit
def test_work_queue_add_update_roundtrip(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)

    from desktop_app.sulainis_api import load_work_queue_panel, mutate_work_queue

    cfg = MagicMock(work_queue_enabled=True)
    created = mutate_work_queue("add", {"title": "Sulainis test task", "priority": "high"}, cfg)
    assert created["ok"] is True
    item_id = created["item"]["id"]

    panel = load_work_queue_panel(cfg)
    assert any(i["id"] == item_id for i in panel["items"])

    done = mutate_work_queue("update", {"item_id": item_id, "status": "done"}, cfg)
    assert done["ok"] is True

    panel2 = load_work_queue_panel(cfg)
    assert not any(i["id"] == item_id for i in panel2["items"])


@pytest.mark.unit
def test_work_queue_disabled():
    from desktop_app.sulainis_api import load_work_queue_panel, mutate_work_queue

    cfg = MagicMock(work_queue_enabled=False)
    panel = load_work_queue_panel(cfg)
    assert panel["enabled"] is False
    out = mutate_work_queue("add", {"title": "x"}, cfg)
    assert out["ok"] is False


@pytest.mark.unit
def test_flask_work_queue_route(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)

    from desktop_app.memory_viewer import app

    client = app.test_client()
    r = client.post(
        "/api/sulainis/work-queue",
        json={"operation": "add", "title": "From Flask"},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True

    r2 = client.get("/api/sulainis/work-queue")
    assert r2.status_code == 200
    panel = r2.get_json()
    assert panel["ok"] is True
    assert any(i.get("title") == "From Flask" for i in panel.get("items", []))
