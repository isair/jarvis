"""Flask cafe-agent proxy routes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_cafe_agent_health_route():
    from desktop_app.memory_viewer import app

    with patch(
        "desktop_app.cafe_agent_proxy.fetch_health",
        return_value={"status": "ok", "service": "cafe-orchestrator"},
    ):
        client = app.test_client()
        resp = client.get("/api/cafe-agent/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


@pytest.mark.unit
def test_cafe_agent_task_route():
    from desktop_app.memory_viewer import app

    fake = {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "agent": "weather_check",
        "result": {"ok": True, "summary": "fine", "data": {}},
    }

    with patch("desktop_app.cafe_agent_proxy.post_task", return_value=fake):
        client = app.test_client()
        resp = client.post(
            "/api/cafe-agent/task",
            data=json.dumps({"task": {"type": "weather_check", "days": 2}}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["result"]["ok"] is True


@pytest.mark.unit
def test_cafe_agent_schedule_task_route():
    from desktop_app.memory_viewer import app

    fake = {
        "task_id": "00000000-0000-0000-0000-000000000002",
        "agent": "schedule_plan",
        "result": {
            "ok": True,
            "summary": "Draft week",
            "data": {
                "planner": "heuristic",
                "days": [{"date": "2026-05-12", "shifts": []}],
            },
        },
    }

    with patch("desktop_app.cafe_agent_proxy.post_task", return_value=fake):
        client = app.test_client()
        resp = client.post(
            "/api/cafe-agent/task",
            data=json.dumps(
                {
                    "task": {
                        "type": "schedule_plan",
                        "week_start": "2026-05-12",
                        "persist": False,
                    }
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"]["data"]["planner"] == "heuristic"
