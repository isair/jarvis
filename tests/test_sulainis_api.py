"""Tests for Sulainis API helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.config import Settings


def _cfg(**kw) -> Settings:
    base = {
        "product_name": "Sulainis",
        "wake_word": "Johnny",
        "latvian_quality_enabled": True,
        "operator_name": "Jansona kungs",
    }
    base.update(kw)
    return MagicMock(**base)


@pytest.mark.unit
def test_uses_latvian_only_when_reply_language_lv():
    from desktop_app.sulainis_api import _uses_latvian

    assert _uses_latvian(_cfg()) is False
    assert _uses_latvian(_cfg(reply_language="en")) is False
    assert _uses_latvian(_cfg(reply_language="lv")) is True
    assert _uses_latvian(_cfg(latvian_quality_enabled=True)) is False


@pytest.mark.unit
def test_build_action_prompt_draft_email_latvian():
    from desktop_app.sulainis_api import build_action_prompt

    cfg = _cfg(reply_language="lv")
    p = build_action_prompt(
        "draft_email",
        {"from": "Līna", "subject": "Maiņa", "snippet": "Vai vari rīt?"},
        cfg,
    )
    assert "Johnny" in p
    assert "NEsūti" in p or "neapstiprin" in p.lower()
    assert "Līna" in p


@pytest.mark.unit
def test_build_today_stream_orders_items():
    from desktop_app.sulainis_api import build_today_stream

    stream = build_today_stream(
        {"messages": [{"from": "a", "subject": "S", "snippet": "x"}]},
        {"channels": {"whatsapp": [{"from": "b", "text": "hi", "at": "2026-05-19T12:00:00Z"}]}},
        {"events": [{"title": "Meet", "start": "2026-05-19T10:00:00Z"}]},
    )
    kinds = {row["kind"] for row in stream}
    assert "calendar" in kinds
    assert "gmail" in kinds
    assert "whatsapp" in kinds


@pytest.mark.unit
def test_build_sulainis_overview_includes_cafe_agent():
    from desktop_app.sulainis_api import build_sulainis_overview

    with patch(
        "desktop_app.cafe_agent_proxy.fetch_health",
        return_value={"status": "ok", "service": "cafe-orchestrator", "sales_rows": 3},
    ):
        overview = build_sulainis_overview(_cfg())
    assert overview["cafe_agent"]["online"] is True
    assert overview["cafe_agent"]["task_url"] == "/api/cafe-agent/task"


@pytest.mark.unit
def test_build_ticker_feed_no_raw_message_bodies():
    from desktop_app.sulainis_api import build_ticker_feed

    feed = build_ticker_feed(
        _cfg(),
        operator="Test",
        product="Sulainis",
        weather={"ok": True, "current": {"temp_c": 20, "description": "Clear"}, "location": "Riga"},
        calendar_schedule={"days": [{"is_today": True, "events": []}]},
        work_queue={"items": [{"title": "Ship report", "status": "open", "priority": "high"}]},
        unread_count=3,
        wa_thread_count=2,
    )
    texts = " ".join(x["text"] for x in feed)
    assert "Ship report" in texts
    assert "WhatsApp: 2 conversations" in texts
    assert "snippet" not in texts.lower()


@pytest.mark.unit
def test_mutate_work_queue_add(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    from desktop_app.sulainis_api import mutate_work_queue

    cfg = _cfg(work_queue_enabled=True)
    out = mutate_work_queue("add", {"title": "Test task"}, cfg)
    assert out["ok"] is True
    assert out["item"]["title"] == "Test task"


@pytest.mark.unit
def test_build_calendar_schedule_groups_events():
    from desktop_app.sulainis_api import build_calendar_schedule

    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).date()
    d0 = today.isoformat()
    d1 = (today + __import__("datetime").timedelta(days=1)).isoformat()
    sched = build_calendar_schedule(
        {
            "events": [
                {"title": "Morning", "start": f"{d0}T09:00:00Z"},
                {"title": "Afternoon", "start": f"{d0}T14:00:00Z"},
                {"title": "Tomorrow", "start": f"{d1}T10:00:00Z"},
            ],
            "horizon_days": 3,
        },
        horizon_days=3,
    )
    assert sched["total_events"] == 3
    assert len(sched["days"]) == 3
    today_row = next(d for d in sched["days"] if d["is_today"])
    assert today_row["event_count"] == 2


@pytest.mark.unit
def test_run_task_queue_prompt():
    from desktop_app.sulainis_api import build_action_prompt

    p = build_action_prompt("run_task_queue", {}, _cfg(reply_language="en"))
    assert "manageWorkQueue" in p
    assert 'operation "summary"' in p
    assert "one at a time" in p.lower()


@pytest.mark.unit
def test_add_calendar_event_prompt():
    from desktop_app.sulainis_api import build_action_prompt

    p = build_action_prompt(
        "add_calendar_event",
        {"title": "Meet", "start": "2026-05-20T10:00:00", "end": "2026-05-20T11:00:00"},
        _cfg(reply_language="en"),
    )
    assert "createEvent" in p or "create_event" in p
    assert "Meet" in p


@pytest.mark.unit
def test_group_whatsapp_threads_merges_same_chat():
    from desktop_app.sulainis_api import group_whatsapp_threads

    threads = group_whatsapp_threads(
        [
            {"from": "Grupa", "chat": "Grupa", "text": "viens", "at": "2026-05-19T10:00:00Z"},
            {"from": "Grupa", "chat": "Grupa", "text": "divi", "at": "2026-05-19T11:00:00Z"},
            {"from": "Cits", "chat": "Cits", "text": "x", "at": "2026-05-19T09:00:00Z"},
        ]
    )
    assert len(threads) == 2
    grupa = next(t for t in threads if t["chat"] == "Grupa")
    assert grupa["message_count"] == 2
    assert grupa["messages"][-1]["text"] == "divi"


@pytest.mark.unit
def test_queue_sulainis_action_calls_text_input():
    from desktop_app.sulainis_api import queue_sulainis_action

    with patch("jarvis.text_input.deliver_text_query", return_value="stdin") as deliver:
        assert queue_sulainis_action("run_task_queue", {}) == "stdin"
    assert deliver.call_count == 1


@pytest.mark.unit
def test_flask_run_task_queue_action():
    from unittest.mock import patch

    from desktop_app.memory_viewer import app

    with patch(
        "desktop_app.sulainis_api.queue_sulainis_action", return_value="stdin"
    ):
        client = app.test_client()
        r = client.post(
            "/api/sulainis/action",
            json={"action": "run_task_queue", "payload": {}},
        )
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data.get("delivery") == "stdin"
