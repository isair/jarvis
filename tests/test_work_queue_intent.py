"""Tests for work-queue routing and preflight."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_mentions_work_queue_english():
    from jarvis.work_queue_intent import mentions_work_queue

    assert mentions_work_queue("work through my task queue one at a time")
    assert mentions_work_queue("do all tasks automatically")
    assert mentions_work_queue('manageWorkQueue with operation "summary"')


@pytest.mark.unit
def test_mentions_work_queue_latvian():
    from jarvis.work_queue_intent import mentions_work_queue

    assert mentions_work_queue("izpildi uzdevumu rindu pa vienam")


@pytest.mark.unit
def test_boost_adds_manage_work_queue():
    from jarvis.work_queue_intent import boost_work_queue_tool_names

    out = boost_work_queue_tool_names(
        ["stop"],
        "do all tasks automatically",
        work_queue_enabled=True,
        full_catalog_names={"manageWorkQueue", "stop"},
    )
    assert "manageWorkQueue" in out


@pytest.mark.unit
def test_preflight_summary_when_enabled():
    from jarvis.work_queue_intent import try_resolve_work_queue_preflight

    cfg = MagicMock(work_queue_enabled=True)
    got = try_resolve_work_queue_preflight(
        "work through my task queue",
        ["manageWorkQueue", "stop"],
        cfg,
    )
    assert got == ("manageWorkQueue", {"operation": "summary"})


@pytest.mark.unit
def test_build_work_queue_action_plan_with_items(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    item = operator.work_queue.create_item(title="Reply to Anna", item_type="task")
    from jarvis.work_queue_intent import build_work_queue_action_plan

    plan = build_work_queue_action_plan(MagicMock(work_queue_enabled=True))
    assert any("in_progress" in s for s in plan)
    assert any(item["id"] in s for s in plan)
    assert any("done" in s for s in plan)
    assert plan[-1].startswith("Brief")


@pytest.mark.unit
def test_integration_tools_for_calendar_item(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    operator.work_queue.create_item(
        title="Add event today at 22:00",
        item_type="task",
    )
    from jarvis.work_queue_intent import integration_tools_for_work_queue

    catalog = {
        "manageWorkQueue",
        "google_workspace__searchGmail",
        "google_workspace__createCalendarEvent",
        "whatsapp__send_message",
        "stop",
    }
    got = integration_tools_for_work_queue(catalog)
    assert "google_workspace__createCalendarEvent" in got
    assert "google_workspace__searchGmail" not in got
    assert "whatsapp__send_message" not in got


@pytest.mark.unit
def test_integration_tools_for_email_type(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    operator.work_queue.create_item(
        title="Reply to supplier",
        item_type="comms_email_reply",
    )
    from jarvis.work_queue_intent import integration_tools_for_work_queue

    catalog = {
        "google_workspace__searchGmail",
        "google_workspace__createCalendarEvent",
    }
    got = integration_tools_for_work_queue(catalog)
    assert "google_workspace__searchGmail" in got
    assert "google_workspace__createCalendarEvent" not in got


@pytest.mark.unit
def test_work_queue_tool_steps_remaining():
    from jarvis.work_queue_intent import work_queue_tool_steps_remaining

    plan = [
        "manageWorkQueue operation='update' status='in_progress'",
        "Do the work",
        "manageWorkQueue operation='update' status='done'",
        "Brief the operator",
    ]
    baseline = 1
    messages = [{"role": "user", "tool_name": "manageWorkQueue"}] * 2
    assert work_queue_tool_steps_remaining(messages, plan, baseline) is True
    messages = [{"role": "user", "tool_name": "manageWorkQueue"}] * 4
    assert work_queue_tool_steps_remaining(messages, plan, baseline) is False


@pytest.mark.unit
def test_preflight_skipped_when_disabled():
    from jarvis.work_queue_intent import try_resolve_work_queue_preflight

    cfg = MagicMock(work_queue_enabled=False)
    assert (
        try_resolve_work_queue_preflight(
            "task queue", ["manageWorkQueue"], cfg
        )
        is None
    )
