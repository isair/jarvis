"""Tests for comms_state draft helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_save_and_load_assistant_draft(tmp_path, monkeypatch):
    from jarvis import comms_state

    state_file = tmp_path / "comms_state.json"
    monkeypatch.setattr(comms_state, "_state_path", lambda: state_file)

    comms_state.save_assistant_draft("Hello draft", context={"action": "draft_email"})
    loaded = comms_state.load_assistant_draft()
    assert loaded.get("text") == "Hello draft"
    assert loaded.get("context", {}).get("action") == "draft_email"


@pytest.mark.unit
def test_clear_assistant_draft(tmp_path, monkeypatch):
    from jarvis import comms_state

    state_file = tmp_path / "comms_state.json"
    monkeypatch.setattr(comms_state, "_state_path", lambda: state_file)
    comms_state.save_assistant_draft("To clear")
    comms_state.clear_assistant_draft()
    assert comms_state.load_assistant_draft() == {}
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "assistant_draft" not in data


@pytest.mark.unit
def test_count_inbox_items_sums_channels():
    from jarvis.comms_state import count_inbox_items

    with patch("desktop_app.pulse_api.load_gmail_preview") as gmail:
        with patch("desktop_app.pulse_api.load_comms_log") as comms:
            gmail.return_value = {"messages": [{}, {}]}
            comms.return_value = {
                "channels": {
                    "whatsapp": [
                        {"from": "A", "chat": "A", "text": "1"},
                        {"from": "A", "chat": "A", "text": "2"},
                        {"from": "B", "chat": "B", "text": "3"},
                    ]
                }
            }
            assert count_inbox_items() == 4
