"""Tests for Sulainis cross-process prompt bridge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_enqueue_and_drain_prompt_queue(tmp_path, monkeypatch):
    from jarvis import sulainis_bridge as bridge

    qpath = tmp_path / "sulainis_prompt_queue.jsonl"
    monkeypatch.setattr(bridge, "_queue_path", lambda: qpath)

    assert bridge.enqueue_sulainis_prompt("Hello Johnny", action="ask")
    delivered: list[str] = []

    def deliver(text: str) -> bool:
        delivered.append(text)
        return True

    n = bridge.drain_sulainis_prompt_queue(deliver)
    assert n == 1
    assert delivered == ["Hello Johnny"]
    assert not qpath.is_file()


@pytest.mark.unit
def test_desktop_state_roundtrip(tmp_path, monkeypatch):
    from jarvis import sulainis_bridge as bridge

    spath = tmp_path / "desktop_state.json"
    monkeypatch.setattr(bridge, "_desktop_state_path", lambda: spath)

    bridge.write_desktop_state(is_listening=True)
    assert bridge.is_daemon_listening() is True
    bridge.write_desktop_state(is_listening=False)
    assert bridge.is_daemon_listening() is False
