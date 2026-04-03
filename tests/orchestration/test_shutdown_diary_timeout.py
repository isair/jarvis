"""Orchestration tests — graceful shutdown diary flush timeout (spec 5.3 / 5.7).

Tests that ShutdownManager does not hang indefinitely if the LLM diary
update takes longer than the configured timeout.
"""

from __future__ import annotations

import threading
import time
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ShutdownManager unit
# ---------------------------------------------------------------------------

def _make_shutdown_manager(timeout: float = 5.0):
    from jarvis.runtime.shutdown_manager import ShutdownManager

    class _FakeCfg:
        shutdown_diary_timeout_sec: float = timeout

    health = MagicMock()
    manager = ShutdownManager(_FakeCfg(), health)
    return manager


def _make_fake_db():
    db = MagicMock()
    db.close = MagicMock()
    return db


def _make_fake_tts():
    tts = MagicMock()
    tts.stop = MagicMock()
    return tts


# ---------------------------------------------------------------------------
# Diary flush respects timeout
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_shutdown_completes_within_timeout_when_diary_blocks(monkeypatch):
    """Shutdown must finish even if the diary update hangs indefinitely."""
    manager = _make_shutdown_manager(timeout=2.0)
    db = _make_fake_db()
    tts = _make_fake_tts()
    manager.register_db(db)
    manager.register_tts(tts)

    # Simulate a blocking diary update
    def _blocking_diary(*args, **kwargs):
        time.sleep(60.0)  # Much longer than the timeout

    monkeypatch.setattr(
        "jarvis.memory.conversation.update_diary_from_dialogue_memory",
        _blocking_diary,
        raising=False,
    )

    dialogue_memory = MagicMock()
    dialogue_memory.should_update_diary.return_value = True
    dialogue_memory.get_pending_chunks.return_value = ["chunk1"]
    manager.register_dialogue_memory(dialogue_memory)

    start = time.time()
    manager.shutdown(timeout_sec=2.0)
    elapsed = time.time() - start

    # Should finish within ~3 seconds (2s timeout + small buffer)
    assert elapsed < 5.0, f"Shutdown took too long: {elapsed:.1f}s"


@pytest.mark.unit
def test_db_closed_after_timeout(monkeypatch):
    """Database is closed even when the diary flush times out."""
    manager = _make_shutdown_manager(timeout=1.0)
    db = _make_fake_db()
    tts = _make_fake_tts()
    manager.register_db(db)
    manager.register_tts(tts)

    def _blocking_diary(*args, **kwargs):
        time.sleep(30.0)

    monkeypatch.setattr(
        "jarvis.memory.conversation.update_diary_from_dialogue_memory",
        _blocking_diary,
        raising=False,
    )

    dialogue_memory = MagicMock()
    dialogue_memory.should_update_diary.return_value = True
    dialogue_memory.get_pending_chunks.return_value = ["chunk1"]
    manager.register_dialogue_memory(dialogue_memory)

    manager.shutdown(timeout_sec=1.0)
    db.close.assert_called_once()


@pytest.mark.unit
def test_tts_stopped_after_timeout(monkeypatch):
    """TTS engine is stopped even when the diary flush times out."""
    manager = _make_shutdown_manager(timeout=1.0)
    db = _make_fake_db()
    tts = _make_fake_tts()
    manager.register_db(db)
    manager.register_tts(tts)

    def _blocking_diary(*args, **kwargs):
        time.sleep(30.0)

    monkeypatch.setattr(
        "jarvis.memory.conversation.update_diary_from_dialogue_memory",
        _blocking_diary,
        raising=False,
    )

    dialogue_memory = MagicMock()
    dialogue_memory.should_update_diary.return_value = True
    dialogue_memory.get_pending_chunks.return_value = ["chunk1"]
    manager.register_dialogue_memory(dialogue_memory)

    manager.shutdown(timeout_sec=1.0)
    tts.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Fast shutdown when no diary update is needed
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_shutdown_fast_when_no_diary_pending():
    """Shutdown completes quickly when there are no pending diary chunks."""
    manager = _make_shutdown_manager(timeout=30.0)
    db = _make_fake_db()
    tts = _make_fake_tts()
    manager.register_db(db)
    manager.register_tts(tts)

    dialogue_memory = MagicMock()
    dialogue_memory.should_update_diary.return_value = False
    dialogue_memory.get_pending_chunks.return_value = []
    manager.register_dialogue_memory(dialogue_memory)

    start = time.time()
    manager.shutdown(timeout_sec=30.0)
    elapsed = time.time() - start

    assert elapsed < 3.0, f"Expected fast shutdown, got {elapsed:.1f}s"
    db.close.assert_called_once()
