"""Unit tests for the undo_registry module."""

import time
import pytest

from jarvis.undo_registry import (
    UndoEntry,
    UndoRegistry,
    get_undo_registry,
    push_undo,
    pop_last_undo,
    pop_undo_by_id,
    MAX_ENTRIES,
    DEFAULT_EXPIRY_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    step_id: str = "abc123",
    description: str = "wrote notes.txt",
    expires_in: float = DEFAULT_EXPIRY_SECONDS,
) -> UndoEntry:
    """Create a minimal valid UndoEntry."""
    now = time.time()
    return UndoEntry(
        step_id=step_id,
        description=description,
        tool_name="localFiles",
        tool_args={"operation": "write", "path": "notes.txt"},
        undo_tool="localFiles",
        undo_args={"operation": "write", "path": "notes.txt", "content": "original"},
        snapshot="original",
        created_at=now,
        expires_at=now + expires_in,
    )


@pytest.fixture(autouse=True)
def isolated_registry():
    """Each test gets a fresh UndoRegistry to avoid state bleed."""
    from jarvis import undo_registry as _mod
    old = _mod._registry
    _mod._registry = UndoRegistry()
    yield _mod._registry
    _mod._registry = old


# ---------------------------------------------------------------------------
# UndoEntry tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_entry_not_expired_when_fresh():
    entry = _make_entry(expires_in=60.0)
    assert entry.is_expired is False


@pytest.mark.unit
def test_entry_expired_after_deadline():
    entry = _make_entry(expires_in=-1.0)  # deadline already in the past
    assert entry.is_expired is True


@pytest.mark.unit
def test_entry_age_seconds_is_non_negative():
    entry = _make_entry()
    assert entry.age_seconds >= 0.0


# ---------------------------------------------------------------------------
# Push / size
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_push_increases_size():
    reg = UndoRegistry()
    assert reg.size == 0
    reg.push(_make_entry("s1"))
    assert reg.size == 1
    reg.push(_make_entry("s2"))
    assert reg.size == 2


@pytest.mark.unit
def test_len_equals_size():
    reg = UndoRegistry()
    reg.push(_make_entry("s1"))
    assert len(reg) == reg.size


@pytest.mark.unit
def test_push_expired_entry_is_still_stored_but_not_counted():
    """Expired entries are pruned on the *next* push, not immediately."""
    reg = UndoRegistry()
    expired = _make_entry("expired", expires_in=-1.0)
    reg.push(expired)
    # Because push prunes expired entries *before* adding the new one,
    # adding the expired entry itself means: prune (nothing yet), append,
    # so size is 1 immediately after push — but size ignores expired.
    assert reg.size == 0  # counted as expired


@pytest.mark.unit
def test_expired_entries_pruned_on_next_push():
    reg = UndoRegistry()
    expired = _make_entry("exp", expires_in=-1.0)
    # Directly append without pruning to seed an expired entry
    reg._entries.append(expired)
    assert len(reg._entries) == 1

    # A new push triggers pruning of the expired entry
    reg.push(_make_entry("fresh"))
    assert reg.size == 1
    assert reg._entries[0].step_id == "fresh"


@pytest.mark.unit
def test_cap_enforced_oldest_dropped():
    reg = UndoRegistry(max_entries=3)
    for i in range(5):
        reg.push(_make_entry(step_id=f"s{i}", description=f"step {i}"))
    # Only the 3 most-recent should remain
    assert reg.size == 3
    remaining_ids = [e.step_id for e in reg._entries]
    assert "s0" not in remaining_ids
    assert "s1" not in remaining_ids
    assert "s4" in remaining_ids


# ---------------------------------------------------------------------------
# pop_last
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pop_last_returns_most_recent_first():
    reg = UndoRegistry()
    reg.push(_make_entry("s1", expires_in=60))
    reg.push(_make_entry("s2", expires_in=60))
    reg.push(_make_entry("s3", expires_in=60))

    result = reg.pop_last(2)
    assert len(result) == 2
    assert result[0].step_id == "s3"  # most-recent first
    assert result[1].step_id == "s2"
    assert reg.size == 1
    assert reg._entries[0].step_id == "s1"


@pytest.mark.unit
def test_pop_last_1_removes_newest():
    reg = UndoRegistry()
    reg.push(_make_entry("older", expires_in=60))
    reg.push(_make_entry("newer", expires_in=60))

    result = reg.pop_last(1)
    assert result[0].step_id == "newer"
    assert reg.size == 1


@pytest.mark.unit
def test_pop_last_from_empty_registry_returns_empty():
    reg = UndoRegistry()
    assert reg.pop_last(3) == []


@pytest.mark.unit
def test_pop_last_skips_expired_entries():
    reg = UndoRegistry()
    reg._entries.append(_make_entry("e1", expires_in=-1.0))  # pre-expired
    reg._entries.append(_make_entry("e2", expires_in=60))

    result = reg.pop_last(1)
    assert len(result) == 1
    assert result[0].step_id == "e2"


@pytest.mark.unit
def test_pop_last_n_greater_than_stack():
    reg = UndoRegistry()
    reg.push(_make_entry("only", expires_in=60))

    result = reg.pop_last(10)
    assert len(result) == 1
    assert reg.size == 0


# ---------------------------------------------------------------------------
# pop_by_id
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pop_by_id_finds_and_removes_entry():
    reg = UndoRegistry()
    reg.push(_make_entry("target", expires_in=60))
    reg.push(_make_entry("other", expires_in=60))

    entry = reg.pop_by_id("target")
    assert entry is not None
    assert entry.step_id == "target"
    assert reg.size == 1
    assert reg._entries[0].step_id == "other"


@pytest.mark.unit
def test_pop_by_id_missing_step_returns_none():
    reg = UndoRegistry()
    assert reg.pop_by_id("nonexistent") is None


@pytest.mark.unit
def test_pop_by_id_expired_entry_returns_none_and_removes():
    reg = UndoRegistry()
    reg._entries.append(_make_entry("exp", expires_in=-1.0))

    result = reg.pop_by_id("exp")
    assert result is None
    assert len(reg._entries) == 0  # expired entry removed


# ---------------------------------------------------------------------------
# peek_all
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_peek_all_returns_non_expired_oldest_first():
    reg = UndoRegistry()
    reg.push(_make_entry("s1", expires_in=60))
    reg.push(_make_entry("s2", expires_in=60))

    snapshot = reg.peek_all()
    assert len(snapshot) == 2
    assert snapshot[0].step_id == "s1"
    assert snapshot[1].step_id == "s2"
    # peek_all must not remove entries
    assert reg.size == 2


@pytest.mark.unit
def test_peek_all_excludes_expired():
    reg = UndoRegistry()
    reg._entries.append(_make_entry("exp", expires_in=-1.0))
    reg._entries.append(_make_entry("fresh", expires_in=60))

    snapshot = reg.peek_all()
    assert len(snapshot) == 1
    assert snapshot[0].step_id == "fresh"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clear_empties_registry():
    reg = UndoRegistry()
    reg.push(_make_entry("s1", expires_in=60))
    reg.push(_make_entry("s2", expires_in=60))
    reg.clear()
    assert reg.size == 0
    assert reg.peek_all() == []


# ---------------------------------------------------------------------------
# Module-level singleton convenience wrappers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_push_undo_and_pop_last_undo_via_singleton(isolated_registry):
    push_undo(_make_entry("sin1", expires_in=60))
    push_undo(_make_entry("sin2", expires_in=60))

    result = pop_last_undo(1)
    assert len(result) == 1
    assert result[0].step_id == "sin2"


@pytest.mark.unit
def test_pop_undo_by_id_via_singleton(isolated_registry):
    push_undo(_make_entry("find_me", expires_in=60))
    push_undo(_make_entry("leave_me", expires_in=60))

    entry = pop_undo_by_id("find_me")
    assert entry is not None
    assert entry.step_id == "find_me"

    remaining = pop_last_undo(10)
    assert len(remaining) == 1
    assert remaining[0].step_id == "leave_me"


@pytest.mark.unit
def test_get_undo_registry_returns_singleton(isolated_registry):
    reg = get_undo_registry()
    push_undo(_make_entry("x", expires_in=60))
    assert reg.size == 1
