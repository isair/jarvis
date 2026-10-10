"""Tests for the DialogueMemory conversation-scoped scratch cache and the
``is_tool_message`` helper.

The cache is a per-conversation primitive used by the reply engine to
memoise idempotent per-turn work (warm profile, memory extractor, tool
router). Entries persist for the lifetime of the active conversation and
are wiped on ``clear_hot_cache()``; the warm profile entry can also be
invalidated on demand via ``invalidate_warm_profile()``.
"""

import time
from types import SimpleNamespace

from src.jarvis.memory import conversation

import pytest

from src.jarvis.memory.conversation import DialogueMemory, is_tool_message


@pytest.mark.unit
class TestHotCachePrimitives:
    @pytest.mark.parametrize('value', [{"v": 1}, "", [], {}, False, 0, None])
    def test_cache_roundtrip_and_replacement(self, value):
        dm = DialogueMemory()
        assert dm.hot_cache_get("k") is None
        dm.hot_cache_put("k", value)
        assert dm.hot_cache_get("k") == value
        dm.hot_cache_put("k", "replacement")
        assert dm.hot_cache_get("k") == "replacement"

    def test_entries_persist_past_recent_window_age(self, monkeypatch):
        """Elapsed time alone does not expire conversation-scoped entries."""
        clock = {"now": time.time()}
        monkeypatch.setattr(conversation, 'time', SimpleNamespace(time=lambda: clock['now']))
        dm = DialogueMemory(inactivity_timeout=300.0)
        dm.hot_cache_put("k", "v")
        clock['now'] += dm.RECENT_WINDOW_SEC + 10
        assert dm.hot_cache_get("k") == "v"

    def test_invalidate_warm_profile_drops_only_that_key(self):
        dm = DialogueMemory()
        dm.hot_cache_put(dm.WARM_PROFILE_CACHE_KEY, "warm-block")
        dm.hot_cache_put("router:abc", ["webSearch"])
        dm.invalidate_warm_profile()
        assert dm.hot_cache_get(dm.WARM_PROFILE_CACHE_KEY) is None
        assert dm.hot_cache_get("router:abc") == ["webSearch"]

    def test_clear_hot_cache_drops_all_entries(self):
        dm = DialogueMemory()
        dm.hot_cache_put("a", 1)
        dm.hot_cache_put("b", 2)
        dm.clear_hot_cache()
        assert dm.hot_cache_get("a") is None
        assert dm.hot_cache_get("b") is None



@pytest.mark.unit
class TestHotCacheLRUCap:
    """The hot cache must not grow without bound. Per-query keys (router
    output, enrichment extractor output) are unique per turn, so a long
    session would otherwise accumulate one entry per unique query.
    """

    def test_only_the_latest_capacity_entries_remain(self):
        dm = DialogueMemory()
        cap = dm.HOT_CACHE_MAX_ENTRIES
        count = cap + 50
        for i in range(count):
            dm.hot_cache_put(f"key:{i}", i)
        for i in range(count):
            assert dm.hot_cache_get(f"key:{i}") == (i if i >= count - cap else None)

    @pytest.mark.parametrize('access', ['read', 'replace'])
    @pytest.mark.parametrize('oldest_value', [0, False, '', [], {}, None])
    def test_access_preserves_oldest_entry_and_evicts_next(self, access, oldest_value):
        dm = DialogueMemory()
        cap = dm.HOT_CACHE_MAX_ENTRIES
        dm.hot_cache_put("k0", oldest_value)
        for i in range(1, cap):
            dm.hot_cache_put(f"k{i}", i)
        if access == 'read':
            assert dm.hot_cache_get("k0") == oldest_value
            expected = oldest_value
        else:
            expected = "updated"
            dm.hot_cache_put("k0", expected)
        dm.hot_cache_put("new", "v")
        assert dm.hot_cache_get("k0") == expected
        assert dm.hot_cache_get("k1") is None
        assert dm.hot_cache_get("new") == "v"
        for i in range(2, cap):
            assert dm.hot_cache_get(f"k{i}") == i


@pytest.mark.unit
class TestNextTsMonotonic:
    """``_next_ts`` exists because ``time.time()`` has ~16ms granularity
    on Windows and consecutive calls can return identical values. Without
    the epsilon bump, text/tool messages recorded in the same tick would
    collide and break interleave ordering downstream.
    """

    def test_consecutive_calls_strictly_increase(self):
        dm = DialogueMemory()
        with dm._lock:
            t1 = dm._next_ts()
            t2 = dm._next_ts()
            t3 = dm._next_ts()
        assert t1 < t2 < t3

    def test_advances_past_artificially_high_last_ts(self):
        """Even if ``_last_ts`` is ahead of the wall clock (clock skew,
        manual seed), the next call must still advance.
        """
        dm = DialogueMemory()
        future = time.time() + 100.0
        with dm._lock:
            dm._last_ts = future
            nxt = dm._next_ts()
        assert nxt > future
        assert nxt - future < 0.01  # only an epsilon bump, not a wall jump


@pytest.mark.unit
class TestToolTurnsStorageCap:
    def test_tool_turns_capped_to_max_storage(self):
        dm = DialogueMemory()
        # Push more entries than the cap; each call appends one turn.
        for i in range(dm._tool_turns_max_storage + 5):
            dm.record_tool_turn([
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"},
            ])
        assert len(dm._tool_turns) == dm._tool_turns_max_storage
        # The oldest entries are dropped — last one survives.
        last_msg = dm._tool_turns[-1][1][0]["content"]
        assert last_msg.endswith(str(dm._tool_turns_max_storage + 4))


@pytest.mark.unit
class TestIsToolMessage:
    def test_native_tool_role(self):
        assert is_tool_message({"role": "tool", "content": "x"}) is True

    def test_assistant_with_tool_calls(self):
        assert is_tool_message({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1"}],
        }) is True

    def test_assistant_without_tool_calls(self):
        assert is_tool_message({"role": "assistant", "content": "hi"}) is False

    def test_text_tool_user_with_tool_name(self):
        assert is_tool_message({
            "role": "user", "content": "result", "tool_name": "webSearch",
        }) is True

    def test_plain_user_message(self):
        assert is_tool_message({"role": "user", "content": "hi"}) is False

    def test_non_dict_returns_false(self):
        assert is_tool_message("tool") is False
        assert is_tool_message(None) is False
