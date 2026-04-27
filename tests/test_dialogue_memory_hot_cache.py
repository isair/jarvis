"""Tests for the DialogueMemory conversation-scoped scratch cache and the
``is_tool_message`` helper.

The cache is a per-conversation primitive used by the reply engine to
memoise idempotent per-turn work (warm profile, memory extractor, tool
router). Entries persist for the lifetime of the active conversation and
are wiped on ``clear_hot_cache()``; the warm profile entry can also be
invalidated on demand via ``invalidate_warm_profile()``.
"""

import pytest

from src.jarvis.memory.conversation import DialogueMemory, is_tool_message


@pytest.mark.unit
class TestHotCachePrimitives:
    def test_get_returns_none_for_missing_key(self):
        dm = DialogueMemory()
        assert dm.hot_cache_get("nope") is None

    def test_put_then_get_roundtrips(self):
        dm = DialogueMemory()
        dm.hot_cache_put("k", {"v": 1})
        assert dm.hot_cache_get("k") == {"v": 1}

    def test_entries_persist_past_recent_window_age(self):
        """Cache entries are conversation-scoped, not bounded by
        RECENT_WINDOW_SEC. A long active conversation must keep the
        cache hot even when the original write is older than the window.
        """
        dm = DialogueMemory(inactivity_timeout=300.0)
        dm.hot_cache_put("k", "v")
        with dm._lock:
            ts, value = dm._hot_cache["k"]
            dm._hot_cache["k"] = (ts - (dm.RECENT_WINDOW_SEC + 10), value)
        # Age alone must NOT cause the value to disappear; only explicit
        # invalidation should drop it.
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

    def test_put_overwrites_existing_value(self):
        dm = DialogueMemory()
        dm.hot_cache_put("k", "old")
        dm.hot_cache_put("k", "new")
        assert dm.hot_cache_get("k") == "new"


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
