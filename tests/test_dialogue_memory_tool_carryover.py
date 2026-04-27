"""Tests for DialogueMemory tool-message carryover across turns.

Behaviour under test: within the hot-window (RECENT_WINDOW_SEC), tool-call
and tool-result messages generated during one reply must be retrievable as
part of the next reply's initial messages, so follow-up turns can reuse the
prior tool output instead of re-fetching.
"""

import time
import pytest

from src.jarvis.memory.conversation import DialogueMemory


@pytest.mark.unit
class TestToolCarryover:
    def test_record_tool_turn_stores_messages(self):
        dm = DialogueMemory()
        dm.add_message("user", "who is justin bieber")
        dm.record_tool_turn([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "webSearch",
                                  "arguments": {"query": "justin bieber"}}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1",
             "content": "Justin Bieber is a Canadian singer..."},
        ])
        dm.add_message("assistant", "He is a Canadian singer.")

        out = dm.get_recent_turns_with_tools()
        roles = [m.get("role") for m in out]
        # Order: user, assistant-with-tool_calls, tool, assistant
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert out[1].get("tool_calls")
        assert out[2].get("tool_call_id") == "call_1"
        assert "Canadian singer" in out[2]["content"]

    def test_carryover_survives_second_add_message(self):
        """Tool rows must interleave at the correct timestamps between text messages."""
        dm = DialogueMemory()
        dm.add_message("user", "q1")
        dm.record_tool_turn([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "webSearch",
                                          "arguments": {"query": "q1"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        ])
        dm.add_message("assistant", "a1")
        time.sleep(0.005)
        dm.add_message("user", "q2")

        out = dm.get_recent_turns_with_tools()
        roles = [m.get("role") for m in out]
        assert roles == ["user", "assistant", "tool", "assistant", "user"]

    def test_truncates_large_tool_content(self):
        dm = DialogueMemory()
        huge = "x" * 5000
        dm.add_message("user", "q")
        dm.record_tool_turn([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "webSearch",
                                          "arguments": {"query": "q"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": huge},
        ])
        out = dm.get_recent_turns_with_tools(per_entry_chars=1200)
        tool_msg = next(m for m in out if m.get("role") == "tool")
        assert len(tool_msg["content"]) <= 1201  # 1200 + ellipsis char

    def test_caps_to_max_tool_turns(self):
        dm = DialogueMemory()
        for i in range(4):
            dm.add_message("user", f"q{i}")
            dm.record_tool_turn([
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": f"c{i}", "type": "function",
                                 "function": {"name": "webSearch",
                                              "arguments": {"q": f"q{i}"}}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"},
            ])
            dm.add_message("assistant", f"a{i}")

        out = dm.get_recent_turns_with_tools(max_tool_turns=2)
        tool_contents = [m["content"] for m in out if m.get("role") == "tool"]
        # Only the most recent 2 tool turns survive
        assert tool_contents == ["r2", "r3"]

    def test_clear_tool_carryover_drops_tool_msgs_only(self):
        dm = DialogueMemory()
        dm.add_message("user", "q")
        dm.record_tool_turn([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "webSearch",
                                          "arguments": {"q": "x"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ])
        dm.add_message("assistant", "a")

        dm.clear_tool_carryover()

        out = dm.get_recent_turns_with_tools()
        roles = [m.get("role") for m in out]
        # Tool rows gone, but user/assistant prose preserved
        assert roles == ["user", "assistant"]

    def test_tool_turns_survive_past_recent_window_age(self):
        """Tool carryover is conversation-scoped, not RECENT_WINDOW_SEC-
        bounded. An ongoing conversation must keep prior tool results
        visible regardless of how long ago each tool fired; the engine
        clears them on new-conversation entry and on ``stop``.
        """
        dm = DialogueMemory(inactivity_timeout=300.0)
        dm.add_message("user", "q")
        dm.record_tool_turn([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "webSearch",
                                          "arguments": {"q": "x"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ])
        # Even when we backdate the tool-turn timestamp past the window,
        # the carryover survives until explicitly cleared.
        with dm._lock:
            dm._tool_turns = [
                (ts - (dm.RECENT_WINDOW_SEC + 10), msgs)
                for ts, msgs in dm._tool_turns
            ]

        out = dm.get_recent_turns_with_tools()
        assert any(m.get("role") == "tool" for m in out), (
            "tool carryover must persist beyond RECENT_WINDOW_SEC age"
        )

        dm.clear_tool_carryover()
        out_after_clear = dm.get_recent_turns_with_tools()
        assert not any(m.get("role") == "tool" for m in out_after_clear)

    def test_tool_call_arguments_are_scrubbed(self):
        """Native tool-call arguments can carry secrets too (e.g. an
        email or token in the search query). They must be scrubbed
        on record so re-injection on the next turn cannot leak them.
        """
        dm = DialogueMemory()
        dm.record_tool_turn([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "webSearch",
                        "arguments": {
                            "query": "look up alice@example.com please",
                        },
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        stored_call = dm._tool_turns[0][1][0]["tool_calls"][0]
        stored_args = stored_call["function"]["arguments"]
        assert "alice@example.com" not in stored_args["query"]
        assert "[REDACTED_EMAIL]" in stored_args["query"]

    def test_tool_call_arguments_list_form_is_scrubbed(self):
        """Some providers / custom tools pass arguments as a list of
        scalars or dicts. Each element must be scrubbed too — otherwise
        a positional secret slips through.
        """
        dm = DialogueMemory()
        dm.record_tool_turn([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": [
                            "alice@example.com",
                            {"note": "ping bob@example.com"},
                        ],
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        stored = dm._tool_turns[0][1][0]["tool_calls"][0]["function"]["arguments"]
        flat = repr(stored)
        assert "alice@example.com" not in flat
        assert "bob@example.com" not in flat
        assert flat.count("[REDACTED_EMAIL]") >= 2

    def test_tool_call_arguments_string_form_is_scrubbed(self):
        """Some providers serialise arguments as a JSON string, not a dict."""
        dm = DialogueMemory()
        dm.record_tool_turn([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "webSearch",
                        "arguments": '{"query": "alice@example.com"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        stored_args = dm._tool_turns[0][1][0]["tool_calls"][0]["function"]["arguments"]
        assert "alice@example.com" not in stored_args
        assert "[REDACTED_EMAIL]" in stored_args

    def test_tool_payloads_are_scrubbed_of_secrets(self):
        """Tool results may contain emails, API tokens, JWTs. record_tool_turn
        must scrub those before persisting so follow-up injection can't leak.
        """
        dm = DialogueMemory()
        dm.add_message("user", "look up the api")
        dirty = (
            "Contact: alice@example.com\n"
            "Bearer token: eyJhbGciOiJIUzI1NiJ9.abc.def\n"
            "Fine content stays."
        )
        dm.record_tool_turn([
            {"role": "tool", "tool_call_id": "c1", "content": dirty},
        ])
        stored = dm._tool_turns[0][1][0]["content"]
        assert "alice@example.com" not in stored
        assert "[REDACTED_EMAIL]" in stored
        assert "eyJhbGciOiJIUzI1NiJ9" not in stored
        assert "Fine content stays." in stored

    def test_truncation_preserves_untrusted_fence_end_marker(self):
        """When a tool result carrying an UNTRUSTED WEB EXTRACT fence is
        truncated, the closing marker must be re-appended so the downstream
        prompt-injection defence fence stays intact.
        """
        dm = DialogueMemory()
        dm.add_message("user", "q")
        begin = "<<<BEGIN UNTRUSTED WEB EXTRACT>>>"
        end = "<<<END UNTRUSTED WEB EXTRACT>>>"
        payload = (
            "Search result:\n" + begin + "\n" + ("x" * 5000) + "\n" + end
        )
        dm.record_tool_turn([
            {"role": "tool", "tool_call_id": "c1", "content": payload},
        ])
        out = dm.get_recent_turns_with_tools(per_entry_chars=500)
        tool_msg = next(m for m in out if m.get("role") == "tool")
        assert begin in tool_msg["content"]
        assert end in tool_msg["content"], (
            "closing fence marker must survive truncation"
        )

    def test_get_pending_chunks_excludes_tool_rows(self):
        """Tool messages must not pollute the diary summariser input."""
        dm = DialogueMemory()
        dm.add_message("user", "q")
        dm.record_tool_turn([
            {"role": "tool", "tool_call_id": "c1",
             "content": "raw web extract with secrets"},
        ])
        dm.add_message("assistant", "a")

        chunks = dm.get_pending_chunks()
        joined = " | ".join(chunks)
        assert "raw web extract" not in joined
        assert "User: q" in joined
        assert "Assistant: a" in joined
