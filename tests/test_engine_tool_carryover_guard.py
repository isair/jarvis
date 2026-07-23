"""Engine-level tool carry-over guard.

Field trace (2026-05-03, gemma4:e2b):
  Turn 1 user: "how's the weather tomorrow Jarvis?" → no location set →
    assistant invokes ``getWeather``, tool returns ``success=False``
    ("I couldn't auto-detect your location, please tell me a city"),
    assistant relays the request.
  Turn 2 user: "I'm in London" → small-model router picks ``webSearch``
    instead of ``getWeather``, planner falls back to a web search for
    "weather in london tomorrow", DDG fails, Wikipedia matches the 2014
    film "Edge of Tomorrow", and the assistant parrots the film summary
    as the weather answer.

Fix: when the previous assistant turn invoked a tool that reported
``success=False`` on its ``ToolExecutionResult``, union the previous
turn's tool name into the allow-list. The ``tool_failed`` flag stamped
onto each recorded tool result is the truth source. Gating on failure
(rather than recency or query length) means a successful chain followed
by a genuine new short ask ("play some music") correctly does NOT carry
over the prior tool.

The carry-over is an engine-side per-turn overlay: the router cache
stores only the raw router output, so future identical queries are
unaffected.
"""

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine


def _mock_cfg():
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"
    cfg.llm_chat_model = "test-large"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.db_path = ":memory:"
    return cfg


def _tool_names_from_chat_call(call) -> set[str]:
    """Pull function names out of the OpenAI-style tools schema passed
    to chat_with_messages.
    """
    schema = call.kwargs.get("tools") or []
    names: set[str] = set()
    for entry in schema:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        nm = fn.get("name")
        if isinstance(nm, str):
            names.add(nm)
    return names


def _failed_tool_turn(tool_name: str, tool_call_id: str = "c1") -> list[dict]:
    """Plant a previous-turn tool turn where the tool was invoked and
    reported failure. Mirrors the message shape the engine records for a
    native tool call whose ``ToolExecutionResult.success`` was False.
    """
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": tool_call_id, "type": "function",
            "function": {"name": tool_name, "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": tool_call_id,
         "tool_name": tool_name,
         "content": "I couldn't auto-detect your location.",
         "tool_failed": True},
    ]


def _succeeded_tool_turn(tool_name: str, tool_call_id: str = "c1") -> list[dict]:
    """Plant a previous-turn tool turn where the tool succeeded."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": tool_call_id, "type": "function",
            "function": {"name": tool_name, "arguments": {"location": "London"}},
        }]},
        {"role": "tool", "tool_call_id": tool_call_id,
         "tool_name": tool_name,
         "content": "London: 15°C and partly cloudy.",
         "tool_failed": False},
    ]


def _failed_text_tool_turn(tool_name: str) -> list[dict]:
    """Plant a previous-turn tool turn in the text-tool fallback shape
    (small models). Tool error is appended as a ``role=user`` message
    tagged with both ``tool_name`` and ``tool_failed=True``.
    """
    return [
        {"role": "assistant",
         "content": (
             "```tool_call\n"
             '{"name": "' + tool_name + '", "arguments": {}}\n'
             "```"
         )},
        {"role": "user",
         "content": (
             "[Tool error: " + tool_name + "] I couldn't auto-detect "
             "your location."
         ),
         "tool_name": tool_name,
         "tool_failed": True},
    ]


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_followup_carries_over_failed_previous_tool(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Previous turn invoked ``getWeather`` and the tool reported failure;
    this turn's router only picked ``webSearch``. The engine must union
    ``getWeather`` back in so the chat model can re-call it with the
    location the user just supplied.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Weather in London is 15°C."}},
    ]
    mock_extract.side_effect = ["Weather in London is 15°C."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather tomorrow Jarvis")
    dm.record_tool_turn(_failed_tool_turn("getWeather"))
    dm.add_message("assistant", "I do not have a location set.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in London", dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "webSearch" in tool_names, (
        f"router pick must remain visible; saw {sorted(tool_names)}"
    )
    assert "getWeather" in tool_names, (
        "previous-turn failed tool must be carried over; "
        f"saw {sorted(tool_names)}"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_successful_previous_tool_does_not_trigger_carryover(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """A successful prior tool call means the chain completed. A genuine
    new short ask ("log my breakfast") must NOT inherit the prior tool —
    that would noisily widen the allow-list for unrelated turns and
    risks small models replaying the previous tool. The router pick
    stands on its own.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Logged."}},
    ]
    mock_extract.side_effect = ["Logged."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather in London")
    dm.record_tool_turn(_succeeded_tool_turn("getWeather"))
    dm.add_message("assistant", "It's 15°C and partly cloudy in London.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["logMeal"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="log my breakfast", dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "logMeal" in tool_names
    assert "getWeather" not in tool_names, (
        "successful prior tool must not be carried over; "
        f"saw {sorted(tool_names)}"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_cold_start_does_not_trigger_carryover(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Empty dialogue memory — the carry-over path must be a no-op."""
    mock_chat.side_effect = [
        {"message": {"content": "Hello."}},
    ]
    mock_extract.side_effect = ["Hello."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()  # cold start — no prior turns

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="hi", dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "webSearch" in tool_names
    assert "getWeather" not in tool_names


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_carryover_does_not_pollute_router_cache(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """The router cache stores the raw router output. Carry-over is a
    per-turn overlay layered on top — it must NOT be written back to the
    cache, otherwise every replay of the same query inherits a
    contaminated tool list.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Weather in London is 15°C."}},
    ]
    mock_extract.side_effect = ["Weather in London is 15°C."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather tomorrow Jarvis")
    dm.record_tool_turn(_failed_tool_turn("getWeather"))
    dm.add_message("assistant", "I do not have a location set.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in London", dialogue_memory=dm)

    cached_router_entries = [
        (k, v) for k, v in dm._hot_cache.items() if k.startswith("router:")
    ]
    assert cached_router_entries, "router output should have been cached"
    for key, (_ts, value) in cached_router_entries:
        assert value == ["webSearch"], (
            f"router cache for {key!r} should hold raw router output "
            f"['webSearch']; got {value!r}"
        )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_long_followup_still_carries_over_when_previous_failed(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Failure-gated carry-over does NOT depend on query length. A long
    follow-up that supplies the missing arg ("Right, sorry — I'm in
    Edinburgh, please try the lookup again for tomorrow") must still
    carry over the failed tool. The earlier char-length heuristic was
    dropped because it false-negatived this shape; the failure flag is
    the right signal.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Edinburgh weather: 12°C."}},
    ]
    mock_extract.side_effect = ["Edinburgh weather: 12°C."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather tomorrow Jarvis")
    dm.record_tool_turn(_failed_tool_turn("getWeather"))
    dm.add_message("assistant", "I do not have a location set.")

    long_followup = (
        "Right, sorry — I'm in Edinburgh, please try the lookup again for "
        "tomorrow morning if you would."
    )
    assert len(long_followup) > 80

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text=long_followup, dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "getWeather" in tool_names, (
        f"long follow-up to a failed tool must still carry over; "
        f"saw {sorted(tool_names)}"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_text_tool_fallback_failure_carries_over(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Small-model path: the previous turn's tool error was stored as a
    ``role=user`` message tagged with ``tool_name`` and
    ``tool_failed=True``. The walker must collect the name from this
    shape too, not only from native ``assistant.tool_calls`` + ``role=tool``
    pairs.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Weather in Berlin is 9°C."}},
    ]
    mock_extract.side_effect = ["Weather in Berlin is 9°C."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather")
    dm.record_tool_turn(_failed_text_tool_turn("getWeather"))
    dm.add_message("assistant", "I couldn't auto-detect your location.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in Berlin", dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "getWeather" in tool_names, (
        "text-tool fallback failure shape must be carried over; "
        f"saw {sorted(tool_names)}"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_multi_tool_call_only_failed_sibling_carries_over(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """When an assistant message carries multiple tool_calls but only
    one of them failed, only the failed name must be carried over. The
    successful sibling stays the chat model's responsibility through
    its own routing.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Sure."}},
    ]
    mock_extract.side_effect = ["Sure."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "weather and search Pushkin")
    dm.record_tool_turn([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-w", "type": "function",
             "function": {"name": "getWeather", "arguments": {}}},
            {"id": "c-s", "type": "function",
             "function": {"name": "webSearch", "arguments": {"query": "Pushkin"}}},
        ]},
        # getWeather failed (no location), webSearch succeeded.
        {"role": "tool", "tool_call_id": "c-w",
         "tool_name": "getWeather",
         "content": "I couldn't auto-detect your location.",
         "tool_failed": True},
        {"role": "tool", "tool_call_id": "c-s",
         "tool_name": "webSearch",
         "content": "Pushkin was a Russian poet (1799-1837).",
         "tool_failed": False},
    ])
    dm.add_message("assistant",
                   "Pushkin was a Russian poet. I couldn't auto-detect "
                   "your location for the weather lookup.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["fetchWebPage"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in Paris", dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "getWeather" in tool_names, (
        "failed sibling tool_call must be carried over; "
        f"saw {sorted(tool_names)}"
    )
    assert "webSearch" not in tool_names, (
        "successful sibling tool_call must NOT be carried over; "
        f"saw {sorted(tool_names)}"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
@patch("src.jarvis.reply.engine.run_tool_with_retries")
def test_planner_direct_exec_stamps_tool_failed(
    mock_tool, mock_chat, mock_extract, _mock_extract_mem,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """The planner's direct-exec path (text-tool mode + concrete plan
    step) appends tool results without going through the chat-model
    loop. Verify that path stamps ``tool_failed`` so the next turn's
    walker can see prior failures planted by direct-exec.
    """
    from src.jarvis.tools.types import ToolExecutionResult

    cfg = _mock_cfg()
    cfg.ollama_chat_model = "gemma4:e2b"  # triggers SMALL/text-tool path
    cfg.llm_chat_model = "gemma4:e2b"  # triggers SMALL/text-tool path

    # First reply: planner emits a getWeather step, direct-exec runs the
    # tool which returns success=False (no location), then the chat
    # model produces a final text reply.
    mock_tool.return_value = ToolExecutionResult(
        success=False,
        reply_text="I couldn't auto-detect your location.",
    )
    mock_chat.side_effect = [
        {"message": {"content": "Tell me which city."}},
    ]
    mock_extract.side_effect = ["Tell me which city."]

    db = Mock()
    dm = DialogueMemory()

    # Concrete plan step the resolver fast-path can parse without an LLM.
    with patch(
        "src.jarvis.reply.engine.plan_query",
        return_value=["getWeather", "Reply to the user."],
    ), patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["getWeather"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="how's the weather",
                         dialogue_memory=dm)

    # The direct-exec path should have recorded a tool turn with the
    # failure flag set so a follow-up turn can carry over getWeather.
    assert dm._tool_turns, (
        "planner direct-exec path must record a tool turn into "
        "dialogue memory carryover"
    )
    stored_msgs = [m for _ts, msgs in dm._tool_turns for m in msgs]
    failed_entries = [
        m for m in stored_msgs
        if m.get("tool_failed") and m.get("tool_name") == "getWeather"
    ]
    assert failed_entries, (
        "direct-exec failure must stamp tool_failed=True; "
        f"stored messages: {stored_msgs}"
    )


@pytest.mark.unit
def test_walker_logs_orphan_assistant_tool_call(caplog):
    """When an assistant tool_call has no matching role=tool result in
    the recent window (e.g. truncation, scrub, eviction), the walker
    should fail-open and log a diagnostic — never crash, never silently
    widen the allow-list with the orphan name.
    """
    from src.jarvis.reply.engine import _previous_turn_failed_tool_names

    recent = [
        {"role": "user", "content": "weather please"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-orphan", "type": "function",
             "function": {"name": "getWeather", "arguments": {}}},
        ]},
        # No matching role=tool result for c-orphan.
        {"role": "assistant", "content": "I couldn't auto-detect."},
    ]

    names = _previous_turn_failed_tool_names(recent)
    # No failed tool result was seen, so nothing carries over even
    # though an assistant tool_call exists.
    assert names == [], (
        f"orphan tool_call must not be carried over; got {names}"
    )
