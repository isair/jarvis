"""Engine-level tool carry-over guard.

Field trace (2026-05-03, gemma4:e2b):
  Turn 1 user: "how's the weather tomorrow Jarvis?" → no location set →
    assistant invokes ``getWeather``, gets "no location set", replies asking
    for a location.
  Turn 2 user: "I'm in London" → small-model router picks ``webSearch``
    instead of ``getWeather``, planner falls back to a web search for
    "weather in london tomorrow", DDG fails, Wikipedia matches the 2014 film
    "Edge of Tomorrow", and the assistant parrots the film summary as the
    weather answer.

Fix: when the previous assistant turn invoked a tool and the current user
query is a short follow-up (≤ ~80 chars), union the previous turn's tool
names back into the allow-list even if the small router missed them. This
lets the chat model (or planner direct-exec) continue the original tool
chain with the new info the user just supplied.

The carry-over is an engine-side per-turn overlay: the router cache stores
only the raw router output, so future identical queries are unaffected.
"""

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine


def _mock_cfg():
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embed_timeout_sec = 10.0
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


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_short_followup_carries_over_previous_turn_tool(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Previous turn invoked ``getWeather``; this turn's router only picked
    ``webSearch``. The engine must union ``getWeather`` back in so the
    chat model can re-call it with the location the user just supplied.
    """
    mock_chat.side_effect = [
        # Turn 2 only — single text reply.
        {"message": {"content": "Weather in London is 15°C."}},
    ]
    mock_extract.side_effect = ["Weather in London is 15°C."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    # Plant the previous turn's footprint: user asked for the weather, the
    # assistant invoked getWeather, the tool reported no location, the
    # assistant asked for one.
    dm.add_message("user", "how's the weather tomorrow Jarvis")
    dm.record_tool_turn([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "getWeather", "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "No location is configured."},
    ])
    dm.add_message("assistant", "I do not have a location set.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in London", dialogue_memory=dm)

    # The chat model on turn 2 must see BOTH webSearch and getWeather.
    turn2_call = mock_chat.call_args_list[-1]
    tool_names = _tool_names_from_chat_call(turn2_call)
    assert "webSearch" in tool_names, (
        f"router pick must remain visible; saw {sorted(tool_names)}"
    )
    assert "getWeather" in tool_names, (
        "previous-turn tool must be carried over for short follow-ups; "
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
def test_long_followup_does_not_trigger_carryover(
    mock_chat, mock_extract, _mock_extract_mem, _mock_plan,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """A long follow-up means the user has likely changed topics on purpose.
    Carry-over must NOT fire — the router pick is authoritative.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Sure thing."}},
    ]
    mock_extract.side_effect = ["Sure thing."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    dm.add_message("user", "how's the weather tomorrow Jarvis")
    dm.record_tool_turn([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "getWeather", "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": "No location."},
    ])
    dm.add_message("assistant", "I do not have a location set.")

    long_query = (
        "Forget the weather — instead can you write me a haiku about "
        "the futility of cataloguing chess endgames in long-form prose."
    )
    assert len(long_query) > 80

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text=long_query, dialogue_memory=dm)

    tool_names = _tool_names_from_chat_call(mock_chat.call_args_list[-1])
    assert "webSearch" in tool_names
    assert "getWeather" not in tool_names, (
        "long-query follow-ups should not inherit prior-turn tools; "
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
    # No prior tool turn → no carry-over candidates at all.
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
    dm.record_tool_turn([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "getWeather", "arguments": {}},
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": "No location."},
    ])
    dm.add_message("assistant", "I do not have a location set.")

    with patch(
        "src.jarvis.reply.engine.select_tools",
        return_value=["webSearch"],
    ):
        run_reply_engine(db=db, cfg=cfg, tts=None,
                         text="I'm in London", dialogue_memory=dm)

    # Find the router cache entry for "I'm in London" and confirm it stores
    # only the raw router output (webSearch), not the carry-over union.
    cached_router_entries = [
        (k, v) for k, v in dm._hot_cache.items() if k.startswith("router:")
    ]
    assert cached_router_entries, "router output should have been cached"
    for key, (_ts, value) in cached_router_entries:
        assert value == ["webSearch"], (
            f"router cache for {key!r} should hold raw router output "
            f"['webSearch']; got {value!r}"
        )
