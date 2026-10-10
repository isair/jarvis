"""End-to-end coverage for the hot-window scratch caches in run_reply_engine.

Three caches share one primitive (DialogueMemory.hot_cache_*):

1. Warm profile block — query-agnostic, keyed on a constant.
2. Memory enrichment extractor — keyed on query (+topic hint) and context.
3. Tool router output — keyed on query, context, strategy and tool definitions.

Identical data inputs can reuse results within the active conversation.
Changed dialogue, live facts or tool definitions require fresh computations.

Also covers the C1 fix: when the planner explicitly emits a `searchMemory`
step, the recall gate must NOT short-circuit memory enrichment even when
hot-window coverage is high.
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


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="warm-block")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "u", "directives": "d"})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=[])
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_warm_profile_cached_across_turns(
    mock_chat, mock_extract, _mock_extractor, _mock_plan,
    _mock_select, _mock_graph, mock_build, _mock_fmt,
):
    """Warm profile is query-agnostic; second turn must reuse the cached
    block instead of re-traversing the graph store.
    """
    mock_chat.side_effect = [
        {"message": {"content": "a"}},
        {"message": {"content": "b"}},
    ]
    mock_extract.side_effect = ["a", "b"]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None, text="hi", dialogue_memory=dm)
    run_reply_engine(db=db, cfg=cfg, tts=None, text="anything else", dialogue_memory=dm)

    assert mock_build.call_count == 1, (
        f"warm profile should be built once and cached; got {mock_build.call_count} calls"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=["webSearch"])
@patch(
    "src.jarvis.reply.engine.plan_query",
    return_value=["searchMemory topic='justin bieber'", "reply"],
)
@patch("src.jarvis.reply.engine.extract_search_params_for_memory",
       return_value={"keywords": ["bieber"], "questions": []})
@patch("src.jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[])
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_planner_search_memory_overrides_recall_gate(
    mock_chat, mock_extract, _mock_search, mock_extractor,
    _mock_plan, _mock_select, _mock_graph, _mock_warm, _mock_fmt,
):
    """C1 fix: when the planner emits `searchMemory`, the recall gate must
    NOT short-circuit memory enrichment even though the hot window contains
    a fresh tool result that overlaps the query.
    """
    mock_chat.side_effect = [
        {"message": {"content": "Canadian singer."}},
    ]
    mock_extract.side_effect = ["Canadian singer."]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()
    # Plant a fresh tool result that would otherwise satisfy the recall gate.
    dm.add_message("user", "who is justin bieber")
    dm.record_tool_turn([
        {"role": "tool", "tool_call_id": "c1",
         "content": "Justin Bieber is a Canadian singer with the song Baby."},
    ])
    dm.add_message("assistant", "Canadian singer.")

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="bieber more about justin", dialogue_memory=dm)

    # Planner explicitly demanded memory → extractor must run.
    assert mock_extractor.call_count == 1, (
        "extractor must run when planner emits searchMemory, "
        "regardless of recall-gate coverage"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=[])
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_new_conversation_clears_cache_and_carryover(
    mock_chat, mock_extract, _mock_extractor, _mock_plan, mock_select,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """When the previous conversation has lapsed past the inactivity
    window, the engine must wipe the conversation-scoped cache and any
    leftover tool carryover before running the new turn. Otherwise stale
    state from a previous session would leak into a fresh one.
    """
    mock_chat.side_effect = [
        {"message": {"content": "fresh"}},
    ]
    mock_extract.side_effect = ["fresh"]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    # Plant cache + carryover from a prior (now-lapsed) session.
    dm.hot_cache_put(dm.WARM_PROFILE_CACHE_KEY, "old-block")
    dm.hot_cache_put("router:old", ["webSearch"])
    dm.record_tool_turn([
        {"role": "tool", "tool_call_id": "c1", "content": "ancient result"},
    ])
    assert dm._tool_turns
    assert dm.hot_cache_get(dm.WARM_PROFILE_CACHE_KEY) == "old-block"

    # No recent messages → engine treats this turn as a new conversation.
    run_reply_engine(db=db, cfg=cfg, tts=None, text="hello", dialogue_memory=dm)

    # Stale router entry must be gone (full hot-cache wipe), and the old
    # tool carryover must not be visible to the new conversation.
    assert dm.hot_cache_get("router:old") is None
    # The tool carryover from before must have been cleared on entry; any
    # tool turns recorded later in this turn would only come from THIS
    # reply (mock chat returns a final reply with no tool calls).
    assert dm._tool_turns == []
