"""End-to-end coverage for the hot-window scratch caches in run_reply_engine.

Three caches share one primitive (DialogueMemory.hot_cache_*):

1. Warm profile block — query-agnostic, keyed on a constant.
2. Memory enrichment extractor — keyed on the redacted query (+topic hint).
3. Tool router output — keyed on redacted query + strategy + catalogue.

All three should fire on the second matching turn within the hot window so
follow-up queries don't pay for SQLite reads or LLM hops they already did.

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


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=[])
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_tool_router_cached_across_turns(
    mock_chat, mock_extract, mock_extractor, mock_plan, mock_select,
    _mock_graph, _mock_warm, _mock_fmt,
):
    """Two identical queries within the same DialogueMemory should call the
    tool router exactly once — the second turn must hit the hot-window cache.
    """
    mock_chat.side_effect = [
        {"message": {"content": "hello"}},
        {"message": {"content": "hello again"}},
    ]
    mock_extract.side_effect = ["hello", "hello again"]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None, text="say hi", dialogue_memory=dm)
    run_reply_engine(db=db, cfg=cfg, tts=None, text="say hi", dialogue_memory=dm)

    assert mock_select.call_count == 1, (
        f"router should be cached on identical query; called {mock_select.call_count} times"
    )


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=[])
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={"keywords": ["x"], "questions": []})
@patch("src.jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[])
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_memory_extractor_cached_across_turns(
    mock_chat, mock_extract, _mock_search, mock_extractor,
    _mock_plan, _mock_select, _mock_graph, _mock_warm, _mock_fmt,
):
    """Empty plan → fail-open path runs the extractor. The second identical
    follow-up must skip the extractor LLM call.

    The recall gate would also fire on a tool-grounded follow-up, so we
    keep the dialogue free of tool messages here to exercise the extractor
    path on both turns.
    """
    mock_chat.side_effect = [
        {"message": {"content": "first"}},
        {"message": {"content": "second"}},
    ]
    mock_extract.side_effect = ["first", "second"]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="tell me about pushkin", dialogue_memory=dm)
    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="tell me about pushkin", dialogue_memory=dm)

    assert mock_extractor.call_count == 1, (
        f"extractor should be cached; called {mock_extractor.call_count} times"
    )


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
@patch("src.jarvis.reply.engine.select_tools", return_value=[])
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
