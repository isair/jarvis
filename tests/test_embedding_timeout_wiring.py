"""Reply preparation and mid-turn discovery honour configured model budgets."""
import pytest

from jarvis.llm import Tier, resolve_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("budget_offset", [1.25, 17.5])
def test_reply_preparation_uses_configured_deadlines_and_fast_model(
    mock_config, monkeypatch, budget_offset,
):
    from jarvis.reply import engine

    cfg = mock_config
    cfg.llm_embedding_timeout_sec += budget_offset
    cfg.llm_tools_timeout_sec += budget_offset
    cfg.fast_model = "fast-control"
    cfg.llm_chat_model = "chat-control"
    cfg.agentic_max_turns = 0
    cfg.memory_enrichment_source = "diary"
    cfg.voice_debug = False
    observed = {}

    def route(**kwargs):
        observed["routing"] = kwargs
        return ["stop"]

    def extract(query, settings, model, **kwargs):
        observed["extraction"] = (model, kwargs["timeout_sec"])
        return {"keywords": ["city"], "questions": []}

    def search(**kwargs):
        observed["diary"] = kwargs["timeout_sec"]
        return []

    monkeypatch.setattr(engine, "select_tools", route)
    monkeypatch.setattr(engine, "plan_query", lambda *args, **kwargs: [])
    monkeypatch.setattr(engine, "extract_search_params_for_memory", extract)
    monkeypatch.setattr("jarvis.memory.conversation.search_conversation_memory_by_keywords", search)
    monkeypatch.setattr(engine, "chat_with_messages", lambda **kwargs: None)
    engine.run_reply_engine(None, cfg, None, "Which city did I mention when we discussed my commute last week?", None)

    assert observed["routing"]["embed_timeout_sec"] == cfg.llm_embedding_timeout_sec
    assert observed["routing"]["llm_timeout_sec"] == cfg.llm_tools_timeout_sec
    assert observed["routing"]["llm_model"] == resolve_model(cfg, Tier.FAST)
    assert observed["extraction"] == (resolve_model(cfg, Tier.FAST), cfg.llm_tools_timeout_sec)
    assert observed["diary"] == cfg.llm_embedding_timeout_sec


@pytest.mark.parametrize("budget_offset", [1.25, 17.5])
def test_mid_turn_discovery_uses_configured_deadlines_and_fast_model(
    mock_config, monkeypatch, budget_offset,
):
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin import tool_search

    cfg = mock_config
    cfg.llm_embedding_timeout_sec += budget_offset
    cfg.llm_tools_timeout_sec += budget_offset
    cfg.fast_model = "fast-control"
    cfg.llm_chat_model = "chat-control"
    observed = {}

    def route(**kwargs):
        observed.update(kwargs)
        return ["stop"]

    monkeypatch.setattr(tool_search, "select_tools", route)
    context = ToolContext(db=None, cfg=cfg, system_prompt="", original_prompt="weather",
                          redacted_text="weather", max_retries=0, user_print=lambda *_: None)
    result = tool_search.ToolSearchTool().run({"query": "weather"}, context)

    assert result.success is True
    assert result.reply_text == "No additional tools found for that description."
    assert observed["embed_timeout_sec"] == cfg.llm_embedding_timeout_sec
    assert observed["llm_timeout_sec"] == cfg.llm_tools_timeout_sec
    assert observed["llm_model"] == resolve_model(cfg, Tier.FAST)
