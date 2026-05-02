"""End-to-end: tool-call + tool-result messages from one reply must be
visible to the LLM on the next reply within the hot window, so the model
can synthesise from prior results rather than re-fetching.
"""

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine


def _mock_cfg():
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"  # avoid SMALL-model text-tool path
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
    return cfg


@pytest.mark.unit
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_tool_carryover_makes_prior_result_visible_to_next_turn(
    mock_chat, mock_extract, mock_tool, _mock_extract, _mock_plan
):
    # Turn 1: model emits webSearch call, then final text.
    mock_tool.return_value = Mock(
        reply_text="Justin Bieber is a Canadian singer.",
        error_message=None,
    )
    mock_chat.side_effect = [
        # Turn 1a: tool call
        {"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "webSearch",
                         "arguments": {"query": "justin bieber"}},
        }]}},
        # Turn 1b: final reply
        {"message": {"content": "He is a Canadian singer."}},
        # Turn 2a: final reply directly — reuse from prior context
        {"message": {"content": "His breakout song was Baby."}},
    ]
    mock_extract.side_effect = [
        "",
        "He is a Canadian singer.",
        "His breakout song was Baby.",
    ]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="who is justin bieber",
                     dialogue_memory=dm)

    # Confirm carryover was recorded
    assert len(dm._tool_turns) == 1
    stored = dm._tool_turns[0][1]
    stored_roles = [m.get("role") for m in stored]
    assert "tool" in stored_roles
    assert any(m.get("tool_calls") for m in stored)

    # Turn 2: query on the same topic — the turn-2 LLM call should receive
    # the turn-1 tool messages in its `messages` argument.
    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="what is his most famous song",
                     dialogue_memory=dm)

    # The third chat_with_messages call is turn-2's only turn (single text).
    turn2_kwargs = mock_chat.call_args_list[-1].kwargs
    turn2_messages = turn2_kwargs.get("messages")
    roles_in_turn2 = [m.get("role") for m in turn2_messages]
    assert "tool" in roles_in_turn2, (
        f"Expected prior tool-role message to be injected on turn 2; "
        f"got roles={roles_in_turn2}"
    )
    # The tool message content must be the prior webSearch result
    tool_contents = [
        m.get("content") for m in turn2_messages if m.get("role") == "tool"
    ]
    assert any("Canadian singer" in (c or "") for c in tool_contents)


@pytest.mark.unit
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_stop_signal_clears_tool_carryover(
    mock_chat, mock_extract, mock_tool, _mock_extract, _mock_plan
):
    """Turn 1 runs a tool; turn 2 receives the stop signal. After turn 2,
    carryover must be empty so the next wake-word turn starts fresh.
    """
    from src.jarvis.tools.builtin.stop import STOP_SIGNAL

    mock_tool.side_effect = [
        Mock(reply_text="Justin Bieber is a Canadian singer.", error_message=None),
        Mock(reply_text=STOP_SIGNAL, error_message=None),
    ]
    mock_chat.side_effect = [
        # Turn 1a: tool call
        {"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "webSearch", "arguments": {"query": "bieber"}},
        }]}},
        # Turn 1b: final reply
        {"message": {"content": "He is a Canadian singer."}},
        # Turn 2: stop tool
        {"message": {"content": "", "tool_calls": [{
            "id": "c2", "type": "function",
            "function": {"name": "stop", "arguments": {}},
        }]}},
    ]
    mock_extract.side_effect = ["", "He is a Canadian singer.", ""]

    db = Mock()
    cfg = _mock_cfg()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="who is justin bieber", dialogue_memory=dm)
    assert len(dm._tool_turns) == 1, "turn-1 tool carryover should be recorded"

    reply = run_reply_engine(db=db, cfg=cfg, tts=None,
                             text="stop", dialogue_memory=dm)
    assert reply is None, "stop signal returns None"
    assert dm._tool_turns == [], (
        "stop signal must clear carryover so the next wake-word turn is clean"
    )


@pytest.mark.unit
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_tool_carryover_text_tool_mode(
    mock_chat, mock_extract, mock_tool, _mock_extract, _mock_plan
):
    """Small-model path: tool results come back as role=user with a
    ``tool_name`` tag. Carryover must pick those up too.
    """
    cfg = _mock_cfg()
    cfg.ollama_chat_model = "gemma4:e2b"  # triggers SMALL/text-tool path

    mock_tool.return_value = Mock(
        reply_text="Paris is the capital of France.", error_message=None,
    )
    fence_call = (
        "```tool_call\n"
        '{"name": "webSearch", "arguments": {"query": "paris"}}\n'
        "```"
    )
    mock_chat.side_effect = [
        # Turn 1a: text-tool call emitted inside a markdown fence
        {"message": {"content": fence_call}},
        # Turn 1b: final reply
        {"message": {"content": "Paris is in France."}},
        # Turn 2: follow-up reply
        {"message": {"content": "Its population is about 2.1 million."}},
    ]
    mock_extract.side_effect = [
        fence_call,
        "Paris is in France.",
        "Its population is about 2.1 million.",
    ]

    db = Mock()
    dm = DialogueMemory()

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="what about paris", dialogue_memory=dm)

    assert len(dm._tool_turns) == 1
    stored = dm._tool_turns[0][1]
    roles = [m.get("role") for m in stored]
    # Text-tool fallback stores tool results as role=user with tool_name.
    assert "user" in roles
    assert any(m.get("tool_name") == "webSearch" for m in stored)

    run_reply_engine(db=db, cfg=cfg, tts=None,
                     text="tell me more", dialogue_memory=dm)

    turn2_messages = mock_chat.call_args_list[-1].kwargs.get("messages") or []
    # The prior tool payload should appear in the turn-2 messages list —
    # either as role=tool (native) or role=user with tool_name (text-tool).
    tool_like = [
        m for m in turn2_messages
        if m.get("role") == "tool"
        or (m.get("role") == "user" and m.get("tool_name"))
    ]
    assert tool_like, (
        f"expected prior text-tool result to be carried over; got roles="
        f"{[m.get('role') for m in turn2_messages]}"
    )
    assert any(
        "Paris" in (m.get("content") or "") for m in tool_like
    )
