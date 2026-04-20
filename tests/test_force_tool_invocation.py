"""
Unit tests for the force-invocation safety net.

Small models (gemma4:e2b, 2B params) sometimes ignore the tool router's
selection and either:

  1. Emit text from priors without invoking any tool, even when the
     router confidently picked exactly one real tool. The user then sees
     a confabulated answer.

  2. Emit gemma's native Google-training tool_code syntax targeting a
     tool that isn't in the routed allow-list (e.g. google_search.search
     when only getWeather is routed). Our parser can't dispatch it, so
     the raw syntax leaks to the user.

Both are captured from a 2026-04-20 field session on gemma4:e2b. The
force-invocation safety net covers both cases by invoking the router's
single selected tool when the content signals either failure mode.

These tests mock the reply engine's chat call and the tool runner, so
no real Ollama instance is required.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


@dataclass
class ToolCallCapture:
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: Dict[str, Any]):
        self.calls.append({"name": name, "args": args or {}})

    def has_tool(self, name: str) -> bool:
        return any(c["name"] == name for c in self.calls)

    def tool_names(self) -> List[str]:
        return [c["name"] for c in self.calls]


def _msg(content: str):
    return {"message": {"content": content, "role": "assistant"}}


def _run_engine_with_fixed_router(
    *,
    user_query: str,
    model: str,
    router_pick: List[str],
    chat_responses: List[Dict[str, Any]],
    mock_config,
    db,
    dialogue_memory,
    tool_reply: str = "OK-TOOL-RESULT",
):
    """Drive the reply engine with a scripted LLM and router selection.

    Returns (response_text, capture) where capture records all tool calls.
    """
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = model
    capture = ToolCallCapture()

    def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=True, reply_text=tool_reply)

    responses_iter = iter(chat_responses)

    def mock_chat(
        base_url, chat_model, messages, timeout_sec,
        extra_options=None, tools=None, thinking=False,
    ):
        try:
            return next(responses_iter)
        except StopIteration:
            # Final fallback if engine loops more than scripted — plain reply.
            return _msg("Done.")

    with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
         patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
         patch('jarvis.reply.engine.select_tools', return_value=router_pick), \
         patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

        response = run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text=user_query, dialogue_memory=dialogue_memory,
        )
    return response, capture


class TestForceInvocationGemmaToolCodeLeak:
    """Fix 1: gemma emits tool_code targeting a tool we didn't route.

    Field capture (2026-04-20): user asked "how is the weather today" after
    a hot-window turn. Router picked getWeather. Model emitted:

        tool_code
        print(google_search.search(query="current weather"))<unused88>

    Our parser can't dispatch google_search against a {getWeather, stop}
    allow-list. Before this fix, the raw syntax leaked to the user.
    """

    def test_tool_code_leak_triggers_force_invoke_of_router_tool(
        self, mock_config, db, dialogue_memory,
    ):
        raw_gemma_leak = (
            'tool_code\n'
            'print(google_search.search(query="current weather"))<unused88>'
        )
        response, capture = _run_engine_with_fixed_router(
            user_query="how is the weather today?",
            model="gemma4:e2b",
            router_pick=["getWeather", "stop"],
            chat_responses=[
                _msg(raw_gemma_leak),
                # After force-invoke + tool result, model produces a proper reply.
                _msg("It's 14°C and partly cloudy in your area."),
            ],
            mock_config=mock_config, db=db, dialogue_memory=dialogue_memory,
            tool_reply="Weather in Hackney: 14C partly cloudy.",
        )

        # The router's selected tool must have been invoked even though the
        # model never emitted a valid tool call.
        assert capture.has_tool("getWeather"), (
            f"Router picked getWeather but force-invoke safety net didn't fire. "
            f"Tools called: {capture.tool_names()}"
        )

        # None of the gemma-specific leak tokens may reach the user-visible
        # reply. If any do, the parser failed AND scrubbing failed.
        lowered = (response or "").lower()
        for leak in ("tool_code", "google_search.search", "<unused"):
            assert leak not in lowered, (
                f"Gemma tool_code leak {leak!r} reached user-visible reply: "
                f"{response!r}"
            )


class TestForceInvocationShortConfabulation:
    """Fix 2: model produces a short confabulation without any tool call.

    Field capture (2026-04-20): user asked "tell me about the movie
    possessor". Router picked webSearch. Model replied:

        The movie Possessor is a science fiction film from 2006 directed
        by Brandon Cronenberg.

    No tool call. Wrong year (Possessor is 2020). Answering a named-entity
    query without searching is the exact failure mode this safety net
    covers.
    """

    def test_short_confab_triggers_force_invoke_webSearch(
        self, mock_config, db, dialogue_memory,
    ):
        confab = (
            "The movie Possessor is a 2006 film directed by Brandon Cronenberg."
        )
        response, capture = _run_engine_with_fixed_router(
            user_query="tell me about the movie possessor",
            model="gemma4:e2b",
            router_pick=["webSearch", "stop"],
            chat_responses=[
                _msg(confab),
                _msg("Possessor is a 2020 film by Brandon Cronenberg..."),
            ],
            mock_config=mock_config, db=db, dialogue_memory=dialogue_memory,
            tool_reply="Search result: Possessor is a 2020 film...",
        )

        assert capture.has_tool("webSearch"), (
            f"Router picked webSearch but force-invoke didn't fire on a "
            f"short confabulation. Tools called: {capture.tool_names()}"
        )
        # webSearch requires search_query — must have been derived from the
        # user's turn, not left empty.
        ws_args = next(c["args"] for c in capture.calls if c["name"] == "webSearch")
        assert ws_args.get("search_query"), (
            f"Forced webSearch invocation produced no search_query: {ws_args}"
        )


class TestForceInvocationGating:
    """Negative cases: the safety net MUST NOT fire when it would be wrong."""

    def test_large_models_are_not_force_invoked(
        self, mock_config, db, dialogue_memory,
    ):
        """On large models a short reply is more likely a considered answer.

        Force-invoke would override genuine reasoning. Gate on model size.
        """
        response, capture = _run_engine_with_fixed_router(
            user_query="tell me about the movie possessor",
            model="gpt-oss:20b",
            router_pick=["webSearch", "stop"],
            chat_responses=[
                _msg("I'd rather not speculate about that film."),
            ],
            mock_config=mock_config, db=db, dialogue_memory=dialogue_memory,
        )
        assert not capture.has_tool("webSearch"), (
            f"Force-invoke fired on a LARGE model — this should be gated off. "
            f"Tools called: {capture.tool_names()}"
        )

    def test_force_skipped_when_router_picked_multiple_real_tools(
        self, mock_config, db, dialogue_memory,
    ):
        """If the router offered multiple real tools the pick is ambiguous.

        Picking one arbitrarily would be worse than shipping the model's
        reply as-is, so skip.
        """
        response, capture = _run_engine_with_fixed_router(
            user_query="what should I do right now",
            model="gemma4:e2b",
            router_pick=["webSearch", "getWeather", "stop"],
            chat_responses=[
                _msg("Maybe take a walk."),
            ],
            mock_config=mock_config, db=db, dialogue_memory=dialogue_memory,
        )
        assert not capture.has_tool("webSearch"), (
            f"Force-invoke fired when router offered multiple real tools — "
            f"that's ambiguous. Tools called: {capture.tool_names()}"
        )

    def test_force_skipped_when_reply_is_substantial(
        self, mock_config, db, dialogue_memory,
    ):
        """A long, substantive reply is more likely a genuine answer than a
        short confabulation. Don't steamroll it."""
        long_reply = (
            "Here's a detailed explanation of the movie Possessor. " * 30
        )  # ~1700 chars — well above the confab threshold
        response, capture = _run_engine_with_fixed_router(
            user_query="tell me about possessor",
            model="gemma4:e2b",
            router_pick=["webSearch", "stop"],
            chat_responses=[_msg(long_reply)],
            mock_config=mock_config, db=db, dialogue_memory=dialogue_memory,
        )
        assert not capture.has_tool("webSearch"), (
            f"Force-invoke fired on a substantive reply. "
            f"Tools called: {capture.tool_names()}"
        )
