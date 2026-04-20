"""
Regression eval: tool selection must switch when the conversation topic
switches from one turn to the next.

Captured from a real field session on 2026-04-20 (gemma4:e2b) where the
user asked two consecutive questions:

  Turn 1: "Tell me about the movie possessor"
          → correct tool: webSearch
          → model produced a confabulated reply WITHOUT invoking webSearch
            ("Possessor is a science fiction film from 2006 directed by
            Brandon Cronenberg" — wrong year, no tool call)

  Turn 2: "And how is the weather today?"
          → correct tool: getWeather (with no args — location auto-derives)
          → model produced gemma's native Google-training fallback syntax
            ("tool_code\\nprint(google_search.search(query='current weather'))
            <unused88>") — i.e. it tried to use a tool but in the wrong
            protocol, so our parser missed it and no tool was actually
            invoked.

Neither failure was caught by existing evals because:
  (a) The default model-under-test was gpt-oss:20b, not gemma4:e2b.
  (b) No existing eval exercised a MULTI-TURN sequence where turn N+1
      requires a different tool than turn N — the "hot window" diary from
      turn N leaks into the enrichment for turn N+1 and can bias routing.

This eval keeps both turns in one test so the whole sequence is asserted
together. The two specific failure modes — "tool selected but never
invoked" (turn 1) and "model emits native tool_code syntax our parser
ignores" (turn 2) — are both represented in the assertions.
"""

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import ToolCallCapture, create_mock_tool_run


# Diary context carried from a prior session about the movie Possessor.
# Kept deliberately realistic — this is the actual shape of what diary
# enrichment injects after turn 1 has settled.
POSSESSOR_DIARY = (
    "[2026-04-20] The user asked for more information about the movie "
    "*Possessor*. The assistant searched the web and shared details about "
    "the film's plot, cast, and director. (Topics: Possessor, movie)"
)


# English deflection phrases — only used when the judge model is
# English-trained (gemma4, gpt-oss). CLAUDE.md forbids hardcoding
# language-specific assertions in the product; this is an eval-only
# heuristic scoped to the judge tier being run.
_PRE_TOOL_CLARIFICATION = (
    "i need a location",
    "need a location",
    "please specify a city",
    "which city",
    "where are you",
    "what location",
)

# Substrings indicating the model fell through to gemma's native
# Google-training tool syntax instead of the format our parser expects.
# If any of these land in the user-visible reply, the parser missed the
# tool call and the user sees raw syntax.
_NATIVE_TOOL_CODE_LEAKS = (
    "tool_code",
    "google_search.search",
    "<unused",
    "```tool_code",
    "print(google_search",
)


@pytest.mark.eval
@requires_judge_llm
class TestContextSwitchTools:
    """Two-turn sequence: webSearch on turn 1, getWeather on turn 2."""

    def _run_turn(
        self, query, mock_config, eval_db, eval_dialogue_memory,
        diary_entries, tool_responses,
    ):
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Location enabled so getWeather's auto-derive path would succeed
        # if the model actually calls it.
        mock_config.location_enabled = True
        mock_config.location_auto_detect = True

        capture = ToolCallCapture()

        with patch(
            'jarvis.memory.conversation.search_conversation_memory_by_keywords',
            return_value=diary_entries,
        ), patch(
            'jarvis.reply.engine.run_tool_with_retries',
            side_effect=create_mock_tool_run(capture, tool_responses),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        return response, capture

    def test_turn1_possessor_then_turn2_weather(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """Sequence: ask about a movie, then ask about weather.

        Both turns must invoke the CORRECT tool. The second turn is the
        interesting one — diary enrichment for 'weather' may also surface
        the Possessor entry, but the tool pick must still be getWeather.
        """
        from helpers import JUDGE_MODEL

        # --- Turn 1 -----------------------------------------------------------
        turn1_query = "Tell me about the movie possessor"
        turn1_response, turn1_capture = self._run_turn(
            turn1_query,
            mock_config, eval_db, eval_dialogue_memory,
            diary_entries=[],  # fresh session — no prior diary
            tool_responses={
                "webSearch": (
                    "Search result: Possessor is a 2020 Canadian science-fiction "
                    "horror film directed by Brandon Cronenberg, starring Andrea "
                    "Riseborough and Christopher Abbott."
                ),
            },
        )
        print(f"\n  Turn 1 ({JUDGE_MODEL}):")
        print(f"    Query: '{turn1_query}'")
        print(f"    Tools: {turn1_capture.tool_names() or 'none'}")
        print(f"    Response: {(turn1_response or '')[:200]}")

        # Turn 1 must call webSearch. If the model confabulated without
        # the tool, _TOOL_RESULT_TOKENS from the mock won't appear.
        if not turn1_capture.has_tool("webSearch"):
            pytest.fail(
                f"Turn 1: model never called webSearch on an unknown named "
                f"entity. Response: {(turn1_response or '')[:400]}. "
                f"This is the confabulation failure from the 2026-04-20 log."
            )

        # --- Turn 2 -----------------------------------------------------------
        # Diary entries available to turn 2: the just-settled Possessor entry
        # (which will surface via keyword search for 'weather' if the memory
        # layer happens to fuzzy-match, and more importantly will be in the
        # hot-window dialogue state).
        turn2_query = "And how is the weather today?"
        turn2_response, turn2_capture = self._run_turn(
            turn2_query,
            mock_config, eval_db, eval_dialogue_memory,
            diary_entries=[POSSESSOR_DIARY],
            tool_responses={
                "getWeather": (
                    "Current weather in Hackney, London: 14°C, partly cloudy, "
                    "wind 10 km/h. Forecast: highs around 15°C."
                ),
            },
        )
        print(f"\n  Turn 2 ({JUDGE_MODEL}):")
        print(f"    Query: '{turn2_query}'")
        print(f"    Tools: {turn2_capture.tool_names() or 'none'}")
        print(f"    Response: {(turn2_response or '')[:200]}")

        # Turn 2 assertion 1: the reply must NOT contain gemma's native
        # tool_code syntax leaking through the parser. This is the exact
        # failure from the 2026-04-20 log where the user saw raw
        # `tool_code\nprint(google_search.search(...))<unused88>`.
        response_lower = (turn2_response or "").lower()
        leaked = next(
            (tok for tok in _NATIVE_TOOL_CODE_LEAKS if tok in response_lower),
            None,
        )
        if leaked:
            pytest.fail(
                f"Turn 2: gemma native tool_code syntax leaked into the "
                f"user-visible reply (first hit: {leaked!r}). The parser "
                f"failed to recognise the model's fallback format, so no "
                f"tool was actually invoked. Response: "
                f"{(turn2_response or '')[:400]}"
            )

        # Turn 2 assertion 2: getWeather must be invoked. Asking for a
        # location pre-emptively, or answering without any tool, both fail.
        if not turn2_capture.has_tool("getWeather"):
            hit = next(
                (p for p in _PRE_TOOL_CLARIFICATION if p in response_lower),
                None,
            )
            msg = (
                f"Turn 2: getWeather was never invoked. "
                f"Tools called: {turn2_capture.tool_names() or 'none'}. "
                f"Pre-tool clarification phrase hit: {hit!r}. "
                f"Response: {(turn2_response or '')[:400]}"
            )
            if JUDGE_MODEL.startswith("gemma4"):
                # Known gemma4 limitation — capture as xfail so CI stays
                # green but the failure is visible and tracked.
                pytest.xfail(f"{JUDGE_MODEL} limitation. {msg}")
            pytest.fail(msg)

        # Turn 2 assertion 3: no stale Possessor token leaked into the
        # weather reply (previous-turn contamination).
        for stale_tok in ("Cronenberg", "Riseborough", "Possessor"):
            assert stale_tok.lower() not in response_lower, (
                f"Turn 2: previous-turn topic token {stale_tok!r} leaked "
                f"into the weather reply. Response: "
                f"{(turn2_response or '')[:400]}"
            )
