"""
End-to-end eval — two-turn flow where the user supplies a missing tool
argument on the second turn.

Field trace (2026-05-03, gemma4:e2b):

  Turn 1: "how's the weather tomorrow Jarvis?"
    → location not configured → getWeather reports "no location set"
    → assistant asks the user for a location.

  Turn 2: "I'm in London"
    → small router picks webSearch (not getWeather), planner does
      `webSearch query='weather in london tomorrow'`, DDG bot-challenges,
      Wikipedia fallback matches "Edge of Tomorrow" (the 2014 Tom Cruise
      film) on the keyword "tomorrow", and the assistant parrots the film
      summary as the weather answer.

The fix lives at the engine level: when the previous assistant turn
invoked a tool and the current user query is a short follow-up
(≤ ~80 chars), the previous tool name is unioned back into the allow-list
so the chat model can continue the original tool chain with the new info.

This eval drives the full reply engine over both turns and asserts that
``getWeather`` is invoked twice — once with empty args (turn 1) and once
with ``location='London'`` (turn 2) — and that the final reply mentions
the London forecast, not "Edge of Tomorrow".

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh followup_supplies_missing_tool_arg
"""

from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    JUDGE_MODEL,
)


_LONDON_FORECAST = (
    "Weather for London, UK:\n"
    "Today: 15°C, partly cloudy. High 17°C, low 10°C.\n"
    "Tomorrow: 14°C, light rain, high 16°C, low 9°C."
)


def _make_get_weather_runner(capture: ToolCallCapture):
    """Mock for ``run_tool_with_retries`` that responds to getWeather based
    on the location argument.

    Empty args → ``success=False`` ("could not auto-detect location") to
    match the real getWeather behaviour and stamp ``tool_failed=True`` on
    the recorded tool turn (turn 1 shape).
    ``location='London'`` (or any non-empty location) → ``success=True``
    plus the canned forecast.
    Everything else falls through to ``success=True`` "OK".
    """
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "getWeather":
            location = ((tool_args or {}).get("location") or "").strip()
            if not location:
                return ToolExecutionResult(
                    success=False,
                    reply_text=(
                        "I couldn't auto-detect your location. Please "
                        "tell me which city to check the weather for."
                    ),
                )
            return ToolExecutionResult(
                success=True,
                reply_text=_LONDON_FORECAST,
            )
        # If the model misroutes to webSearch we want to make damn sure we
        # don't accidentally satisfy the assertion via a confabulated
        # success — return something the model cannot honestly turn into
        # a London forecast.
        if tool_name == "webSearch":
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "UNTRUSTED WEB EXTRACT:\n"
                    "Edge of Tomorrow is a 2014 American science fiction "
                    "action film directed by Doug Liman, starring Tom Cruise."
                ),
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestFollowupSuppliesMissingToolArg:
    """End-to-end regression for the engine-level tool carry-over guard."""

    def test_short_followup_continues_previous_tool_chain(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Geoip disabled — the only way the model gets a location is
        # from the user supplying one on turn 2.
        mock_config.location_enabled = False

        capture = ToolCallCapture()

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_get_weather_runner(capture),
        ):
            turn1 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="how's the weather tomorrow Jarvis?",
                dialogue_memory=eval_dialogue_memory,
            )
            turn2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="I'm in London",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Followup Carry-over ({JUDGE_MODEL}):")
        print(f"  Turn 1 reply: {(turn1 or '')[:200]}")
        print(f"  Turn 2 reply: {(turn2 or '')[:200]}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")

        assert_not_fallback_reply(turn1, context="turn-1")
        assert_not_fallback_reply(turn2, context="turn-2")

        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
        assert len(weather_calls) >= 2, (
            "Expected getWeather to be invoked at least twice (once with "
            "empty args on turn 1, once with location='London' on turn 2). "
            f"Tools observed: {capture.tool_names()}. Calls: {capture.calls}"
        )

        # Turn-2 call must carry the location the user supplied.
        london_calls = [
            c for c in weather_calls
            if "london" in (c["args"].get("location") or "").lower()
        ]
        assert london_calls, (
            "getWeather was never re-invoked with location='London' on "
            "turn 2 — the carry-over guard did not preserve the previous "
            f"tool's place in the allow-list. All getWeather calls: "
            f"{[c['args'] for c in weather_calls]}"
        )

        # webSearch must NOT have been the path — that's the field-trace
        # failure mode (Edge of Tomorrow). If it fired anyway, the user
        # answer must still be about London weather, not the film.
        turn2_lower = (turn2 or "").lower()
        assert "edge of tomorrow" not in turn2_lower, (
            "Reply parroted the Wikipedia fallback for 'Edge of Tomorrow'. "
            f"Reply: {(turn2 or '')[:400]}"
        )
        assert "london" in turn2_lower, (
            "Turn-2 reply does not mention London weather. "
            f"Reply: {(turn2 or '')[:400]}"
        )
