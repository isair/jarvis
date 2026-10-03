"""A supplied location continues an incomplete weather request.

The first turn has no location and the second supplies London. Both turns
must return usable replies, and the final answer must use the tool forecast.
An unrelated film search result cannot substitute for weather evidence.
"""

from unittest.mock import patch

import pytest

from evals.memory_tool_grounding import assert_usable_answer, assert_forecast_reply

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    JUDGE_MODEL,
    voice_config,
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
    ``location='London'`` → ``success=True``
    plus the canned forecast.
    Other locations fail. Non-weather tools return their own fixture data.
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
            if "london" not in location.casefold():
                return ToolExecutionResult(
                    success=False,
                    reply_text="This fixture has no weather for that location.",
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
        self, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        cfg = voice_config()
        # Geoip disabled — the only way the model gets a location is
        # from the user supplying one on turn 2.
        cfg.location_enabled = False

        capture = ToolCallCapture()

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_get_weather_runner(capture),
        ):
            turn1 = run_reply_engine(
                db=eval_db, cfg=cfg, tts=None,
                text="how's the weather tomorrow Jarvis?",
                dialogue_memory=eval_dialogue_memory,
            )
            turn2 = run_reply_engine(
                db=eval_db, cfg=cfg, tts=None,
                text="I'm in London",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  🔁 Follow-up Carry-over ({JUDGE_MODEL}):")
        print(f"  💬 Turn 1 reply: {(turn1 or '')[:200]}")
        print(f"  💬 Turn 2 reply: {(turn2 or '')[:200]}")
        print(f"  🛠️ Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    🔧 {c['name']}({c['args']})")

        assert_usable_answer(turn1, "turn-1")
        assert_forecast_reply(turn2, _LONDON_FORECAST, "turn-2")

        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
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
