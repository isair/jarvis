"""Warm-profile location facts ground weather tool arguments and answers.

GeoIP is disabled. The User branch supplies Edinburgh without an explicit
memory-search step. Model inference uses the selected evaluation transport.
"""

from unittest.mock import patch

import pytest

from evals.memory_tool_grounding import assert_forecast_reply

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    JUDGE_MODEL,
    voice_config,
)


_EDINBURGH_FORECAST = (
    "Weather for Edinburgh, UK:\n"
    "Today: 11°C, partly cloudy. High 13°C, low 7°C.\n"
    "Tomorrow: 12°C, light rain, high 14°C, low 8°C."
)


def _make_runner(capture: ToolCallCapture):
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
            if "edinburgh" not in location.casefold():
                return ToolExecutionResult(
                    success=False,
                    reply_text="This fixture has no weather for that location.",
                )
            return ToolExecutionResult(
                success=True,
                reply_text=_EDINBURGH_FORECAST,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestGraphSuppliesMissingToolArg:
    """Warm-profile injection path: a User-branch fact ("lives in
    Edinburgh") is always loaded into the system prompt, so the chat
    model can supply it as the location argument without an extra
    memory search."""

    def test_warm_profile_user_fact_grounds_get_weather_call(
        self, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        cfg = voice_config()
        # Geoip disabled — the only way the model gets a location is from
        # the warm profile loaded out of the graph.
        cfg.location_enabled = False

        capture = ToolCallCapture()

        # Inject a User-branch fact directly into the warm-profile builder
        # rather than seeding the SQLite-backed graph store. The warm-
        # profile path the engine relies on is `build_warm_profile` →
        # `format_warm_profile_block`; seeding via the public API replays
        # the production shape without depending on graph-mutation
        # listeners or branch-root bootstrapping in the test DB.
        warm_profile = {
            "user": "The user lives in Edinburgh.",
            "directives": "",
        }

        with patch(
            "jarvis.memory.graph_ops.build_warm_profile",
            return_value=warm_profile,
        ), patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_runner(capture),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=cfg, tts=None,
                text="how's the weather, Jarvis?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  🧠 Graph Supplies Missing Tool Arg ({JUDGE_MODEL}):")
        print(f"  🛠️ Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    🔧 {c['name']}({c['args']})")
        print(f"  💬 Response: {(response or '')[:300]}")

        assert_forecast_reply(response, _EDINBURGH_FORECAST, "warm-profile")

        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
        edinburgh_calls = [
            c for c in weather_calls
            if "edinburgh" in (c["args"].get("location") or "").lower()
        ]
        assert edinburgh_calls, (
            "getWeather was not invoked with location='Edinburgh' even "
            "though the warm profile names Edinburgh as the user's home. "
            "The chat model must use always-loaded user facts as tool "
            "arguments without an explicit prompt to do so. "
            f"All getWeather calls: {[c['args'] for c in weather_calls]}. "
            f"Tools observed: {capture.tool_names()}. "
            f"Response: {(response or '')[:400]}"
        )

        response_lower = (response or "").lower()
        assert "edinburgh" in response_lower, (
            "Reply does not mention Edinburgh despite the warm profile "
            f"naming it as the user's location. Response: {(response or '')[:400]}"
        )

        assert "hackney" not in response_lower, (
            "Reply mentions Hackney — the warm profile clearly states "
            "Edinburgh, and geoip is disabled in this test. The model "
            f"leaked a hardcoded default. Response: {(response or '')[:400]}"
        )
