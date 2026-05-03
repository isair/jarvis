"""
End-to-end eval — single-turn flow where the user's location lives in the
User branch of the knowledge graph (warm profile). The warm profile is
always-loaded into the system prompt, so the chat model and planner can
ground ``getWeather`` on it without a ``searchMemory`` step.

This stresses the warm-profile-injection path. It complements:
  - ``evals/test_followup_supplies_missing_tool_arg.py`` (hot-window
    carry-over, two-turn).
  - ``evals/test_diary_supplies_missing_tool_arg.py`` (diary recall via
    planner-emitted ``searchMemory``).

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh graph_supplies_missing_tool_arg
"""

from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    JUDGE_MODEL,
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
                    success=True,
                    reply_text="No location is configured for the user.",
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
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Geoip disabled — the only way the model gets a location is from
        # the warm profile loaded out of the graph.
        mock_config.location_enabled = False

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
                db=eval_db, cfg=mock_config, tts=None,
                text="how's the weather, Jarvis?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Graph Supplies Missing Tool Arg ({JUDGE_MODEL}):")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Response: {(response or '')[:300]}")

        assert_not_fallback_reply(response, context="warm-profile")

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
