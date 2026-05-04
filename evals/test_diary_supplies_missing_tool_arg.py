"""
End-to-end eval — single-turn flow where the user's location lives only
in the diary from a past conversation. The planner must emit
``searchMemory``, the diary must surface "Manchester", and ``getWeather``
must then be invoked with ``location='Manchester'``.

This stresses the diary-recall path. It complements the carry-over
guard's hot-window path (covered by
``evals/test_followup_supplies_missing_tool_arg.py``) by exercising the
slower long-term-memory path: the user said "I live in Manchester" days
ago, the conversation has lapsed, and now the user asks "how's the
weather, Jarvis?" with no live geoip and nothing in the hot window.

Memory-recall reliability on small models is itself an open failure
mode separate from the tool carry-over guard. If gemma4:e2b consistently
deflects rather than grounding the search, this eval is best read as an
upper-bound regression guard: a green run on a reliable judge model
proves the wiring works, while a red run on a small model is expected
until follow-up memory work lands.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh diary_supplies_missing_tool_arg
"""

from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    seed_diary_summaries,
    JUDGE_MODEL,
)


_DIARY_MANCHESTER = [
    (
        "2026-04-26",
        "The user mentioned they live in Manchester and prefer celsius "
        "for weather queries.",
    ),
]


_MANCHESTER_FORECAST = (
    "Weather for Manchester, UK:\n"
    "Today: 12°C, overcast. High 14°C, low 8°C.\n"
    "Tomorrow: 13°C, light rain, high 15°C, low 9°C."
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
            return ToolExecutionResult(
                success=True,
                reply_text=_MANCHESTER_FORECAST,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestDiarySuppliesMissingToolArg:
    """Diary-recall path: location surfaced from a prior conversation
    grounds the getWeather call without needing the hot window or
    explicit user re-statement."""

    def test_diary_location_grounds_get_weather_call(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Geoip disabled — the only way the model gets a location is from
        # diary recall.
        mock_config.location_enabled = False
        mock_config.memory_enrichment_source = "diary"

        seed_diary_summaries(eval_db, _DIARY_MANCHESTER)

        capture = ToolCallCapture()

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_runner(capture),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="how's the weather, Jarvis?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Diary Supplies Missing Tool Arg ({JUDGE_MODEL}):")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Response: {(response or '')[:300]}")

        assert_not_fallback_reply(response, context="diary-recall")

        # The reply must actually use the recalled location, both at the
        # tool call layer and in the user-facing reply.
        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
        manchester_calls = [
            c for c in weather_calls
            if "manchester" in (c["args"].get("location") or "").lower()
        ]
        assert manchester_calls, (
            "getWeather was not invoked with location='Manchester' even "
            "though the diary contains the user's stated location. The "
            "memory enrichment → tool argument grounding path is broken. "
            f"All getWeather calls: {[c['args'] for c in weather_calls]}. "
            f"Tools observed: {capture.tool_names()}. "
            f"Response: {(response or '')[:400]}"
        )

        response_lower = (response or "").lower()
        assert "manchester" in response_lower, (
            "Reply does not mention Manchester despite the diary stating "
            f"the user lives there. Response: {(response or '')[:400]}"
        )

        # Guard against a hardcoded-default leak: any reply that mentions
        # Hackney here is wrong (Hackney is the test fixture's geoip
        # default, but geoip is disabled in this test).
        assert "hackney" not in response_lower, (
            "Reply mentions Hackney — the diary clearly states Manchester, "
            "and geoip is disabled in this test. The model leaked a "
            f"hardcoded default. Response: {(response or '')[:400]}"
        )
