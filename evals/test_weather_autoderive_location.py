"""
Regression eval: getWeather must be called without asking for location.

Field failures captured 2026-04-20 and 2026-04-21:

  - 2026-04-20 "what's the weather this week": the LLM replied "What location
    are you asking about?" without calling the tool.
  - 2026-04-21 "How's the weather, Jarvis?": with ten prior diary entries
    about weather loaded (~890 char digest), gemma produced malformed
    output and the engine shipped the canned fallback "I had trouble
    understanding that request." The tool was never invoked.

The tool's description explicitly states it uses the user's current location
when none is given. This eval asserts the model respects that contract
instead of asking for an argument the tool already handles — AND that a
warm memory state (the normal production condition) doesn't tip gemma into
scaffolding mode where the malformed guard silently eats the turn.

Two parametrised variants cover:
  - ``cold-memory``: fresh dialogue memory + empty diary (old behaviour).
  - ``warm-memory``: ten prior weather-related diary summaries, matching
    the field log at 2026-04-21. This is the state that actually ships
    to users and was previously never exercised in evals.

Historical note: this eval used to ``pytest.xfail`` every gemma failure
as "flakiness", which meant the exact field regressions above were
recorded as expected-failures rather than real failures. The xfail
escape hatches have been removed — if gemma breaks here, we want CI
to shout.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh weather_autoderive
"""

from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    create_mock_tool_run,
    seed_diary_summaries,
)


# Phrases that indicate the model deflected to asking for location instead of
# calling the tool. These are English-language signals for the gpt-oss/gemma
# judge models we evaluate against. CLAUDE.md forbids hardcoded language
# patterns in production code paths (the assistant supports arbitrary
# languages), but eval assertions against a specific English-speaking judge
# model are scoped to that judge and don't leak into the product.
_LOCATION_CLARIFICATION_PHRASES = (
    "what location",
    "which location",
    "where are you",
    "your location",
    "specify a location",
    "specify the location",
    "tell me your location",
    "tell me the location",
    "what city",
    "which city",
    "where do you want",
)


# Ten dated summaries approximating the field-log state where the user has
# asked about weather repeatedly over a fortnight. The digest built from
# these is ~800-900 chars, matching the production shape that tipped
# gemma into malformed output.
_WARM_WEATHER_DIARY = [
    ("2026-04-07", "The user asked whether it would rain in Hackney in the evening; the assistant provided the forecast showing light rain after 18:00."),
    ("2026-04-08", "The user inquired about the weekend weather; the assistant reported dry conditions with highs of 15°C."),
    ("2026-04-10", "The user requested a weather check for Tuesday; the assistant replied with partly cloudy 13°C."),
    ("2026-04-11", "The user asked about the weather for tomorrow; the assistant returned cool and overcast conditions."),
    ("2026-04-13", "The user asked about this afternoon's weather; the assistant reported bright sun and mild temperatures."),
    ("2026-04-15", "The user inquired about the weather for tomorrow; since no location was supplied, the assistant used Hackney and returned the forecast."),
    ("2026-04-16", "The user asked what the weather was doing; the assistant reported intermittent rain and temperatures around 11°C."),
    ("2026-04-17", "The user inquired about the current weather; the assistant provided a snapshot showing overcast and mild."),
    ("2026-04-18", "The user asked about the weekend outlook; the assistant reported mixed conditions with rain Sunday afternoon."),
    ("2026-04-20", "The user asked about the weather this week; the assistant delivered a multi-day forecast for Hackney."),
]


def _run_weather_query(mock_config, eval_db, eval_dialogue_memory, query: str):
    from helpers import JUDGE_MODEL
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.location_enabled = True

    capture = ToolCallCapture()

    weather_payload = (
        "Weather for Hackney, London, UK:\n"
        "Today: 14°C, partly cloudy. High 16°C, low 9°C.\n"
        "This week: mixed cloud, some rain Thursday, sunny Saturday."
    )

    with patch(
        'jarvis.utils.location.get_location_info',
        return_value={"city": "Hackney", "region": "England", "country": "UK"},
    ), patch(
        'jarvis.reply.engine.run_tool_with_retries',
        side_effect=create_mock_tool_run(capture, {
            "getWeather": weather_payload,
        }),
    ):
        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None,
            text=query, dialogue_memory=eval_dialogue_memory,
        )
    return capture, response


@pytest.mark.eval
@requires_judge_llm
class TestWeatherAutoDerivesLocation:
    """Regression guard: getWeather must be called without nagging for location,
    even under warm memory state."""

    @pytest.mark.parametrize(
        "variant,query",
        [
            ("cold-memory-week-forecast", "what's the weather this week"),
            ("cold-memory-short-query", "how's the weather"),
            ("warm-memory-short-query", "how's the weather"),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_weather_query_calls_tool_and_grounds_reply(
        self, mock_config, eval_db, eval_dialogue_memory, variant, query,
    ):
        from helpers import JUDGE_MODEL

        if variant.startswith("warm-memory"):
            seed_diary_summaries(eval_db, _WARM_WEATHER_DIARY)

        capture, response = _run_weather_query(
            mock_config, eval_db, eval_dialogue_memory, query,
        )

        print(f"\n  Weather Auto-Derive [{variant}] ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:300]}")

        # Shield against the engine silently shipping the "I had trouble
        # understanding that request" canned fallback — that's the malformed
        # guard firing, which masks the real model failure from eval
        # assertions that only check tool calls.
        assert_not_fallback_reply(response, context=variant)

        lowered = (response or "").lower()
        asked_for_location = next(
            (p for p in _LOCATION_CLARIFICATION_PHRASES if p in lowered), None,
        )

        assert capture.has_tool("getWeather"), (
            f"[{variant}] Model failed to call getWeather despite the "
            f"tool's description stating it uses the user's current "
            f"location when none is given, and the user's location being "
            f"injected into the system prompt. "
            f"Tools called: {capture.tool_names() or 'none'}. "
            f"Location-clarification phrase hit: {asked_for_location!r}. "
            f"Response: {(response or '')[:400]}"
        )

        assert asked_for_location is None, (
            f"[{variant}] Model called getWeather but also asked the user "
            f"for a location — that's the deflection pattern the prompt "
            f"clause is meant to prevent. "
            f"Phrase hit: {asked_for_location!r}. "
            f"Response: {(response or '')[:400]}"
        )
