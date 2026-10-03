"""Live local classifier accuracy, independent of chat generation or fallback.

Run with EVAL_DECISION_BASE_URL pointing at a resident self-hosted classifier.
An unavailable service fails when explicitly enabled, rather than counting as
a passing routing evaluation.
"""

import os

import pytest

from jarvis.tools.decision import classify_tools


CASES = [
    ("What's the weather in London tomorrow?", None, {"getWeather"}),
    ("Search the web for Python tutorials", None, {"webSearch"}),
    ("Log a chicken salad for lunch", None, {"logMeal"}),
    ("What did I eat yesterday?", None, {"fetchMeals"}),
    ("Hello!", None, set()),
    ("What's today's date?", "Current local time: 2026-10-03 12:00 Europe/London", set()),
    ("I'm in London", "Recent dialogue (short-term memory):\nuser: What's the weather?\nassistant: Which city?", {"getWeather"}),
    ("Londra'da yarın hava nasıl?", None, {"getWeather"}),
    ("¿Qué tiempo hará mañana en Madrid?", None, {"getWeather"}),
    ("明日の東京の天気は？", None, {"getWeather"}),
    ("Log my lunch and check tomorrow's forecast", None, {"logMeal", "getWeather"}),
]

DESCRIPTIONS = {
    "getWeather": "Get current weather or a forecast for a location.",
    "webSearch": "Search the web for information, news or research.",
    "logMeal": "Record what the user ate in their meal diary.",
    "fetchMeals": "Read meals recorded in the user's meal diary.",
}


@pytest.mark.eval
@pytest.mark.skipif(not os.getenv("EVAL_DECISION_BASE_URL"), reason="Local decision eval not enabled")
@pytest.mark.parametrize("query,context,expected", CASES)
def test_local_classifier_routes_without_generated_text(query, context, expected):
    selected = classify_tools(
        query, DESCRIPTIONS,
        base_url=os.environ["EVAL_DECISION_BASE_URL"],
        model=os.environ.get("EVAL_DECISION_MODEL", "multilingual"),
        threshold=0.7, timeout_sec=30, context_hint=context, max_selected=5,
    )
    assert set(selected) == expected
