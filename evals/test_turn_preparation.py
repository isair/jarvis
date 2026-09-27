"""Live quality checks for one-call routing, planning and memory decisions."""
import pytest

from conftest import requires_judge_llm
from helpers import MockConfig, JUDGE_MODEL

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize("query,expected_tools,memory", [
    ("Hello", set(), False),
    ("Compare the weather in Paris and London", {"getWeather"}, False),
    ("What did I tell you about my old address last month?", set(), True),
    ("Geçen ay sana eski adresim hakkında ne söyledim?", set(), True),
    ("Compare la météo à Paris et à Londres", {"getWeather"}, False),
])
def test_combined_preparation_quality(query, expected_tools, memory):
    from jarvis.reply.preparation import prepare_turn
    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    decision = prepare_turn(
        cfg=cfg, query=query, dialogue_context="",
        tools=[("getWeather", "Current weather and forecast for a named location"),
               ("webSearch", "Search the public web"),
               ("logMeal", "Save a meal the user ate")],
        context_hint="Current date: 2026-09-27 UTC", timeout_sec=30,
    )
    assert decision is not None, "Preparation must produce a valid decision"
    assert set(decision.tools) == expected_tools
    assert decision.needs_memory is memory
    if memory:
        assert decision.search_params["keywords"]
    if expected_tools:
        assert any("Paris" in step for step in decision.steps)
        assert any("London" in step or "Londres" in step for step in decision.steps)
