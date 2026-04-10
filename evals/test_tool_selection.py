"""
Tool Selection Evaluations

Tests that the embedding-based tool selection strategy actually filters tools
meaningfully — a weather query should select weather-related tools, not all tools.

Run: .venv/bin/python -m pytest evals/test_tool_selection.py -v
"""

import pytest

from conftest import requires_judge_llm


# =============================================================================
# Test Data
# =============================================================================

# Queries paired with the tools they MUST include and a maximum tool count.
# The max count ensures the strategy actually filters rather than passing everything.
TOOL_SELECTION_CASES = [
    pytest.param(
        "what's the weather like tomorrow",
        ["getWeather"],
        5,
        id="weather query selects getWeather and few others",
    ),
    pytest.param(
        "what's the weather in London this weekend",
        ["getWeather"],
        5,
        id="location weather query selects getWeather and few others",
    ),
    pytest.param(
        "log that I had a chicken salad for lunch",
        ["logMeal"],
        5,
        id="meal logging selects logMeal and few others",
    ),
    pytest.param(
        "what did I eat yesterday",
        ["fetchMeals"],
        5,
        id="meal recall selects fetchMeals and few others",
    ),
    pytest.param(
        "search the web for Python tutorials",
        ["webSearch"],
        5,
        id="web search query selects webSearch and few others",
    ),
]


@pytest.mark.eval
class TestToolSelectionFiltering:
    """Validates that embedding tool selection meaningfully filters tools."""

    @requires_judge_llm
    @pytest.mark.parametrize("query, must_include, max_tools", TOOL_SELECTION_CASES)
    def test_embedding_selects_relevant_tools(
        self,
        mock_config,
        query,
        must_include,
        max_tools,
    ):
        """Embedding strategy should select relevant tools, not all of them."""
        from jarvis.tools.selection import select_tools, ToolSelectionStrategy
        from jarvis.tools.registry import BUILTIN_TOOLS

        selected = select_tools(
            query=query,
            builtin_tools=BUILTIN_TOOLS,
            mcp_tools={},
            strategy=ToolSelectionStrategy.EMBEDDING,
            llm_base_url=mock_config.ollama_base_url,
            embed_model=mock_config.ollama_embed_model,
            embed_timeout_sec=10.0,
        )

        total_builtin = len(BUILTIN_TOOLS)

        # Must include the expected tools
        for tool in must_include:
            assert tool in selected, (
                f"Expected '{tool}' in selected tools but got: {selected}"
            )

        # Must include 'stop' (always included)
        assert "stop" in selected, f"'stop' should always be included, got: {selected}"

        # Must NOT include everything — that means filtering isn't working
        assert len(selected) <= max_tools, (
            f"Expected at most {max_tools} tools but got {len(selected)}/{total_builtin}: {selected}"
        )

        print(f"  ✅ Selected {len(selected)}/{total_builtin} tools: {selected}")
