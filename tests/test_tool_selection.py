"""Tests for tool selection strategies."""

import pytest
from unittest.mock import patch, Mock

from jarvis.tools.selection import (
    select_tools,
    _tokenise,
    _build_tool_keywords,
    _ALWAYS_INCLUDED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTool:
    """Minimal tool stand-in for testing."""
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description


class FakeToolSpec:
    """Minimal ToolSpec stand-in for testing."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


def _builtin():
    """Return a small set of fake builtin tools."""
    return {
        "webSearch": FakeTool("webSearch", "Search the web using DuckDuckGo for current information, news, or general queries."),
        "getWeather": FakeTool("getWeather", "Get current weather conditions."),
        "logMeal": FakeTool("logMeal", "Log a single meal when the user mentions eating or drinking something."),
        "fetchMeals": FakeTool("fetchMeals", "Retrieve meals from the database for a given time range."),
        "screenshot": FakeTool("screenshot", "Capture a selected screen region and OCR the text."),
        "localFiles": FakeTool("localFiles", "Safely read, write, list, append, or delete files within your home directory."),
        "stop": FakeTool("stop", "End the current conversation."),
    }


def _mcp():
    """Return a small set of fake MCP tools."""
    return {
        "homeassistant__turn_on": FakeToolSpec("homeassistant__turn_on", "Turn on a smart home device."),
    }


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenise:

    @pytest.mark.unit
    def test_basic_tokenise(self):
        tokens = _tokenise("What's the weather in London?")
        assert "weather" in tokens
        assert "london" in tokens
        # Stop-words removed
        assert "the" not in tokens
        assert "in" not in tokens

    @pytest.mark.unit
    def test_empty_string(self):
        assert _tokenise("") == []


class TestBuildToolKeywords:

    @pytest.mark.unit
    def test_camel_case_split(self):
        kw = _build_tool_keywords("fetchWebPage", "Fetch content from a URL.")
        assert "fetch" in kw
        assert "web" in kw
        assert "page" in kw

    @pytest.mark.unit
    def test_description_tokens(self):
        kw = _build_tool_keywords("getWeather", "Get current weather conditions.")
        assert "weather" in kw
        assert "conditions" in kw


# ---------------------------------------------------------------------------
# Strategy: all
# ---------------------------------------------------------------------------

class TestAllStrategy:

    @pytest.mark.unit
    def test_returns_everything(self):
        result = select_tools("hello", _builtin(), _mcp(), strategy="all")
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_unknown_strategy_falls_back_to_all(self):
        result = select_tools("hello", _builtin(), _mcp(), strategy="banana")
        assert len(result) == len(_builtin()) + len(_mcp())


# ---------------------------------------------------------------------------
# Strategy: keyword
# ---------------------------------------------------------------------------

class TestKeywordStrategy:

    @pytest.mark.unit
    def test_weather_query_selects_weather_tool(self):
        result = select_tools("what's the weather in London", _builtin(), {}, strategy="keyword")
        assert "getWeather" in result

    @pytest.mark.unit
    def test_weather_query_excludes_irrelevant(self):
        result = select_tools("what's the weather in London", _builtin(), {}, strategy="keyword")
        assert "logMeal" not in result
        assert "screenshot" not in result

    @pytest.mark.unit
    def test_meal_query_selects_meal_tools(self):
        result = select_tools("what did I eat yesterday", _builtin(), {}, strategy="keyword")
        assert "fetchMeals" in result or "logMeal" in result

    @pytest.mark.unit
    def test_search_query_selects_web_search(self):
        result = select_tools("search for python tutorials", _builtin(), {}, strategy="keyword")
        assert "webSearch" in result

    @pytest.mark.unit
    def test_stop_always_included(self):
        result = select_tools("what's the weather", _builtin(), {}, strategy="keyword")
        assert "stop" in result

    @pytest.mark.unit
    def test_vague_query_falls_back_to_all(self):
        """A query with no keyword matches should return all tools."""
        result = select_tools("hmm", _builtin(), {}, strategy="keyword")
        assert len(result) == len(_builtin())

    @pytest.mark.unit
    def test_mcp_tools_included(self):
        result = select_tools("turn on the lights", _builtin(), _mcp(), strategy="keyword")
        assert "homeassistant__turn_on" in result

    @pytest.mark.unit
    def test_file_query_selects_local_files(self):
        result = select_tools("read the config file", _builtin(), {}, strategy="keyword")
        assert "localFiles" in result


# ---------------------------------------------------------------------------
# Strategy: llm
# ---------------------------------------------------------------------------

class TestLLMStrategy:

    @pytest.mark.unit
    def test_parses_comma_separated_response(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            return "webSearch, getWeather"

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "what's the weather",
                _builtin(), {},
                strategy="llm",
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "stop" in result  # always included

    @pytest.mark.unit
    def test_none_response_returns_only_mandatory(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            return "none"

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "hello",
                _builtin(), {},
                strategy="llm",
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert result == ["stop"]

    @pytest.mark.unit
    def test_llm_failure_falls_back_to_all(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            raise TimeoutError("LLM timed out")

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "anything",
                _builtin(), _mcp(),
                strategy="llm",
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_empty_response_falls_back_to_all(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            return ""

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "anything",
                _builtin(), {},
                strategy="llm",
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert len(result) == len(_builtin())

    @pytest.mark.unit
    def test_ignores_hallucinated_tool_names(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            return "webSearch, nonExistentTool, getWeather"

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "search and weather",
                _builtin(), {},
                strategy="llm",
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "nonExistentTool" not in result
