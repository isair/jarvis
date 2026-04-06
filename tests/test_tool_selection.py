"""Tests for tool selection strategies."""

import pytest
from unittest.mock import patch

from jarvis.tools.selection import (
    select_tools,
    ToolSelectionStrategy,
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
# Enum
# ---------------------------------------------------------------------------

class TestToolSelectionStrategy:

    @pytest.mark.unit
    def test_enum_values(self):
        assert ToolSelectionStrategy.ALL.value == "all"
        assert ToolSelectionStrategy.KEYWORD.value == "keyword"
        assert ToolSelectionStrategy.EMBEDDING.value == "embedding"
        assert ToolSelectionStrategy.LLM.value == "llm"

    @pytest.mark.unit
    def test_enum_from_string(self):
        assert ToolSelectionStrategy("all") == ToolSelectionStrategy.ALL
        assert ToolSelectionStrategy("keyword") == ToolSelectionStrategy.KEYWORD
        assert ToolSelectionStrategy("embedding") == ToolSelectionStrategy.EMBEDDING
        assert ToolSelectionStrategy("llm") == ToolSelectionStrategy.LLM

    @pytest.mark.unit
    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ToolSelectionStrategy("banana")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenise:

    @pytest.mark.unit
    def test_basic_tokenise(self):
        tokens = _tokenise("What's the weather in London?")
        assert "weather" in tokens
        assert "london" in tokens
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
        result = select_tools("hello", _builtin(), _mcp(), strategy=ToolSelectionStrategy.ALL)
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_default_strategy_is_all(self):
        result = select_tools("hello", _builtin(), _mcp())
        assert len(result) == len(_builtin()) + len(_mcp())


# ---------------------------------------------------------------------------
# Strategy: keyword
# ---------------------------------------------------------------------------

class TestKeywordStrategy:

    @pytest.mark.unit
    def test_weather_query_selects_weather_tool(self):
        result = select_tools("what's the weather in London", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "getWeather" in result

    @pytest.mark.unit
    def test_weather_query_excludes_irrelevant(self):
        result = select_tools("what's the weather in London", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "logMeal" not in result
        assert "screenshot" not in result

    @pytest.mark.unit
    def test_meal_query_selects_meal_tools(self):
        result = select_tools("what did I eat yesterday", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "fetchMeals" in result or "logMeal" in result

    @pytest.mark.unit
    def test_search_query_selects_web_search(self):
        result = select_tools("search for python tutorials", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "webSearch" in result

    @pytest.mark.unit
    def test_stop_always_included(self):
        result = select_tools("what's the weather", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "stop" in result

    @pytest.mark.unit
    def test_vague_query_falls_back_to_all(self):
        result = select_tools("hmm", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert len(result) == len(_builtin())

    @pytest.mark.unit
    def test_mcp_tools_included(self):
        result = select_tools("turn on the lights", _builtin(), _mcp(), strategy=ToolSelectionStrategy.KEYWORD)
        assert "homeassistant__turn_on" in result

    @pytest.mark.unit
    def test_file_query_selects_local_files(self):
        result = select_tools("read the config file", _builtin(), {}, strategy=ToolSelectionStrategy.KEYWORD)
        assert "localFiles" in result


# ---------------------------------------------------------------------------
# Strategy: embedding
# ---------------------------------------------------------------------------

class TestEmbeddingStrategy:

    def _mock_embedding(self, text_to_vec):
        """Return a mock get_embedding that maps text substrings to vectors."""
        def mock_get_embedding(text, base_url, model, timeout_sec=10.0):
            for key, vec in text_to_vec.items():
                if key in text.lower():
                    return vec
            # Default: zero vector
            return [0.0] * 4
        return mock_get_embedding

    @pytest.mark.unit
    def test_selects_similar_tools(self):
        """Weather query should rank getWeather highest."""
        mock_embed = self._mock_embedding({
            "weather": [1.0, 0.0, 0.0, 0.0],      # query + weather tool
            "search": [0.0, 1.0, 0.0, 0.0],
            "meal": [0.0, 0.0, 1.0, 0.0],
            "screen": [0.0, 0.0, 0.0, 1.0],
            "file": [0.1, 0.1, 0.1, 0.1],
            "conversation": [0.1, 0.1, 0.1, 0.1],
        })
        with patch("jarvis.memory.embeddings.get_embedding", side_effect=mock_embed):
            result = select_tools(
                "what's the weather",
                _builtin(), {},
                strategy=ToolSelectionStrategy.EMBEDDING,
                llm_base_url="http://localhost",
                embed_model="nomic-embed-text",
            )
        assert "getWeather" in result

    @pytest.mark.unit
    def test_stop_always_included(self):
        """Stop tool must be present even if not semantically matched."""
        mock_embed = self._mock_embedding({
            "weather": [1.0, 0.0, 0.0, 0.0],
        })
        with patch("jarvis.memory.embeddings.get_embedding", side_effect=mock_embed):
            result = select_tools(
                "what's the weather",
                _builtin(), {},
                strategy=ToolSelectionStrategy.EMBEDDING,
                llm_base_url="http://localhost",
                embed_model="nomic-embed-text",
            )
        assert "stop" in result

    @pytest.mark.unit
    def test_failed_query_embedding_falls_back(self):
        """If query embedding fails, fall back to all tools."""
        def mock_fail(text, base_url, model, timeout_sec=10.0):
            return None

        with patch("jarvis.memory.embeddings.get_embedding", side_effect=mock_fail):
            result = select_tools(
                "anything",
                _builtin(), _mcp(),
                strategy=ToolSelectionStrategy.EMBEDDING,
                llm_base_url="http://localhost",
                embed_model="nomic-embed-text",
            )
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_returns_minimum_tools(self):
        """Should return at least _MIN_SELECTED tools even if similarity is low."""
        # All tools get zero similarity (orthogonal to query)
        call_count = [0]
        def mock_embed(text, base_url, model, timeout_sec=10.0):
            call_count[0] += 1
            if call_count[0] == 1:  # query
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0, 1.0]  # all tools orthogonal

        with patch("jarvis.memory.embeddings.get_embedding", side_effect=mock_embed):
            result = select_tools(
                "something obscure",
                _builtin(), {},
                strategy=ToolSelectionStrategy.EMBEDDING,
                llm_base_url="http://localhost",
                embed_model="nomic-embed-text",
            )
        # Should still have at least _MIN_SELECTED + stop
        assert len(result) >= 3


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
                strategy=ToolSelectionStrategy.LLM,
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "stop" in result

    @pytest.mark.unit
    def test_none_response_returns_only_mandatory(self):
        def mock_llm(base_url, model, sys, user, timeout_sec=8.0):
            return "none"

        with patch("jarvis.llm.call_llm_direct", side_effect=mock_llm):
            result = select_tools(
                "hello",
                _builtin(), {},
                strategy=ToolSelectionStrategy.LLM,
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
                strategy=ToolSelectionStrategy.LLM,
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
                strategy=ToolSelectionStrategy.LLM,
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
                strategy=ToolSelectionStrategy.LLM,
                llm_base_url="http://localhost",
                llm_model="test",
            )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "nonExistentTool" not in result
