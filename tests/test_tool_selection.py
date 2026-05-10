"""Tests for tool selection strategies."""

import pytest
from unittest.mock import MagicMock

from jarvis.tools.selection import (
    select_tools,
    ToolSelectionStrategy,
    _tokenise,
    _build_tool_keywords,
    _ALWAYS_INCLUDED,
    _RELATIVE_THRESHOLD,
)


def _embedding_backend(text_to_vec=None, fail=False):
    """Build a MagicMock embedding backend whose ``embed`` looks up text by
    substring match against ``text_to_vec`` (a dict). Falls back to a zero
    vector for unmatched text. ``fail=True`` makes embed always return None."""
    backend = MagicMock()
    if fail:
        backend.embed.return_value = None
        return backend

    def _embed(text, model, timeout_sec=10.0):
        if text_to_vec is None:
            return [0.0] * 4
        for key, vec in text_to_vec.items():
            if key in text.lower():
                return vec
        return [0.0] * 4

    backend.embed.side_effect = _embed
    return backend


def _llm_backend(direct_fn=None, return_value=None, raises=None):
    """Build a MagicMock chat backend whose ``direct`` returns or raises."""
    backend = MagicMock()
    if raises is not None:
        backend.direct.side_effect = raises
    elif direct_fn is not None:
        backend.direct.side_effect = direct_fn
    else:
        backend.direct.return_value = return_value
    return backend


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

    @pytest.mark.unit
    def test_selects_similar_tools(self):
        """Weather query should rank getWeather highest."""
        backend = _embedding_backend({
            "weather": [1.0, 0.0, 0.0, 0.0],      # query + weather tool
            "search": [0.0, 1.0, 0.0, 0.0],
            "meal": [0.0, 0.0, 1.0, 0.0],
            "screen": [0.0, 0.0, 0.0, 1.0],
            "file": [0.1, 0.1, 0.1, 0.1],
            "conversation": [0.1, 0.1, 0.1, 0.1],
        })
        result = select_tools(
            "what's the weather",
            _builtin(), {},
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=backend,
            embed_model="nomic-embed-text",
        )
        assert "getWeather" in result

    @pytest.mark.unit
    def test_stop_always_included(self):
        """Stop tool must be present even if not semantically matched."""
        backend = _embedding_backend({"weather": [1.0, 0.0, 0.0, 0.0]})
        result = select_tools(
            "what's the weather",
            _builtin(), {},
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=backend,
            embed_model="nomic-embed-text",
        )
        assert "stop" in result

    @pytest.mark.unit
    def test_failed_query_embedding_falls_back(self):
        """If query embedding fails, fall back to all tools."""
        backend = _embedding_backend(fail=True)
        result = select_tools(
            "anything",
            _builtin(), _mcp(),
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=backend,
            embed_model="nomic-embed-text",
        )
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_no_embedding_backend_falls_back_to_all(self):
        """When no embedding backend is supplied (e.g. unconfigured runtime),
        the embedding strategy degrades to returning the full catalogue
        rather than crashing."""
        result = select_tools(
            "anything",
            _builtin(), _mcp(),
            strategy=ToolSelectionStrategy.EMBEDDING,
            embed_model="nomic-embed-text",
        )
        assert len(result) == len(_builtin()) + len(_mcp())

    @pytest.mark.unit
    def test_returns_minimum_tools(self):
        """Should return at least _MIN_SELECTED tools even if similarity is low."""
        call_count = [0]
        def _embed(text, model, timeout_sec=10.0):
            call_count[0] += 1
            if call_count[0] == 1:  # query
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0, 1.0]
        backend = MagicMock()
        backend.embed.side_effect = _embed

        result = select_tools(
            "something obscure",
            _builtin(), {},
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=backend,
            embed_model="nomic-embed-text",
        )
        assert len(result) >= 3

    @pytest.mark.unit
    def test_relative_threshold_filters_low_similarity(self):
        """Relative threshold keeps only tools near the top score, not everything."""
        import math

        strong = [0.9, 0.436, 0, 0]
        s_norm = math.sqrt(sum(x*x for x in strong))
        strong = [x / s_norm for x in strong]

        good = [0.88, 0.475, 0, 0]
        g_norm = math.sqrt(sum(x*x for x in good))
        good = [x / g_norm for x in good]

        weak = [0.4, 0.917, 0, 0]
        w_norm = math.sqrt(sum(x*x for x in weak))
        weak = [x / w_norm for x in weak]

        mock_map = {
            "weather": [1.0, 0.0, 0.0, 0.0],     # query
            "get weather": strong,                  # getWeather → high sim
            "web search": good,                     # webSearch → just above cutoff
            "log meal": weak,                       # logMeal → low sim
            "fetch meals": weak,                    # fetchMeals → low sim
            "screen": weak,                         # screenshot → low sim
            "file": weak,                           # localFiles → low sim
        }

        backend = _embedding_backend(mock_map)

        result = select_tools(
            "what's the weather",
            _builtin(), {},
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=backend,
            embed_model="nomic-embed-text",
        )

        # Strong and good matches must be included
        assert "getWeather" in result
        assert "webSearch" in result

        # stop is always included
        assert "stop" in result

        # Fewer tools than total — the relative threshold actually filtered
        total_non_stop = len([t for t in _builtin() if t != "stop"])
        selected_non_stop = len([t for t in result if t != "stop"])
        assert selected_non_stop < total_non_stop, (
            f"Expected fewer than {total_non_stop} tools but got {selected_non_stop}: {result}"
        )


# ---------------------------------------------------------------------------
# Strategy: llm
# ---------------------------------------------------------------------------

class TestLLMStrategy:

    @pytest.mark.unit
    def test_parses_comma_separated_response(self):
        backend = _llm_backend(return_value="webSearch, getWeather")
        result = select_tools(
            "what's the weather",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "stop" in result

    @pytest.mark.unit
    def test_none_response_returns_only_mandatory(self):
        backend = _llm_backend(return_value="none")
        result = select_tools(
            "hello",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        assert result == ["stop"]

    @pytest.mark.unit
    def test_llm_failure_falls_back_to_keyword(self):
        """When the router LLM raises (timeout, network, etc.) the fallback is
        keyword scoring — not the full catalogue. A 30+-tool fall-open kills
        small chat models (they choke on 41-tool prompts) and pins the
        conversation cache to "everything"; keyword narrowing preserves at
        least some routing on tool-name overlap with the query."""
        backend = _llm_backend(raises=TimeoutError("LLM timed out"))
        result = select_tools(
            "weather in London",
            _builtin(), _mcp(),
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        # Keyword strategy on "weather" picks getWeather (its name + desc both
        # contain "weather"); irrelevant tools like fetchMeals must NOT appear.
        assert "getWeather" in result
        assert "fetchMeals" not in result
        assert "homeassistant__turn_on" not in result

    @pytest.mark.unit
    def test_empty_response_falls_back_to_keyword(self):
        """Empty router response is treated identically to a hard failure:
        fall back to keyword scoring rather than to the full catalogue."""
        backend = _llm_backend(return_value="")
        result = select_tools(
            "weather report",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        assert "getWeather" in result
        assert "fetchMeals" not in result

    @pytest.mark.unit
    def test_no_llm_backend_falls_back_to_keyword(self):
        """When no LLM backend is supplied (e.g. unconfigured runtime), the
        LLM strategy degrades to keyword scoring rather than crashing."""
        result = select_tools(
            "weather in London",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_model="test",
        )
        assert "getWeather" in result
        assert "fetchMeals" not in result

    @pytest.mark.unit
    def test_unparseable_response_falls_back_to_keyword(self):
        """When the router response is non-empty but no token matches a known
        tool name (small-model garbage), the fallback is keyword scoring.
        Field trace: a small router occasionally produces text like "I think
        we should..." that the parser strips to nothing — pre-fix this fell
        open to all 41 tools; post-fix it narrows on query keywords."""
        backend = _llm_backend(return_value="I think we should pick one")
        result = select_tools(
            "navigate to youtube.com",
            _builtin(),
            {"chrome-devtools__navigate_page": FakeToolSpec(
                "chrome-devtools__navigate_page",
                "Navigate the browser to a given URL.",
            )},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        # Keyword scoring matches "navigate" → chrome-devtools__navigate_page.
        assert "chrome-devtools__navigate_page" in result
        # The full catalogue must NOT be returned — that's the regression we're
        # fixing (small-model 41-tool overload).
        assert len(result) < len(_builtin()) + 1

    @pytest.mark.unit
    def test_ignores_hallucinated_tool_names(self):
        backend = _llm_backend(return_value="webSearch, nonExistentTool, getWeather")
        result = select_tools(
            "search and weather",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        assert "webSearch" in result
        assert "getWeather" in result

    @pytest.mark.unit
    def test_parses_markdown_and_backtick_wrapped_names(self):
        """Chatty routers wrap names in backticks, bullets, or JSON brackets.
        The parser must strip that formatting before matching — a literal
        `webSearch` should resolve to the tool called webSearch, not be
        silently dropped as an unknown token."""
        backend = _llm_backend(return_value="- `webSearch`, * `getWeather`, [logMeal]")
        result = select_tools(
            "chatty router",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        assert "webSearch" in result
        assert "getWeather" in result
        assert "logMeal" in result

    @pytest.mark.unit
    def test_caps_chatty_router_output_at_max(self):
        """A router that echoes the whole catalogue must still produce a
        compact selection — the hard cap guarantees downstream prompt size."""
        from jarvis.tools.selection import _LLM_MAX_SELECTED

        backend = _llm_backend(
            return_value="webSearch, getWeather, logMeal, fetchMeals, screenshot, localFiles, homeassistant__turn_on"
        )
        result = select_tools(
            "arbitrary query",
            _builtin(), _mcp(),
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
        )
        non_mandatory = [t for t in result if t not in _ALWAYS_INCLUDED]
        assert len(non_mandatory) <= _LLM_MAX_SELECTED, (
            f"Expected at most {_LLM_MAX_SELECTED} non-mandatory tools, got "
            f"{len(non_mandatory)}: {non_mandatory}"
        )
        assert non_mandatory[0] == "webSearch"
        assert "nonExistentTool" not in result

    @pytest.mark.unit
    def test_context_hint_splits_into_known_facts_and_recent_dialogue(self):
        """When the hint carries a 'Recent dialogue' subsection, the router
        prompt must surface facts and dialogue under separate labels so the
        router can read a short follow-up ("I'm in London") as a continuation
        of the prior turn rather than as standalone idle chatter."""
        captured = {}

        def _direct(model, sys, user, timeout_sec=8.0):
            captured["sys"] = sys
            captured["user"] = user
            return "getWeather"

        backend = _llm_backend(direct_fn=_direct)

        hint = (
            "Current local time: Sunday, 2026-04-20 17:42 (Europe/London).\n\n"
            "Recent dialogue (short-term memory):\n"
            "- user: what's the weather like?\n"
            "- assistant: Sure — where should I check?"
        )
        select_tools(
            "I'm in London",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
            context_hint=hint,
        )

        assert "KNOWN FACTS" in captured["user"]
        assert "RECENT DIALOGUE" in captured["user"]
        dialogue_idx = captured["user"].index("RECENT DIALOGUE")
        assert "where should I check" in captured["user"][dialogue_idx:]
        assert "continuation" in captured["sys"].lower()

    @pytest.mark.unit
    def test_context_hint_without_dialogue_uses_known_facts_only(self):
        """When the hint carries no dialogue subsection (first turn, no
        recent messages), the router must still work — the facts flow
        through under the KNOWN FACTS label with no dialogue block."""
        captured = {}

        def _direct(model, sys, user, timeout_sec=8.0):
            captured["user"] = user
            return "getWeather"

        backend = _llm_backend(direct_fn=_direct)

        hint = "Current local time: Sunday, 2026-04-20 17:42 (Europe/London)."
        select_tools(
            "what's the weather?",
            _builtin(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=backend,
            llm_model="test",
            context_hint=hint,
        )

        assert "KNOWN FACTS" in captured["user"]
        assert "RECENT DIALOGUE" not in captured["user"]
