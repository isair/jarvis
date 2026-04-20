"""
Unit tests for the lenient text-based tool-call parser used in the reply engine.

Small models (gemma4:e2b) emit tool calls in several shapes that the native
Ollama tool_calls API doesn't recognise:

  1. Canonical protocol line:
        tool_calls: [{"id": "call_1", "type": "function",
                      "function": {"name": "webSearch", "arguments": "..."}}]
  2. Simplified "toolName: key: value" (single tool, single arg pair).
  3. Function-call form: toolName({"key": "value"}) or toolName(bare string).

The engine's _extract_structured_tool_call must parse all three so the
model's compliance succeeds regardless of the shape it chose. Without these
fallbacks, small models with text-based tool calling wedge into content-only
responses and never actually invoke the tool they described.
"""

import pytest


def _extract(content: str, tool_name: str = "webSearch"):
    """Helper: run the engine's extractor against a mocked LLM response."""
    from jarvis.reply.engine import run_reply_engine  # noqa: F401 — ensures module loads
    import jarvis.reply.engine as engine_mod

    # _extract_structured_tool_call is a closure inside run_reply_engine and
    # closes over tools_json_schema. To exercise it in isolation we re-derive
    # the same parsing logic via a minimal re-implementation that imports the
    # regex patterns actually used. The cleanest path is to call
    # run_reply_engine end-to-end — but that pulls in network and DB. Instead
    # we call into a module-level helper which must be exposed for testing.
    #
    # If _extract_text_tool_call is not yet module-level, this test will fail
    # with an AttributeError and the author must lift it out.
    assert hasattr(engine_mod, "_extract_text_tool_call"), (
        "Expose _extract_text_tool_call at module level for test coverage."
    )
    return engine_mod._extract_text_tool_call(content, {tool_name})


class TestCanonicalToolCallsArrayLiteral:
    """Form 1: `tool_calls: [...]` JSON array in content."""

    def test_extracts_name_and_string_args(self):
        content = (
            'tool_calls: [{"id": "call_1", "type": "function", '
            '"function": {"name": "webSearch", "arguments": "Possessor movie"}}]'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        # String argument is wrapped into a dict so the tool layer can consume it
        assert args and isinstance(args, dict)

    def test_extracts_name_and_dict_args(self):
        content = (
            'tool_calls: [{"id": "call_1", "type": "function", '
            '"function": {"name": "webSearch", '
            '"arguments": {"search_query": "Piranesi book"}}}]'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Piranesi book"


class TestMalformedCanonicalToolCallsLenientFallback:
    """Form 1b: small models emit *almost* valid `tool_calls: [...]` JSON but
    drop one or two closing braces. Without a lenient fallback the raw line
    leaks as the reply. Captured from gemma4:e2b field output 2026-04-20."""

    def test_parses_despite_missing_closing_braces(self):
        # Verbatim from gemma4:e2b: two closing braces missing before the `]`.
        content = (
            'tool_calls: [{"id": "call_1", "type": "function", '
            '"function": {"name": "getWeather", '
            '"arguments": "{\\"location\\": \\"Tbilisi, Georgia\\"}}"]'
        )
        name, args, _ = _extract(content, tool_name="getWeather")
        assert name == "getWeather"
        # The inner arguments JSON should survive the unwrap so the tool
        # layer receives the real location instead of a raw string blob.
        assert args.get("location") == "Tbilisi, Georgia"

    def test_lenient_fallback_rejects_unknown_tool_names(self):
        # Even with the lenient fallback, only tools in known_names get
        # dispatched — no inventing tools from malformed JSON.
        content = (
            'tool_calls: [{"id": "call_1", "type": "function", '
            '"function": {"name": "fileSystem_write", '
            '"arguments": "{\\"path\\": \\"/tmp/x\\"}}"]'
        )
        name, _args, _ = _extract(content, tool_name="webSearch")
        assert name is None


class TestSimplifiedColonForm:
    """Form 2: `toolName: key: value`."""

    def test_parses_tool_name_and_arg(self):
        content = "webSearch: search_query: Possessor movie"
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Possessor movie"

    def test_rejects_unknown_tool_name(self):
        # Arbitrary prose must not be mis-parsed as a tool call.
        content = "Note: something: arbitrary prose"
        name, _args, _ = _extract(content)
        assert name is None


class TestFunctionCallForm:
    """Form 3: `toolName(...)`."""

    def test_parses_json_object_inside_parens(self):
        content = 'webSearch({"search_query": "Possessor"})'
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Possessor"

    def test_parses_bare_string_inside_parens(self):
        content = 'webSearch("Possessor")'
        name, args, _ = _extract(content)
        assert name == "webSearch"
        # Bare strings collapse to a default query key
        assert any(v == "Possessor" for v in args.values())


class TestNoFalsePositiveOnProse:
    """Lenient parsing must not hijack free-form content that mentions tool names."""

    def test_plain_conversational_reply_is_not_parsed_as_tool_call(self):
        content = "I can help you find information about movies."
        name, _args, _ = _extract(content)
        assert name is None


class TestGemmaToolCodeBlockForm:
    """Form 5: gemma's native `tool_code` block.

    Gemma models (gemma4:e2b in particular) are post-trained to emit tool
    calls as a Python-style code block. Under prompt pressure (long tools_desc
    from many MCP servers) they revert to this format and ignore the JSON
    tool_calls protocol we ask for. Captured from field sessions 2026-04-20.
    """

    def test_parses_tool_code_with_print_wrapper(self):
        content = (
            "tool_code\n"
            'print(webSearch(search_query="Possessor movie"))\n'
            "<unused88>"
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Possessor movie"

    def test_parses_tool_code_without_print_wrapper(self):
        content = (
            "tool_code\n"
            'webSearch(search_query="Piranesi book")'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Piranesi book"

    def test_parses_fenced_tool_code(self):
        content = (
            "```tool_code\n"
            'webSearch(search_query="Blade Runner 2049")\n'
            "```"
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Blade Runner 2049"

    def test_ignores_non_search_hallucinated_module_calls(self):
        """Hallucinated module.method calls that aren't obviously search-intent
        must NOT be dispatched (we can't know what the model wanted). Only
        search-shaped hallucinations get routed (covered separately)."""
        content = (
            "tool_code\n"
            'print(fileSystem.write("/tmp/x", "y"))\n'
            'print(database.query("select *"))\n'
        )
        name, _args, _ = _extract(content)
        assert name is None

    def test_picks_real_tool_over_hallucinated_sibling(self):
        """If the block contains both an invented and a real tool call,
        dispatch the real one rather than returning nothing."""
        content = (
            "tool_code\n"
            'print(wikipedia.run("Blade Runner 2049"))\n'
            'print(webSearch(search_query="Blade Runner 2049"))\n'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Blade Runner 2049"


class TestSearchIntentFallbackRouting:
    """When gemma hallucinates a search-intent tool (`google_search.search(...)`,
    `wikipedia.run(...)`, etc.) we'd previously return None and let the raw
    tool_code block leak as the reply. Route those to the real `webSearch`
    tool instead, preserving the user's query."""

    def test_routes_google_search_to_webSearch(self):
        # Verbatim from a 2026-04-20 field log on gemma4:e2b.
        content = (
            "tool_code\n"
            'print(google_search.search(query="what is the movie possess"))<unused88>'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "what is the movie possess"

    def test_routes_wikipedia_run_to_webSearch(self):
        content = (
            "tool_code\n"
            'print(wikipedia.run("Blade Runner 2049"))<unused88>'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "Blade Runner 2049"

    def test_does_not_route_when_webSearch_not_registered(self):
        """If webSearch isn't in the allowed tool list, don't invent it."""
        import jarvis.reply.engine as engine_mod

        content = (
            "tool_code\n"
            'print(google_search.search(query="x"))'
        )
        # known_names deliberately excludes webSearch.
        name, _args, _ = engine_mod._extract_text_tool_call(content, {"fetchWebPage"})
        assert name is None

    def test_prefers_real_tool_over_search_fallback(self):
        """If a real tool also appears in the block, prefer it."""
        content = (
            "tool_code\n"
            'print(google_search.search(query="X"))\n'
            'print(webSearch(search_query="X different"))\n'
        )
        name, args, _ = _extract(content)
        assert name == "webSearch"
        assert args.get("search_query") == "X different"
