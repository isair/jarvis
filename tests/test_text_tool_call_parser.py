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
