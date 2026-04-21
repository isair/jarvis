"""Unit tests for the lenient text-based tool-call parser.

Small models emit tool calls in several shapes that the native Ollama
tool_calls API doesn't recognise. The engine's ``_extract_text_tool_call``
must parse these so the model's compliance succeeds regardless of shape.

The gemma-native ``tool_code`` branch was removed in the evaluator-driven
loop refactor — the model is now responsible for producing a valid tool
call, and the evaluator / toolSearchTool path replaces the safety net.
"""

import pytest


def _extract(content: str, tool_name: str = "webSearch"):
    import jarvis.reply.engine as engine_mod
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
    """Form 1b: small models emit almost-valid JSON that drops closing braces.

    Without the lenient fallback the raw line leaks as the reply.
    """

    def test_parses_despite_missing_closing_braces(self):
        content = (
            'tool_calls: [{"id": "call_1", "type": "function", '
            '"function": {"name": "getWeather", '
            '"arguments": "{\\"location\\": \\"Tbilisi, Georgia\\"}}"]'
        )
        name, args, _ = _extract(content, tool_name="getWeather")
        assert name == "getWeather"
        assert args.get("location") == "Tbilisi, Georgia"

    def test_lenient_fallback_rejects_unknown_tool_names(self):
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
        assert any(v == "Possessor" for v in args.values())


class TestNoFalsePositiveOnProse:
    def test_plain_conversational_reply_is_not_parsed_as_tool_call(self):
        content = "I can help you find information about movies."
        name, _args, _ = _extract(content)
        assert name is None


def _is_malformed(content: str) -> bool:
    import jarvis.reply.engine as engine_mod
    assert hasattr(engine_mod, "_is_malformed_model_output"), (
        "Expose _is_malformed_model_output at module level for test coverage."
    )
    return engine_mod._is_malformed_model_output(content)


class TestMalformedModelOutputGuard:
    """``_is_malformed_model_output`` gates content before it can reach the
    user. Covers the field-captured leak shapes we have observed from
    small models (gemma4:e2b/e4b) after tool results."""

    @pytest.mark.parametrize(
        "content,label",
        [
            ("tool_calls: []", "bare tool_calls literal"),
            ("tool_calls: [{}]", "tool_calls with stub entry"),
            ("tool_code\n  print(google_search.search(query='x'))\n  ", "gemma tool_code block"),
            ("tool_output\n[{'snippet': 'x'}]", "gemma tool_output block"),
            ("Okay, here is your answer <unused88>", "unused sentinel inline"),
            ("Reply ends with <unused10>.", "different unused sentinel"),
            ("{\"forecast\": 14, \"high\": 15", "truncated JSON (no closing brace)"),
            ('{"openapi": "3.0.0", "paths": {}}', "OpenAPI spec dump"),
            ('{"location": "Hackney", "forecast": "cloudy"}', "weather JSON dump"),
        ],
    )
    def test_detects_malformed_shape(self, content, label):
        assert _is_malformed(content), f"Should flag: {label!r} -> {content!r}"

    @pytest.mark.parametrize(
        "content",
        [
            "Sure, the capital of France is Paris.",
            "I found three results: Blinding Lights, Anti-Hero, and Levitating.",
            "I couldn't read the page contents this time. Want me to retry?",
            # Starts with { but closes properly AND has a conversational field.
            '{"response": "Here you go."}',
        ],
    )
    def test_allows_normal_prose(self, content):
        assert not _is_malformed(content), f"Should not flag prose: {content!r}"
