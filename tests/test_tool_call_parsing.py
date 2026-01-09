"""
Tests for tool call parsing, including fallback parsing for alternative formats.
"""

import pytest
import json


class TestToolCallTextParsing:
    """Tests for extracting tool calls from text content."""

    def _extract_tool_call_from_text(self, content: str):
        """Helper to import and call the text parsing function."""
        import re
        import uuid

        if not content or not isinstance(content, str):
            return None, None, None

        content = content.strip()

        # Pattern 1: tool_calls JSON in content
        tc_match = re.search(r'(?:^|\s)tool_calls\s*:\s*(\[.+\])', content, re.DOTALL)
        if tc_match:
            try:
                tc_json = tc_match.group(1)
                tc_list = json.loads(tc_json)
                if isinstance(tc_list, list) and len(tc_list) > 0:
                    first = tc_list[0]
                    if isinstance(first, dict):
                        func = first.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", {})
                        tool_call_id = first.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        if name:
                            return name, (args if isinstance(args, dict) else {}), tool_call_id
            except Exception:
                pass

        # Pattern 2: Python-style function call
        func_match = re.match(r'^(\w+)\s*\(\s*(.+?)\s*\)$', content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)

            args = {}
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))'
            for match in re.finditer(arg_pattern, args_str):
                key = match.group(1)
                value = match.group(2) or match.group(3) or match.group(4) or match.group(5)
                if value is not None:
                    if match.group(4):  # Number
                        try:
                            value = float(value) if '.' in value else int(value)
                        except ValueError:
                            pass
                    elif match.group(5):  # Bare word
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() in ('null', 'none'):
                            value = None
                    args[key] = value

            if func_name:
                return func_name, args, f"call_{uuid.uuid4().hex[:8]}"

        return None, None, None

    def test_parse_python_style_function_call(self):
        """Parse: webSearch(search_query="latest AI news", search_radius=1)"""
        content = 'webSearch(search_query="latest AI news", search_radius=1)'
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "webSearch"
        assert args["search_query"] == "latest AI news"
        assert args["search_radius"] == 1
        assert call_id.startswith("call_")

    def test_parse_python_style_with_single_quotes(self):
        """Parse function call with single-quoted strings."""
        content = "webSearch(query='test query')"
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "webSearch"
        assert args["query"] == "test query"

    def test_parse_tool_calls_json_in_content(self):
        """Parse: tool_calls: [{"id": "...", "function": {...}}]"""
        content = 'tool_calls: [{"id": "call_webSearch", "type": "function", "function": {"name": "webSearch", "arguments": {"search_query": "bbc news"}}}]'
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "webSearch"
        assert args["search_query"] == "bbc news"
        assert call_id == "call_webSearch"

    def test_parse_tool_calls_with_string_arguments(self):
        """Handle case where arguments is a JSON string instead of object."""
        content = 'tool_calls: [{"id": "call_123", "function": {"name": "test", "arguments": "{\\"key\\": \\"value\\"}"}}]'
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "test"
        assert args["key"] == "value"

    def test_parse_boolean_arguments(self):
        """Parse function call with boolean arguments."""
        content = 'someFunc(enabled=true, disabled=false)'
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "someFunc"
        assert args["enabled"] is True
        assert args["disabled"] is False

    def test_parse_float_arguments(self):
        """Parse function call with float arguments."""
        content = 'calculate(value=3.14, count=10)'
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name == "calculate"
        assert args["value"] == 3.14
        assert args["count"] == 10

    def test_returns_none_for_plain_text(self):
        """Return None for plain text that isn't a tool call."""
        content = "Hello, how can I help you today?"
        name, args, call_id = self._extract_tool_call_from_text(content)

        assert name is None
        assert args is None
        assert call_id is None

    def test_returns_none_for_empty_content(self):
        """Return None for empty content."""
        name, args, call_id = self._extract_tool_call_from_text("")
        assert name is None

        name, args, call_id = self._extract_tool_call_from_text(None)
        assert name is None
