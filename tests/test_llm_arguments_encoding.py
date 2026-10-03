"""Test that tool_calls arguments are properly encoded for the wire format.

The OpenAI API spec requires ``tool_calls[*].function.arguments`` to be a
JSON-encoded string. The ``normalise_openai_response`` helper decodes it to a
dict for internal use, but when that assistant message is sent back to the
server on the next turn, the dict must be re-encoded as a string.

This test verifies the round-trip: decode -> store -> re-encode before send.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestToolCallArgumentsWireFormat:
    """Verifies that tool_calls arguments are JSON-encoded strings on the wire."""

    def _make_backend(self):
        from jarvis.llm.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(base_url="http://localhost:11434")

    def _patch_post(self, backend, messages, model="some-model"):
        """Call backend.chat() with a mocked POST and return the sent payload."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "OK", "role": "assistant"}}]
            }
            mock_post.return_value = mock_response
            backend.chat(model, messages)
        return mock_post.call_args[1]["json"]

    def test_dict_arguments_encoded_as_json_string(self):
        """Empty dict arguments -> JSON string."""
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "getWeather", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny"},
        ]
        payload = self._patch_post(backend, messages)

        for msg in payload["messages"]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                args = msg["tool_calls"][0]["function"]["arguments"]
                assert isinstance(args, str), f"Expected str, got {type(args).__name__}"
                assert json.loads(args) == {}
                return
        pytest.fail("No assistant message with tool_calls found")

    def test_nested_dict_arguments(self):
        """Nested dict arguments (e.g. location, units) -> valid JSON string."""
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "Assistant."},
            {"role": "user", "content": "Weather in London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "getWeather",
                            "arguments": {
                                "location": "London",
                                "units": "metric",
                                "days": 3,
                            },
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "15C rainy"},
        ]
        payload = self._patch_post(backend, messages)

        for msg in payload["messages"]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                args = msg["tool_calls"][0]["function"]["arguments"]
                assert isinstance(args, str), f"Expected str, got {type(args).__name__}"
                decoded = json.loads(args)
                assert decoded["location"] == "London"
                assert decoded["units"] == "metric"
                assert decoded["days"] == 3
                return
        pytest.fail("No assistant message with tool_calls found")

    def test_string_arguments_left_as_is(self):
        """Already-encoded string arguments must not be double-encoded."""
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "Assistant."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',  # already a string
                        },
                    }
                ],
            },
        ]
        payload = self._patch_post(backend, messages)

        for msg in payload["messages"]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                args = msg["tool_calls"][0]["function"]["arguments"]
                assert isinstance(args, str)
                # Should still be the exact same string, not double-encoded
                assert args == '{"query": "test"}'
                return
        pytest.fail("No assistant message with tool_calls found")

    def test_no_tool_calls_unchanged(self):
        """Messages without tool_calls should be left as-is."""
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "Assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        payload = self._patch_post(backend, messages)
        assert payload["messages"] == messages

    def test_multiple_tool_calls_all_encoded(self):
        """All tool_calls in an assistant message should have encoded arguments."""
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "Assistant."},
            {"role": "user", "content": "Compare weather in Paris and Tokyo"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "getWeather", "arguments": {"location": "Paris"}},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "getWeather", "arguments": {"location": "Tokyo"}},
                    },
                ],
            },
        ]
        payload = self._patch_post(backend, messages)

        for msg in payload["messages"]:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    assert isinstance(args, str), f"Expected str, got {type(args).__name__}"
                    json.loads(args)  # must be valid JSON
                # Verify both locations present
                decoded = [json.loads(tc["function"]["arguments"]) for tc in msg["tool_calls"]]
                assert {"location": "Paris"} in decoded
                assert {"location": "Tokyo"} in decoded
                return
        pytest.fail("No assistant message with tool_calls found")
