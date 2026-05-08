"""Behaviour tests for the OpenAI-compatible LLM backend.

These tests pin the wire shape and response normalisation that lets the
reply engine consume responses from any OpenAI-compatible local server
(LM Studio, oMLX, llama.cpp's ``llama-server``, vLLM, LocalAI, ...) the
same way it consumes Ollama responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_response(*, json_data=None, iter_lines=None, status_code=200, raise_http=None):
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    if iter_lines is not None:
        resp.iter_lines.return_value = iter_lines
    if raise_http is not None:
        resp.raise_for_status.side_effect = raise_http
    else:
        resp.raise_for_status = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    return resp


# ---------------------------------------------------------------------------
# direct() — single-shot system+user
# ---------------------------------------------------------------------------


class TestOpenAICompatibleDirect:
    @patch("jarvis.llm.requests.post")
    def test_returns_assistant_text_from_choices(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "hello"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.direct("any-model", "sys", "user") == "hello"

    @patch("jarvis.llm.requests.post")
    def test_posts_to_chat_completions_endpoint(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1/")
        backend.direct("any-model", "sys", "user")

        url = mock_post.call_args[0][0]
        assert url == "http://localhost:1234/v1/chat/completions"

    @patch("jarvis.llm.requests.post")
    def test_sends_authorization_header_when_api_key_set(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1", api_key="sk-test")
        backend.direct("any-model", "sys", "user")

        headers = mock_post.call_args.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-test"

    @patch("jarvis.llm.requests.post")
    def test_omits_authorization_header_when_no_key(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")
        backend.direct("any-model", "sys", "user")

        headers = mock_post.call_args.kwargs.get("headers") or {}
        assert "Authorization" not in headers

    @patch("jarvis.llm.requests.post")
    def test_temperature_and_model_in_payload(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")
        backend.direct("my-model", "sys", "user", temperature=0.0)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "my-model"
        assert sent["temperature"] == 0.0
        assert sent["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_failure(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = RuntimeError("server down")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.direct("any", "sys", "user") is None


# ---------------------------------------------------------------------------
# streaming() — SSE
# ---------------------------------------------------------------------------


class TestOpenAICompatibleStreaming:
    @patch("jarvis.llm.requests.post")
    def test_invokes_on_token_per_sse_chunk_and_returns_full_text(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        sse = [
            b'data: {"choices": [{"delta": {"content": "hel"}}]}',
            b"",
            b'data: {"choices": [{"delta": {"content": "lo"}}]}',
            b"",
            b"data: [DONE]",
        ]
        mock_post.return_value = _make_response(iter_lines=sse)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        seen: list[str] = []
        result = backend.streaming("any", "sys", "user", on_token=seen.append)

        assert seen == ["hel", "lo"]
        assert result == "hello"

    @patch("jarvis.llm.requests.post")
    def test_works_without_on_token_callback(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        sse = [
            b'data: {"choices": [{"delta": {"content": "x"}}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = _make_response(iter_lines=sse)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.streaming("any", "sys", "user") == "x"

    @patch("jarvis.llm.requests.post")
    def test_skips_keepalive_and_invalid_lines(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        sse = [
            b": keepalive",
            b"not-a-data-line",
            b'data: not-json',
            b'data: {"choices": [{"delta": {"content": "ok"}}]}',
        ]
        mock_post.return_value = _make_response(iter_lines=sse)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.streaming("any", "sys", "user") == "ok"


# ---------------------------------------------------------------------------
# chat() — arbitrary messages, normalised to Ollama-shaped response
# ---------------------------------------------------------------------------


class TestOpenAICompatibleChat:
    @patch("jarvis.llm.requests.post")
    def test_lifts_choices_message_to_top_level(self, mock_post):
        """The reply engine reads ``resp['message']['content']``;
        OpenAI returns ``choices[0].message.content``. The backend must
        normalise the response so existing callers see Ollama's shape."""
        from jarvis.llm import OpenAICompatibleBackend

        upstream = {"choices": [{"message": {"content": "hello"}}]}
        mock_post.return_value = _make_response(json_data=upstream)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        result = backend.chat("any", [{"role": "user", "content": "hi"}])

        assert result is not None
        assert result["message"]["content"] == "hello"

    @patch("jarvis.llm.requests.post")
    def test_decodes_tool_call_arguments_string_to_dict(self, mock_post):
        """OpenAI returns ``tool_calls[*].function.arguments`` as a JSON
        string; Ollama returns it as a dict, and the reply engine
        expects a dict. The backend must decode."""
        from jarvis.llm import OpenAICompatibleBackend

        upstream = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "getWeather",
                                    "arguments": '{"location": "Tbilisi"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_post.return_value = _make_response(json_data=upstream)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        result = backend.chat("any", [{"role": "user", "content": "weather"}])

        tc = result["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "getWeather"
        assert tc["function"]["arguments"] == {"location": "Tbilisi"}

    @patch("jarvis.llm.requests.post")
    def test_passes_tools_in_payload(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")
        backend.chat(
            "any",
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
        )

        sent = mock_post.call_args.kwargs["json"]
        assert sent["tools"][0]["function"]["name"] == "x"

    @patch("jarvis.llm.requests.post")
    def test_raises_tools_not_supported_on_400_with_tools(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend, ToolsNotSupportedError

        http_resp = MagicMock(status_code=400)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        with pytest.raises(ToolsNotSupportedError):
            backend.chat(
                "any",
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_400_without_tools(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        http_resp = MagicMock(status_code=400)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_connection_error(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = requests.exceptions.ConnectionError("server down")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None


# ---------------------------------------------------------------------------
# embed() — POST /embeddings
# ---------------------------------------------------------------------------


class TestOpenAICompatibleEmbed:
    @patch("jarvis.llm.requests.post")
    def test_returns_vector_from_data_array(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        resp = MagicMock()
        resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.embed("hello", "text-embedding-3-small") == [0.1, 0.2, 0.3]

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_error(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = RuntimeError("boom")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.embed("hello", "text-embedding-3-small") is None

    @patch("jarvis.llm.requests.post")
    def test_sends_authorization_when_api_key_set(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        resp = MagicMock()
        resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp
        backend = OpenAICompatibleBackend("http://localhost:1234/v1", api_key="sk-x")
        backend.embed("hello", "model")

        headers = mock_post.call_args.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-x"


# ---------------------------------------------------------------------------
# list_models() — GET /models
# ---------------------------------------------------------------------------


class TestOpenAICompatibleListModels:
    @patch("jarvis.llm.requests.get")
    def test_returns_model_ids(self, mock_get):
        from jarvis.llm import OpenAICompatibleBackend

        resp = MagicMock()
        resp.json.return_value = {
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "lmstudio-community/gemma-3-4b-it-GGUF"},
            ]
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        names = backend.list_models()

        assert names == ["gpt-4o-mini", "lmstudio-community/gemma-3-4b-it-GGUF"]

    @patch("jarvis.llm.requests.get")
    def test_returns_empty_list_on_failure(self, mock_get):
        from jarvis.llm import OpenAICompatibleBackend

        mock_get.side_effect = RuntimeError("boom")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.list_models() == []
