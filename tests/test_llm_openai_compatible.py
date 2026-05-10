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

    @patch("jarvis.llm.requests.post")
    def test_temperature_omitted_when_none(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        backend.direct("any-model", "sys", "user")  # default temperature=None

        sent = mock_post.call_args.kwargs["json"]
        assert "temperature" not in sent


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

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_timeout(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = requests.exceptions.Timeout("slow")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.streaming("any", "sys", "user") is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_connection_error(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = requests.exceptions.ConnectionError("server down")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.streaming("any", "sys", "user") is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_http_error(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        http_resp = MagicMock(status_code=500)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.streaming("any", "sys", "user") is None


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
    def test_propagates_connection_error(self, mock_post):
        """``chat`` re-raises ``ConnectionError`` so callers can distinguish
        an unreachable server from a transient HTTP failure."""
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = requests.exceptions.ConnectionError("server down")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        with pytest.raises(requests.exceptions.ConnectionError):
            backend.chat("any", [{"role": "user", "content": "hi"}])

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_http_500_even_with_tools(self, mock_post):
        """Only HTTP 400 with tools means "model rejects native tools" —
        500 is a server-side failure that should degrade gracefully."""
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        http_resp = MagicMock(status_code=500)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert (
            backend.chat(
                "any",
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )
            is None
        )

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_timeout(self, mock_post):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = requests.exceptions.Timeout("slow")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_generic_exception(self, mock_post):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = RuntimeError("unexpected")
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None


class TestOpenAICompatibleErrorMessagesDoNotLeakUrls:
    """``requests.exceptions.HTTPError`` and ``ConnectionError`` ``str()``s
    typically embed the request URL — and the request URL can carry sensitive
    query strings (e.g. some hosted providers accept ``?api_key=…`` literally,
    and configured endpoints may include team/account identifiers). The chat
    backend prints these errors to stdout, which surfaces in the desktop log
    pane and any captured terminal session. The error message must include
    enough information for diagnosis (status code, exception class) without
    echoing the URL the user configured."""

    _SECRET_URL = "http://internal-server.example.com:1234/v1"

    @patch("jarvis.llm.requests.post")
    def test_http_error_message_does_not_leak_endpoint_url(self, mock_post, capsys):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        # Construct an HTTPError whose str() embeds the URL — exactly what
        # ``requests`` does in real failures.
        http_resp = MagicMock(status_code=401)
        http_resp.url = f"{self._SECRET_URL}/chat/completions"
        err = requests.exceptions.HTTPError(
            f"401 Client Error: Unauthorized for url: {self._SECRET_URL}/chat/completions",
            response=http_resp,
        )
        mock_post.return_value = _make_response(raise_http=err)
        backend = OpenAICompatibleBackend(self._SECRET_URL, api_key="sk-secret")

        backend.chat("any", [{"role": "user", "content": "hi"}])

        captured = capsys.readouterr()
        assert self._SECRET_URL not in captured.out, (
            "HTTPError message must not echo the configured endpoint URL"
        )
        # Status code must still be visible so users can diagnose 401 vs 5xx.
        assert "401" in captured.out

    @patch("jarvis.llm.requests.post")
    def test_connection_error_message_does_not_leak_endpoint_url(self, mock_post, capsys):
        import requests
        from jarvis.llm import OpenAICompatibleBackend

        # Real-world ConnectionError messages include the URL via the
        # underlying urllib3 exception.
        mock_post.side_effect = requests.exceptions.ConnectionError(
            f"HTTPConnectionPool(host='internal-server.example.com', port=1234): "
            f"Max retries exceeded with url: /v1/chat/completions"
        )
        backend = OpenAICompatibleBackend(self._SECRET_URL, api_key="sk-secret")

        # ``chat`` re-raises so the caller can detect "server unreachable";
        # the printed log line must not carry the URL or the API key.
        with pytest.raises(requests.exceptions.ConnectionError):
            backend.chat("any", [{"role": "user", "content": "hi"}])

        captured = capsys.readouterr()
        assert "internal-server.example.com" not in captured.out, (
            "ConnectionError message must not echo the configured host"
        )
        assert "sk-secret" not in captured.out

    @patch("jarvis.llm.requests.post")
    def test_generic_exception_message_does_not_leak_url_or_key(self, mock_post, capsys):
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.side_effect = RuntimeError(
            f"unexpected: {self._SECRET_URL}?api_key=sk-secret"
        )
        backend = OpenAICompatibleBackend(self._SECRET_URL, api_key="sk-secret")

        backend.chat("any", [{"role": "user", "content": "hi"}])

        captured = capsys.readouterr()
        assert self._SECRET_URL not in captured.out
        assert "sk-secret" not in captured.out

    @patch("jarvis.llm.requests.post")
    def test_extra_options_merge_at_payload_root(self, mock_post):
        """OpenAI takes ``temperature`` / ``max_tokens`` at the payload
        root, not under an ``options`` nest like Ollama. The merge must
        land at the root."""
        from jarvis.llm import OpenAICompatibleBackend

        mock_post.return_value = _make_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        backend.chat(
            "any",
            [{"role": "user", "content": "hi"}],
            extra_options={"temperature": 0.5, "top_p": 0.9, "max_tokens": 256},
        )

        sent = mock_post.call_args.kwargs["json"]
        assert sent["temperature"] == 0.5
        assert sent["top_p"] == 0.9
        assert sent["max_tokens"] == 256
        # Must NOT live under a nested 'options' key (that's the Ollama shape).
        assert "options" not in sent


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

    @patch("jarvis.llm.requests.get")
    def test_returns_empty_list_when_data_field_missing(self, mock_get):
        from jarvis.llm import OpenAICompatibleBackend

        resp = MagicMock()
        resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        backend = OpenAICompatibleBackend("http://localhost:1234/v1")

        assert backend.list_models() == []


# ---------------------------------------------------------------------------
# _normalise_response — passthrough and edge cases
# ---------------------------------------------------------------------------


class TestNormaliseResponse:
    """``_normalise_response`` is the wire-shape translator that lets the
    reply engine consume OpenAI responses through the same parsing path
    it uses for Ollama. These tests pin the contract for the shapes a
    real server might emit."""

    def test_ollama_shaped_response_passes_through(self):
        """Hybrid servers that already return ``message`` at the top
        level (some llama.cpp builds do this) must not be re-wrapped."""
        from jarvis.llm.openai_compatible import _normalise_response

        upstream = {"message": {"content": "hi"}}

        assert _normalise_response(upstream) == upstream

    def test_arguments_already_dict_is_preserved(self):
        """Some servers pre-decode tool-call arguments to a dict. The
        normaliser must leave them alone rather than passing them
        through ``json.loads`` (which would raise)."""
        from jarvis.llm.openai_compatible import _normalise_response

        upstream = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "x",
                                    "arguments": {"already": "dict"},
                                }
                            }
                        ],
                    }
                }
            ]
        }

        result = _normalise_response(upstream)

        tc = result["message"]["tool_calls"][0]
        assert tc["function"]["arguments"] == {"already": "dict"}

    def test_malformed_arguments_string_left_as_is(self):
        """If a server returns invalid JSON in ``arguments``, the
        normaliser leaves the string in place so the engine's
        content-mode parser can still attempt recovery — better than
        raising and losing the whole turn."""
        from jarvis.llm.openai_compatible import _normalise_response

        upstream = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "x",
                                    "arguments": "not valid json",
                                }
                            }
                        ],
                    }
                }
            ]
        }

        result = _normalise_response(upstream)

        tc = result["message"]["tool_calls"][0]
        assert tc["function"]["arguments"] == "not valid json"
