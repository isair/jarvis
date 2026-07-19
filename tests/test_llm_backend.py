"""Behaviour tests for the pluggable LLM backend abstraction.

PR 1 covers the Ollama backend only. These tests pin the
provider-agnostic ``LLMBackend`` interface against the ``OllamaBackend``
implementation, and confirm that the function-style entry points
(``call_llm_direct``, ``call_llm_streaming``, ``chat_with_messages``)
dispatch to the same backend.

The tests intentionally exercise observable behaviour (return values,
``on_token`` callbacks, raised errors) rather than implementation
details such as which exact URL was hit.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_response(*, json_data=None, iter_lines=None, status_code=200, raise_http=None):
    """Build a MagicMock that behaves like a ``requests.Response`` with
    context-manager support, since the real code uses ``with requests.post(...)``.
    """
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
# OllamaBackend — chat / direct / streaming
# ---------------------------------------------------------------------------


class TestOllamaBackendDirect:
    @patch("jarvis.llm.requests.post")
    def test_returns_assistant_text(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "hello"}})
        backend = OllamaBackend("http://localhost:11434")

        result = backend.direct("gemma4:e2b", "sys", "user")

        assert result == "hello"

    @patch("jarvis.llm.requests.post")
    def test_strips_trailing_slash_from_base_url(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})
        backend = OllamaBackend("http://localhost:11434/")
        backend.direct("gemma4:e2b", "sys", "user")

        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_empty_content(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "   "}})
        backend = OllamaBackend("http://localhost:11434")

        assert backend.direct("gemma4:e2b", "sys", "user") is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_request_failure(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.side_effect = RuntimeError("boom")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.direct("gemma4:e2b", "sys", "user") is None


class TestOllamaBackendStreaming:
    @patch("jarvis.llm.requests.post")
    def test_invokes_on_token_per_chunk_and_returns_full_text(self, mock_post):
        from jarvis.llm import OllamaBackend

        chunks = [
            json.dumps({"message": {"content": "hel"}}).encode(),
            json.dumps({"message": {"content": "lo"}}).encode(),
            json.dumps({"message": {"content": " world"}}).encode(),
        ]
        mock_post.return_value = _make_response(iter_lines=chunks)
        backend = OllamaBackend("http://localhost:11434")

        seen: list[str] = []
        result = backend.streaming(
            "gemma4:e2b", "sys", "user", on_token=seen.append
        )

        assert seen == ["hel", "lo", " world"]
        assert result == "hello world"

    @patch("jarvis.llm.requests.post")
    def test_returns_none_when_stream_is_empty(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(iter_lines=[])
        backend = OllamaBackend("http://localhost:11434")

        assert backend.streaming("gemma4:e2b", "sys", "user") is None


class TestOllamaBackendChat:
    @patch("jarvis.llm.requests.post")
    def test_returns_raw_response_dict(self, mock_post):
        from jarvis.llm import OllamaBackend

        payload = {"message": {"content": "answer", "tool_calls": [{"function": {"name": "x"}}]}}
        mock_post.return_value = _make_response(json_data=payload)
        backend = OllamaBackend("http://localhost:11434")

        result = backend.chat("gpt-oss:20b", [{"role": "user", "content": "hi"}])

        assert result == payload

    @patch("jarvis.llm.requests.post")
    def test_raises_tools_not_supported_on_http_400_with_tools(self, mock_post):
        import requests
        from jarvis.llm import OllamaBackend, ToolsNotSupportedError

        http_resp = MagicMock(status_code=400)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OllamaBackend("http://localhost:11434")

        with pytest.raises(ToolsNotSupportedError):
            backend.chat(
                "small-model",
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_http_400_without_tools(self, mock_post):
        import requests
        from jarvis.llm import OllamaBackend

        http_resp = MagicMock(status_code=400)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_http_500(self, mock_post):
        import requests
        from jarvis.llm import OllamaBackend

        http_resp = MagicMock(status_code=500)
        err = requests.exceptions.HTTPError(response=http_resp)
        mock_post.return_value = _make_response(raise_http=err)
        backend = OllamaBackend("http://localhost:11434")

        # 500 must not raise ToolsNotSupportedError even when tools are passed
        # — only 400 means "this model does not support native tools".
        assert (
            backend.chat(
                "any",
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )
            is None
        )

    @patch("jarvis.llm.requests.post")
    def test_propagates_connection_error(self, mock_post):
        """``chat`` re-raises ``ConnectionError`` so callers can distinguish
        an unreachable server from a transient HTTP failure and apply their
        own back-off (e.g. the intent judge's 30s cooldown)."""
        import requests
        from jarvis.llm import OllamaBackend

        mock_post.side_effect = requests.exceptions.ConnectionError("server down")
        backend = OllamaBackend("http://localhost:11434")

        with pytest.raises(requests.exceptions.ConnectionError):
            backend.chat("any", [{"role": "user", "content": "hi"}])

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_timeout(self, mock_post):
        import requests
        from jarvis.llm import OllamaBackend

        mock_post.side_effect = requests.exceptions.Timeout("slow")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_generic_exception(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.side_effect = RuntimeError("unexpected")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.chat("any", [{"role": "user", "content": "hi"}]) is None

    @patch("jarvis.llm.requests.post")
    def test_extra_options_merge_into_payload(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})
        backend = OllamaBackend("http://localhost:11434")

        backend.chat(
            "any",
            [{"role": "user", "content": "hi"}],
            extra_options={"temperature": 0.5, "num_ctx": 16384},
        )

        sent = mock_post.call_args.kwargs["json"]
        # caller-supplied options merge over the default; both keys present
        assert sent["options"]["temperature"] == 0.5
        assert sent["options"]["num_ctx"] == 16384

    @patch("jarvis.llm.requests.post")
    def test_extra_options_none_keeps_defaults(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})
        backend = OllamaBackend("http://localhost:11434")

        backend.chat("any", [{"role": "user", "content": "hi"}], extra_options=None)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["options"] == {"num_ctx": 8192}


# ---------------------------------------------------------------------------
# OllamaBackend — direct edge cases
# ---------------------------------------------------------------------------


class TestOllamaBackendDirectEdgeCases:
    @patch("jarvis.llm.requests.post")
    def test_returns_none_for_unknown_response_shape(self, mock_post):
        """When the response carries no recognised content key, ``direct``
        falls through to the empty-content debug log path and returns None."""
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"unexpected": "shape"})
        backend = OllamaBackend("http://localhost:11434")

        assert backend.direct("gemma4:e2b", "sys", "user") is None

    @patch("jarvis.llm.requests.post")
    def test_temperature_forwarded_when_set(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})
        backend = OllamaBackend("http://localhost:11434")

        backend.direct("gemma4:e2b", "sys", "user", temperature=0.0)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["options"]["temperature"] == 0.0

    @patch("jarvis.llm.requests.post")
    def test_temperature_omitted_when_none(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})
        backend = OllamaBackend("http://localhost:11434")

        backend.direct("gemma4:e2b", "sys", "user")  # default temperature=None

        sent = mock_post.call_args.kwargs["json"]
        assert "temperature" not in sent["options"]


# ---------------------------------------------------------------------------
# OllamaBackend — streaming edge cases
# ---------------------------------------------------------------------------


class TestOllamaBackendStreamingEdgeCases:
    @patch("jarvis.llm.requests.post")
    def test_works_without_on_token_callback(self, mock_post):
        """``streaming`` must accumulate the full text even when the caller
        does not provide an ``on_token`` callback."""
        from jarvis.llm import OllamaBackend

        chunks = [
            json.dumps({"message": {"content": "a"}}).encode(),
            json.dumps({"message": {"content": "b"}}).encode(),
        ]
        mock_post.return_value = _make_response(iter_lines=chunks)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.streaming("gemma4:e2b", "sys", "user") == "ab"

    @patch("jarvis.llm.requests.post")
    def test_skips_lines_with_invalid_json(self, mock_post):
        """Malformed JSONL lines must be skipped silently rather than aborting
        the stream — Ollama occasionally interleaves keepalive frames."""
        from jarvis.llm import OllamaBackend

        chunks = [
            b"not-json",
            json.dumps({"message": {"content": "hi"}}).encode(),
            b"",  # blank line
        ]
        mock_post.return_value = _make_response(iter_lines=chunks)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.streaming("gemma4:e2b", "sys", "user") == "hi"


# ---------------------------------------------------------------------------
# extract_text_from_response — fallback shapes
# ---------------------------------------------------------------------------


class TestExtractTextFromResponse:
    """The helper handles Ollama's native shape and three OpenAI-compatible
    fallbacks so callers do not need to special-case proxied responses."""

    def test_ollama_message_content(self):
        from jarvis.llm import extract_text_from_response

        assert extract_text_from_response({"message": {"content": "hi"}}) == "hi"

    def test_openai_choices_message_content(self):
        from jarvis.llm import extract_text_from_response

        data = {"choices": [{"message": {"content": "hi"}}]}
        assert extract_text_from_response(data) == "hi"

    def test_openai_choices_text(self):
        from jarvis.llm import extract_text_from_response

        data = {"choices": [{"text": "hi"}]}
        assert extract_text_from_response(data) == "hi"

    def test_toplevel_content(self):
        from jarvis.llm import extract_text_from_response

        assert extract_text_from_response({"content": "hi"}) == "hi"

    def test_returns_none_for_unknown_shape(self):
        from jarvis.llm import extract_text_from_response

        assert extract_text_from_response({"unexpected": "shape"}) is None
        assert extract_text_from_response({"choices": []}) is None


# ---------------------------------------------------------------------------
# OllamaBackend — embeddings & model listing
# ---------------------------------------------------------------------------


class TestOllamaBackendEmbed:
    @patch("jarvis.llm.requests.post")
    def test_returns_vector(self, mock_post):
        from jarvis.llm import OllamaBackend

        resp = MagicMock()
        resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp
        backend = OllamaBackend("http://localhost:11434")

        vec = backend.embed("hello", "nomic-embed-text")

        assert vec == [0.1, 0.2, 0.3]

    @patch("jarvis.llm.requests.post")
    def test_returns_none_on_error(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.side_effect = RuntimeError("boom")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.embed("hello", "nomic-embed-text") is None


class TestOllamaBackendListModels:
    @patch("jarvis.llm.requests.get")
    def test_returns_model_names(self, mock_get):
        from jarvis.llm import OllamaBackend

        resp = MagicMock()
        resp.json.return_value = {
            "models": [
                {"name": "gemma4:e2b"},
                {"name": "gpt-oss:20b"},
            ]
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        backend = OllamaBackend("http://localhost:11434")

        assert backend.list_models() == ["gemma4:e2b", "gpt-oss:20b"]

    @patch("jarvis.llm.requests.get")
    def test_returns_empty_list_on_failure(self, mock_get):
        from jarvis.llm import OllamaBackend

        mock_get.side_effect = RuntimeError("boom")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.list_models() == []


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFactory:
    def test_get_llm_backend_returns_ollama_backend_for_default_settings(self, mock_config):
        from jarvis.llm import OllamaBackend, get_llm_backend

        backend = get_llm_backend(mock_config)

        assert isinstance(backend, OllamaBackend)


# ---------------------------------------------------------------------------
# Function-style entry points dispatch to the same backend
# ---------------------------------------------------------------------------


class TestFunctionStyleEntryPoints:
    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_returns_text(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_post.return_value = _make_response(json_data={"message": {"content": "hello"}})

        assert call_llm_direct("http://localhost:11434", "gemma4:e2b", "sys", "u") == "hello"

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_returns_dict(self, mock_post):
        from jarvis.llm import chat_with_messages

        mock_post.return_value = _make_response(json_data={"message": {"content": "ok"}})

        result = chat_with_messages(
            "http://localhost:11434", "gemma4:e2b", [{"role": "user", "content": "hi"}]
        )

        assert isinstance(result, dict)
        assert result["message"]["content"] == "ok"

    @patch("jarvis.llm.requests.post")
    def test_call_llm_streaming_invokes_callback(self, mock_post):
        from jarvis.llm import call_llm_streaming

        chunks = [json.dumps({"message": {"content": "x"}}).encode()]
        mock_post.return_value = _make_response(iter_lines=chunks)
        seen: list[str] = []

        result = call_llm_streaming(
            "http://localhost:11434", "gemma4:e2b", "sys", "u", on_token=seen.append
        )

        assert seen == ["x"]
        assert result == "x"

    def test_extract_text_from_response_importable(self):
        from jarvis.llm import extract_text_from_response

        assert extract_text_from_response({"message": {"content": "hi"}}) == "hi"
