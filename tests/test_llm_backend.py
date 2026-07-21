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
# check_version — standalone Ollama identity probe
# ---------------------------------------------------------------------------


class TestCheckVersion:
    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_true_and_version_on_success(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(json_data={"version": "0.5.1"})
        ok, ver = check_version("http://localhost:11434")
        assert ok is True
        assert ver == "0.5.1"

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_on_connection_error(self, mock_get):
        from jarvis.llm import check_version

        mock_get.side_effect = RuntimeError("connection refused")
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_on_non_200_status(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(status_code=404)
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_when_json_missing_version_key(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(json_data={"not_version": "garbage"})
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_when_response_not_json(self, mock_get):
        from jarvis.llm import check_version

        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        mock_get.return_value = resp
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_when_version_is_empty_string(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(json_data={"version": ""})
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_returns_false_when_version_is_not_a_string(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(json_data={"version": 42})
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None

    @patch("jarvis.llm.ollama.requests.get")
    def test_response_is_not_a_dict(self, mock_get):
        from jarvis.llm import check_version

        mock_get.return_value = _make_response(json_data=[1, 2, 3])
        ok, ver = check_version("http://localhost:11434")
        assert ok is False
        assert ver is None


# ---------------------------------------------------------------------------
# OllamaBackend — warm_up (version check + chat ping)
# ---------------------------------------------------------------------------


class TestOllamaBackendWarmUp:
    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_succeeds_when_server_is_ollama(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.return_value = _make_response(json_data={"version": "0.5.1"})
        mock_post.return_value = _make_response(status_code=200)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is True
        get_url = mock_get.call_args[0][0]
        assert "api/version" in get_url
        post_url = mock_post.call_args[0][0]
        assert "api/chat" in post_url

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_fails_when_version_endpoint_unreachable(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.side_effect = RuntimeError("connection refused")
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_fails_when_version_response_not_200(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.return_value = _make_response(status_code=404)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_fails_when_version_json_missing_version_key(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.return_value = _make_response(json_data={"not_version": "garbage"})
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_fails_when_version_response_not_json(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        mock_get.return_value = resp
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_fails_when_chat_fails_after_version_check(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.return_value = _make_response(json_data={"version": "0.5.1"})
        mock_post.return_value = _make_response(status_code=500)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_sends_chat_request_with_correct_payload(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        mock_get.return_value = _make_response(json_data={"version": "0.5.1"})
        mock_post.return_value = _make_response(status_code=200)
        backend = OllamaBackend("http://localhost:11434")

        assert backend.warm_up("gemma4:e2b") is True
        post_body = mock_post.call_args[1]["json"]
        assert post_body["model"] == "gemma4:e2b"
        assert post_body["keep_alive"] == "30m"
        assert post_body["stream"] is False
        # Must use chat endpoint with messages to exercise full inference pipeline
        assert "messages" in post_body
        assert len(post_body["messages"]) == 2
        assert post_body["messages"][0]["role"] == "system"
        assert post_body["messages"][1]["role"] == "user"
        assert post_body["options"]["num_predict"] == 1

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_returns_false_for_empty_model(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        backend = OllamaBackend("http://localhost:11434")
        assert backend.warm_up("") is False
        assert backend.warm_up(None) is False

    @patch("jarvis.llm.requests.post")
    @patch("jarvis.llm.requests.get")
    def test_warmup_returns_false_for_empty_base_url(self, mock_get, mock_post):
        from jarvis.llm import OllamaBackend

        backend = OllamaBackend("")
        assert backend.warm_up("gemma4:e2b") is False
        assert backend.warm_up("gemma4:e2b") is False


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


class TestStripNonstandardMessageFields:
    """``strip_nonstandard_message_fields`` strips engine-internal annotation
    fields from each message, keeping only the subset allowed by the OpenAI
    Chat Completions schema for that role."""

    def test_removes_tool_failed_from_tool_message(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "sunny, 22°C",
                "tool_name": "getWeather",
                "tool_failed": False,
            }
        ]
        result = strip_nonstandard_message_fields(messages)

        assert result[0] == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "sunny, 22°C",
        }

    def test_removes_tool_name_and_is_context_injected_from_system(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "_is_context_injected": True,
            }
        ]
        result = strip_nonstandard_message_fields(messages)

        assert result[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }

    def test_preserves_allowed_assistant_fields(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "getWeather"}}],
                "unknown_field": "should_be_removed",
            }
        ]
        result = strip_nonstandard_message_fields(messages)

        assert "unknown_field" not in result[0]
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert len(result[0]["tool_calls"]) == 1

    def test_preserves_user_content(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [{"role": "user", "content": "How's the weather?", "tool_name": "getWeather"}]
        result = strip_nonstandard_message_fields(messages)

        assert result[0] == {"role": "user", "content": "How's the weather?"}

    def test_preserves_unknown_role_unchanged(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [{"role": "developer", "content": "some instruction", "custom": True}]
        result = strip_nonstandard_message_fields(messages)

        assert result[0] == messages[0]

    def test_original_messages_not_mutated(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        original = [
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "data",
                "tool_name": "getWeather",
                "tool_failed": False,
            }
        ]
        original_copy = [dict(m) for m in original]

        strip_nonstandard_message_fields(original)

        assert original == original_copy

    def test_multiple_messages_in_sequence(self):
        from jarvis.llm.backend import strip_nonstandard_message_fields

        messages = [
            {"role": "system", "content": "Be helpful.", "_is_context_injected": True},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "x"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "result",
                "tool_name": "x",
                "tool_failed": False,
            },
        ]
        result = strip_nonstandard_message_fields(messages)

        assert len(result) == 4
        assert "_is_context_injected" not in result[0]
        assert "tool_name" not in result[3]
        assert "tool_failed" not in result[3]
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "c1"
        assert result[3]["content"] == "result"


class TestOllamaBackendSanitizesMessages:
    """OllamaBackend.chat() strips non-standard fields from messages before
    sending them to the wire."""

    @patch("jarvis.llm.ollama.requests.post")
    def test_tool_name_not_in_wire_payload(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(
            json_data={"message": {"content": "done", "tool_calls": []}}
        )
        backend = OllamaBackend("http://localhost:11434")

        backend.chat(
            "gemma4:e2b",
            [
                {"role": "user", "content": "How's the weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "getWeather"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "1",
                    "content": "Error: no location",
                    "tool_name": "getWeather",
                    "tool_failed": True,
                },
            ],
        )

        sent = mock_post.call_args.kwargs["json"]
        tool_msg = sent["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "1"
        assert tool_msg["content"] == "Error: no location"
        assert "tool_name" not in tool_msg, "tool_name field must be stripped"
        assert "tool_failed" not in tool_msg, "tool_failed field must be stripped"

    @patch("jarvis.llm.ollama.requests.post")
    def test_system_is_context_injected_stripped(self, mock_post):
        from jarvis.llm import OllamaBackend

        mock_post.return_value = _make_response(
            json_data={"message": {"content": "hello", "tool_calls": []}}
        )
        backend = OllamaBackend("http://localhost:11434")

        backend.chat(
            "gemma4:e2b",
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                    "_is_context_injected": True,
                },
                {"role": "user", "content": "Hi"},
            ],
        )

        sent = mock_post.call_args.kwargs["json"]
        sys_msg = sent["messages"][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"] == "You are a helpful assistant."
        assert "_is_context_injected" not in sys_msg
