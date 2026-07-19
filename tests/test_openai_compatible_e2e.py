"""End-to-end integration tests for the OpenAI-compatible provider path.

These stand up a minimal in-process HTTP server that speaks the OpenAI wire
shape and drive the REAL ``jarvis.llm`` backends + factory + config loader
through it over actual HTTP. This catches wire-shape regressions (request
payload, response normalisation, tool-call decoding, SSE streaming, auth
header, per-provider config resolution) without depending on a live vendor
server like LM Studio.

The stub is deliberately strict about the OpenAI shape: a backend change
that drifts from it fails here.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest


class _StubServer:
    """A threaded OpenAI-compatible stub. Records inbound requests so tests
    can assert on what the backend actually sent."""

    def __init__(self):
        self.calls: list[tuple[str, dict, dict]] = []  # (path, headers, body)
        server = self  # close over for the handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send({"data": [{"id": "stub-chat"}, {"id": "stub-embed"}]})
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                server.calls.append((self.path, dict(self.headers), req))

                if self.path.endswith("/chat/completions"):
                    if req.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for tok in ["Hel", "lo"]:
                            chunk = {"choices": [{"delta": {"content": tok}}]}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    if req.get("tools"):
                        # OpenAI returns tool-call arguments as a JSON *string*.
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "c1", "type": "function",
                                "function": {
                                    "name": "getWeather",
                                    "arguments": "{\"location\": \"London\"}",
                                },
                            }],
                        }
                    else:
                        msg = {"role": "assistant", "content": "Hello from stub"}
                    self._send({"choices": [{"message": msg, "finish_reason": "stop"}]})
                elif self.path.endswith("/embeddings"):
                    self._send({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def stub():
    server = _StubServer().start()
    try:
        yield server
    finally:
        server.stop()


def _cfg(stub, **over):
    base = dict(
        llm_provider="openai_compatible",
        llm_base_url=stub.base_url,
        llm_api_key="sk-stub",
        llm_chat_model="stub-chat",
        embedding_provider="",
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="stub-embed",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_chat_model="gemma4:e2b",
        ollama_embed_model="nomic-embed-text",
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestOpenAICompatibleEndToEnd:
    """Drive the real backend over HTTP against the stub."""

    def test_factory_dispatches_to_openai_backend_at_configured_url(self, stub):
        from jarvis.llm import get_llm_backend, OpenAICompatibleBackend
        backend = get_llm_backend(_cfg(stub))
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == stub.base_url

    def test_direct_returns_assistant_text(self, stub):
        from jarvis.llm import get_llm_backend
        out = get_llm_backend(_cfg(stub)).direct("stub-chat", "sys", "hi", timeout_sec=5)
        assert out == "Hello from stub"

    def test_sends_bearer_api_key(self, stub):
        from jarvis.llm import get_llm_backend
        get_llm_backend(_cfg(stub)).direct("stub-chat", "sys", "hi", timeout_sec=5)
        auths = [h.get("Authorization") for p, h, _ in stub.calls if p.endswith("/chat/completions")]
        assert auths and auths[0] == "Bearer sk-stub"

    def test_no_auth_header_when_key_empty(self, stub):
        from jarvis.llm import get_llm_backend
        get_llm_backend(_cfg(stub, llm_api_key="")).direct("stub-chat", "sys", "hi", timeout_sec=5)
        auths = [h.get("Authorization") for p, h, _ in stub.calls if p.endswith("/chat/completions")]
        assert auths and auths[0] is None

    def test_chat_normalises_message_to_top_level(self, stub):
        from jarvis.llm import get_llm_backend
        resp = get_llm_backend(_cfg(stub)).chat(
            "stub-chat", [{"role": "user", "content": "hi"}], timeout_sec=5)
        assert resp["message"]["content"] == "Hello from stub"

    def test_chat_decodes_tool_call_arguments_to_dict(self, stub):
        from jarvis.llm import get_llm_backend
        resp = get_llm_backend(_cfg(stub)).chat(
            "stub-chat", [{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {"name": "getWeather"}}],
            timeout_sec=5)
        tc = resp["message"]["tool_calls"][0]
        args = tc["function"]["arguments"]
        assert isinstance(args, dict) and args["location"] == "London"

    def test_streaming_reassembles_sse_and_fires_on_token(self, stub):
        from jarvis.llm import get_llm_backend
        toks: list[str] = []
        full = get_llm_backend(_cfg(stub)).streaming(
            "stub-chat", "sys", "hi", on_token=toks.append, timeout_sec=5)
        assert full == "Hello"
        assert toks == ["Hel", "lo"]

    def test_embeddings_via_inherited_provider(self, stub):
        from jarvis.llm import get_embedding_backend, OpenAICompatibleBackend
        eb = get_embedding_backend(_cfg(stub))  # embedding_provider="" inherits chat
        assert isinstance(eb, OpenAICompatibleBackend)
        assert eb.embed("hello", "stub-embed", timeout_sec=5) == [0.1, 0.2, 0.3]

    def test_list_models(self, stub):
        from jarvis.llm import get_llm_backend
        models = get_llm_backend(_cfg(stub)).list_models(timeout_sec=5)
        assert "stub-chat" in models and "stub-embed" in models


class TestConfigRoundTrip:
    """A config.json pointing at an OpenAI-compatible server must load and
    dispatch correctly through the real ``load_settings`` path."""

    def test_load_settings_to_backend_round_trip(self, stub, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "_config_version": 2,
            "llm_provider": "openai_compatible",
            "llm_base_url": stub.base_url,
            "llm_api_key": "sk-stub",
            "llm_chat_model": "stub-chat",
            "embedding_model": "stub-embed",
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        from jarvis.llm import get_llm_backend, OpenAICompatibleBackend

        settings = load_settings()
        assert settings.llm_provider == "openai_compatible"
        # Per-provider resolution: the OpenAI-compatible model wins on this path.
        assert settings.llm_chat_model == "stub-chat"
        assert settings.embedding_model == "stub-embed"

        backend = get_llm_backend(settings)
        assert isinstance(backend, OpenAICompatibleBackend)
        out = backend.direct(settings.llm_chat_model, "sys", "hi", timeout_sec=5)
        assert out == "Hello from stub"
