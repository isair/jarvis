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

import copy
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("owned_loopback_http")]


class _StubServer:
    """A threaded OpenAI-compatible stub. Records inbound requests so tests
    can assert on what the backend actually sent."""

    def __init__(self, *, tools=True, embeddings=True):
        self.tools = tools
        self.embeddings = embeddings
        self.calls: list[tuple[str, dict, dict]] = []  # (path, headers, body)
        server = self  # close over for the handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence
                pass

            def _send(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
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
                    if req.get("tools") and not server.tools:
                        self._send({"error": "tools not supported"}, status=400)
                        return
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
                    if not server.embeddings:
                        self._send({"error": "embeddings not supported"}, status=404)
                        return
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

    def test_decoded_response_can_be_sent_as_history_without_mutation(self, stub):
        from jarvis.llm import get_llm_backend
        backend = get_llm_backend(_cfg(stub))
        response = backend.chat("stub-chat", [{"role": "user", "content": "weather?"}],
                                tools=[{"type": "function", "function": {"name": "getWeather"}}],
                                timeout_sec=5)
        history = [response["message"], {"role": "tool", "tool_call_id": "c1", "content": "Synthetic forecast"}]
        saved_response = copy.deepcopy(response)
        saved_history = copy.deepcopy(history)
        backend.chat("stub-chat", history, timeout_sec=5)
        assert response == saved_response
        assert history == saved_history
        wire_arguments = stub.calls[-1][2]["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(wire_arguments) == saved_response["message"]["tool_calls"][0]["function"]["arguments"]

    def test_streaming_reassembles_sse_and_fires_on_token(self, stub):
        from jarvis.llm import get_llm_backend
        toks: list[str] = []
        full = get_llm_backend(_cfg(stub)).streaming(
            "stub-chat", "sys", "hi", on_token=toks.append, timeout_sec=5)
        assert full == "Hello"
        assert toks == ["Hel", "lo"]

    @pytest.mark.parametrize('keys,expected_authorisation', [
        ({'llm_api_key': 'synthetic-shared'}, 'Bearer synthetic-shared'),
        ({'llm_api_key': 'synthetic-shared', 'embedding_api_key': 'synthetic-embed'}, 'Bearer synthetic-embed'),
        ({'llm_api_key': '', 'embedding_api_key': ''}, None),
    ])
    def test_embeddings_use_inherited_provider_and_resolved_credentials(self, stub, keys, expected_authorisation):
        from jarvis.llm import get_embedding_backend, OpenAICompatibleBackend
        eb = get_embedding_backend(_cfg(stub, **keys))
        assert isinstance(eb, OpenAICompatibleBackend)
        assert eb.embed("hello", "stub-embed", timeout_sec=5) == [0.1, 0.2, 0.3]
        requests = [headers for path, headers, _ in stub.calls if path.endswith('/embeddings')]
        assert requests and all(headers.get('Authorization') == expected_authorisation for headers in requests)

    def test_list_models(self, stub):
        from jarvis.llm import get_llm_backend
        models = get_llm_backend(_cfg(stub)).list_models(timeout_sec=5)
        assert "stub-chat" in models and "stub-embed" in models


class TestCheckCapabilities:
    """``check_capabilities`` probes the real endpoints and reports honestly."""

    def test_full_featured_server(self, stub):
        from jarvis.llm import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend(stub.base_url, api_key="sk-stub")
        caps = backend.check_capabilities("stub-chat", "stub-embed", timeout_sec=5)
        assert caps.reachable and caps.chat and caps.tools and caps.embeddings
        assert "stub-chat" in caps.models

    def test_server_without_tools(self):
        from jarvis.llm import OpenAICompatibleBackend
        srv = _StubServer(tools=False).start()
        try:
            caps = OpenAICompatibleBackend(srv.base_url).check_capabilities(
                "stub-chat", "stub-embed", timeout_sec=3)
            assert caps.reachable and caps.chat and caps.embeddings
            assert caps.tools is False
        finally:
            srv.stop()

    def test_server_without_embeddings(self):
        from jarvis.llm import OpenAICompatibleBackend
        srv = _StubServer(embeddings=False).start()
        try:
            caps = OpenAICompatibleBackend(srv.base_url).check_capabilities(
                "stub-chat", "stub-embed", timeout_sec=3)
            assert caps.reachable and caps.chat and caps.tools
            assert caps.embeddings is False
        finally:
            srv.stop()

    def test_unreachable_server_is_not_reachable(self):
        from jarvis.llm import OpenAICompatibleBackend
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
            caps = OpenAICompatibleBackend(f"http://127.0.0.1:{port}/v1").check_capabilities(
                "x", timeout_sec=1)
        assert caps.reachable is False
        assert not (caps.chat or caps.tools or caps.embeddings)


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
