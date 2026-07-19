"""Provider-aware model resolution for the auxiliary LLM contexts.

Every auxiliary context (intent judge, tool router, planner, evaluator,
loop digest) rides the chat provider's backend, so the model name it
resolves to must be one the chat provider actually serves. On the Ollama
path the small-model chain (gemma4:e2b etc.) is correct; on an
OpenAI-compatible chat provider those Ollama pull-names do not exist on
the user's server, so an unset aux model must resolve to the active chat
model instead. Explicit user overrides always win on both paths.

These tests pin the mechanism, not specific model names: they assert that
resolution follows the provider, not that any particular default string
is in place.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest


class _RecordingStub:
    """Minimal OpenAI-compatible chat stub that records the model of every
    inbound /chat/completions request."""

    def __init__(self, reply_text: str = "1. Reply to the user."):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                server.models_requested.append(req.get("model"))
                body = json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": server.reply_text},
                    "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.reply_text = reply_text
        self.models_requested: list = []
        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


def _load_settings_from(tmp_path, monkeypatch, cfg: dict):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"_config_version": 2, **cfg}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
    from jarvis.config import load_settings
    return load_settings()


class TestIntentJudgeModelResolution:
    """The judge model must exist on the chat provider's server."""

    def test_openai_chat_provider_defaults_judge_to_chat_model(self, tmp_path, monkeypatch):
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
        })
        assert settings.intent_judge_model == "qwen-27b"

    def test_openai_chat_provider_keeps_explicit_judge_override(self, tmp_path, monkeypatch):
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
            "intent_judge_model": "my-small-judge",
        })
        assert settings.intent_judge_model == "my-small-judge"

    def test_ollama_path_keeps_the_small_judge_default(self, tmp_path, monkeypatch):
        from jarvis.config import get_default_config
        settings = _load_settings_from(tmp_path, monkeypatch, {})
        assert settings.intent_judge_model == get_default_config()["intent_judge_model"]

    def test_explicit_empty_judge_value_counts_as_unset(self, tmp_path, monkeypatch):
        """An explicit empty string means "no judge model chosen", not "judge
        model is the empty string" — it resolves like an absent key."""
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
            "intent_judge_model": "",
        })
        assert settings.intent_judge_model == "qwen-27b"

    def test_ollama_chat_with_openai_embeddings_keeps_small_judge(self, tmp_path, monkeypatch):
        """The judge rides the CHAT provider; a remote embedding provider must
        not flip the judge off its small Ollama default."""
        from jarvis.config import get_default_config
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "embedding_provider": "openai_compatible",
            "embedding_base_url": "http://localhost:1234/v1",
            "embedding_model": "text-embedding-3-small",
        })
        assert settings.intent_judge_model == get_default_config()["intent_judge_model"]


class TestAuxChainsEndAtResolvedChatModel:
    """The router/planner/digest/evaluator chains must terminate at
    ``llm_chat_model`` (always the resolved active chat model on both
    providers), never at the Ollama-only field."""

    def _pure_openai_cfg(self):
        # Shaped like a pure OpenAI-compatible Settings: aux overrides unset,
        # judge already resolved to the chat model by config load.
        return SimpleNamespace(
            llm_provider="openai_compatible",
            llm_chat_model="qwen-27b",
            ollama_chat_model="gemma4:e2b",
            intent_judge_model="qwen-27b",
            tool_router_model="",
            evaluator_model="",
            planner_model="",
        )

    def test_tool_router_resolves_to_served_model(self):
        from jarvis.reply.engine import resolve_tool_router_model
        assert resolve_tool_router_model(self._pure_openai_cfg()) == "qwen-27b"

    def test_planner_resolves_to_served_model(self):
        from jarvis.reply.planner import resolve_planner_model
        assert resolve_planner_model(self._pure_openai_cfg()) == "qwen-27b"

    def test_loop_digest_resolves_to_served_model(self):
        from jarvis.reply.enrichment import _resolve_loop_digest_model
        assert _resolve_loop_digest_model(self._pure_openai_cfg()) == "qwen-27b"

    def test_evaluator_resolves_to_served_model(self):
        from jarvis.reply.evaluator import _resolve_evaluator_model
        assert _resolve_evaluator_model(self._pure_openai_cfg()) == "qwen-27b"

    def test_ollama_path_unchanged(self):
        """On the Ollama path llm_chat_model IS the resolved Ollama model, so
        the chains behave exactly as they always have."""
        from jarvis.reply.engine import resolve_tool_router_model
        from jarvis.reply.planner import resolve_planner_model
        cfg = SimpleNamespace(
            llm_provider="ollama",
            llm_chat_model="gemma4:e2b",   # config load: = ollama_chat_model
            ollama_chat_model="gemma4:e2b",
            intent_judge_model="gemma4:e2b",
            tool_router_model="",
            evaluator_model="",
            planner_model="",
        )
        assert resolve_tool_router_model(cfg) == "gemma4:e2b"
        assert resolve_planner_model(cfg) == "gemma4:e2b"


class TestPureOpenAIDispatchEndToEnd:
    """A pure OpenAI-compatible config, loaded through the real
    ``load_settings``, must drive the auxiliary contexts against the
    configured server with a model it actually serves — never an Ollama
    pull-name, never the Ollama port."""

    def test_planner_hits_openai_server_with_served_model(self, tmp_path, monkeypatch):
        stub = _RecordingStub(reply_text="1. Reply to the user.").start()
        try:
            settings = _load_settings_from(tmp_path, monkeypatch, {
                "llm_provider": "openai_compatible",
                "llm_base_url": stub.base_url,
                "llm_chat_model": "qwen-27b",
            })
            from jarvis.reply.planner import plan_query
            plan_query(settings, "what should I cook tonight with what I bought",
                       "", [("webSearch", "search the web")])
            assert stub.models_requested, "planner never reached the configured server"
            assert set(stub.models_requested) == {"qwen-27b"}
        finally:
            stub.stop()

    def test_graph_knowledge_extraction_hits_openai_server(self, tmp_path, monkeypatch):
        stub = _RecordingStub(reply_text="NONE").start()
        try:
            settings = _load_settings_from(tmp_path, monkeypatch, {
                "llm_provider": "openai_compatible",
                "llm_base_url": stub.base_url,
                "llm_chat_model": "qwen-27b",
            })
            from jarvis.memory import graph_ops
            graph_ops.call_llm_direct(
                cfg=settings, chat_model=settings.llm_chat_model,
                system_prompt="sys", user_content="usr", timeout_sec=5)
            assert stub.models_requested == ["qwen-27b"]
        finally:
            stub.stop()

    def test_evaluator_and_digest_resolve_served_model_from_real_settings(self, tmp_path, monkeypatch):
        """The evaluator and loop-digest chains, fed a real Settings from
        load_settings (not a hand-built cfg), land on the served model."""
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
        })
        from jarvis.reply.evaluator import _resolve_evaluator_model
        from jarvis.reply.enrichment import _resolve_loop_digest_model
        assert _resolve_evaluator_model(settings) == "qwen-27b"
        assert _resolve_loop_digest_model(settings) == "qwen-27b"

    def test_intent_judge_wiring_uses_served_model(self, tmp_path, monkeypatch):
        """The judge built from a pure OpenAI config must carry a model the
        server serves (the resolved chat model), not the Ollama judge
        default."""
        settings = _load_settings_from(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "qwen-27b",
        })
        from jarvis.listening.intent_judge import create_intent_judge
        judge = create_intent_judge(settings)
        assert judge is not None
        assert judge.config.model == "qwen-27b"
