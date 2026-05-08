"""Behaviour tests for the LLM factory dispatch.

Covers ``get_llm_backend`` and ``get_embedding_backend`` selection
across provider configurations: Ollama default, OpenAI-compatible,
embedding override to Ollama when chat runs on a runtime without
embeddings, and back-fill from the legacy ``ollama_*`` fields when
the new ``llm_*`` fields are unset.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _Cfg:
    llm_provider: str = "ollama"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_chat_model: str = ""
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "gemma4:e2b"
    ollama_embed_model: str = "nomic-embed-text"


class TestGetLLMBackend:
    def test_returns_ollama_for_default_provider(self):
        from jarvis.llm import OllamaBackend, get_llm_backend

        backend = get_llm_backend(_Cfg())

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_returns_openai_compatible_when_provider_set(self):
        from jarvis.llm import OpenAICompatibleBackend, get_llm_backend

        cfg = _Cfg(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="sk-test",
        )

        backend = get_llm_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://localhost:1234/v1"

    def test_falls_back_to_ollama_for_unknown_provider(self):
        from jarvis.llm import OllamaBackend, get_llm_backend

        cfg = _Cfg(llm_provider="lm-studio")  # unknown alias

        backend = get_llm_backend(cfg)

        assert isinstance(backend, OllamaBackend)

    def test_uses_ollama_base_url_when_llm_base_url_empty(self):
        from jarvis.llm import get_llm_backend

        cfg = _Cfg(ollama_base_url="http://1.2.3.4:11434")

        backend = get_llm_backend(cfg)

        assert backend.base_url == "http://1.2.3.4:11434"


class TestGetEmbeddingBackend:
    def test_defaults_to_llm_provider(self):
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:1234/v1",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://localhost:1234/v1"

    def test_override_to_ollama_when_chat_runs_on_openai_compatible(self):
        """The override exists for runtimes that ship chat without
        embeddings (e.g. some oMLX builds). The user pins embeddings to
        Ollama; chat keeps running on the configured provider."""
        from jarvis.llm import OllamaBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:1234/v1",
            embedding_provider="ollama",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_explicit_embedding_base_url_wins(self):
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="ollama",
            embedding_provider="openai_compatible",
            embedding_base_url="http://embed-host:9000/v1",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://embed-host:9000/v1"


class TestConfigMigration:
    def test_v1_config_promotes_ollama_fields_to_new_keys(self, tmp_path, monkeypatch):
        """A pre-PR-2 (v1) config on disk should auto-fill the new
        ``llm_*`` and ``embedding_*`` keys from the matching
        ``ollama_*`` ones so existing installs keep working."""
        import json

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "_config_version": 1,
                    "ollama_base_url": "http://1.2.3.4:11434",
                    "ollama_chat_model": "my-model",
                    "ollama_embed_model": "my-embed",
                }
            )
        )
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        # Reload the module so cached config-path defaults reset.
        from jarvis.config import load_settings

        settings = load_settings()

        assert settings.llm_provider == "ollama"
        assert settings.llm_base_url == "http://1.2.3.4:11434"
        assert settings.llm_chat_model == "my-model"
        assert settings.embedding_model == "my-embed"
        # Old fields stay populated for compatibility.
        assert settings.ollama_base_url == "http://1.2.3.4:11434"
        # Migration is persisted to disk.
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["_config_version"] == 2
        assert on_disk["llm_provider"] == "ollama"
        assert on_disk["llm_base_url"] == "http://1.2.3.4:11434"
