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

    def test_ollama_provider_ignores_stale_llm_base_url(self):
        """``llm_base_url`` is the OpenAI-compatible server's URL. When the
        provider is Ollama, the backend must use ``ollama_base_url`` and
        ignore any ``llm_base_url`` left over from a previous
        OpenAI-compatible configuration — otherwise toggling the provider
        back to Ollama would silently point OllamaBackend at the old
        LM Studio URL."""
        from jarvis.llm import OllamaBackend, get_llm_backend

        cfg = _Cfg(
            llm_provider="ollama",
            llm_base_url="http://lmstudio:1234/v1",  # stale from a prior switch
            ollama_base_url="http://127.0.0.1:11434",
        )

        backend = get_llm_backend(cfg)

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"


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

    def test_inherits_llm_base_url_when_embedding_provider_unset(self):
        """Most common LM Studio config: chat and embeddings on the same
        OpenAI-compatible server. With ``embedding_provider`` unset and
        ``embedding_base_url`` empty, the embedding backend must inherit
        ``llm_base_url`` rather than falling back to the Ollama URL."""
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:1234/v1",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://localhost:1234/v1"

    def test_inherits_llm_api_key_when_embedding_api_key_unset(self):
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="sk-shared",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        # The api_key isn't a public attribute; inspect the bearer
        # header through a probe call instead.
        assert backend._api_key == "sk-shared"

    def test_falls_back_to_default_when_provider_openai_but_no_url(self):
        """When the user picks openai_compatible but provides no URL on
        any of llm_base_url, ollama_base_url, embedding_base_url, the
        factory falls back to the Ollama default rather than raising.
        Construction is fail-soft; the request will fail at call time."""
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        cfg = _Cfg(
            llm_provider="ollama",
            embedding_provider="openai_compatible",
        )

        backend = get_embedding_backend(cfg)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://127.0.0.1:11434"


class TestPerProviderModelResolution:
    """``cfg.llm_chat_model`` / ``cfg.embedding_model`` resolve per-provider:
    the Ollama models are authoritative on the Ollama path, the
    OpenAI-compatible models on that path. This mirrors the factory's
    per-provider base-URL resolution and stops a stale ``llm_chat_model``
    (e.g. promoted by the v2 migration) from shadowing the Ollama model
    picker, which writes ``ollama_chat_model``."""

    def _load(self, tmp_path, monkeypatch, cfg: dict):
        import json
        cfg_path = tmp_path / "config.json"
        cfg.setdefault("_config_version", 2)  # skip migration noise
        cfg_path.write_text(json.dumps(cfg))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        from jarvis.config import load_settings
        return load_settings()

    def test_ollama_chat_model_not_shadowed_by_stale_llm_chat_model(self, tmp_path, monkeypatch):
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "ollama_chat_model": "new-pick:7b",
            "llm_chat_model": "stale-migrated:70b",
        })
        assert settings.llm_chat_model == "new-pick:7b"

    def test_openai_compatible_uses_llm_chat_model(self, tmp_path, monkeypatch):
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_chat_model": "lmstudio/gemma",
            "ollama_chat_model": "gemma4:e2b",
        })
        assert settings.llm_chat_model == "lmstudio/gemma"

    def test_openai_compatible_falls_back_to_ollama_model_when_unset(self, tmp_path, monkeypatch):
        """If the user picked openai_compatible but left the model blank,
        fall back to the Ollama model name rather than the empty string."""
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://localhost:1234/v1",
            "ollama_chat_model": "gemma4:e2b",
        })
        assert settings.llm_chat_model == "gemma4:e2b"

    def test_embedding_model_not_shadowed_on_ollama_path(self, tmp_path, monkeypatch):
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "ollama_embed_model": "nomic-embed-text",
            "embedding_model": "stale-openai-embed",
        })
        assert settings.embedding_model == "nomic-embed-text"

    def test_embedding_model_used_when_embedding_provider_openai(self, tmp_path, monkeypatch):
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "embedding_provider": "openai_compatible",
            "embedding_base_url": "http://embed:9000/v1",
            "embedding_model": "text-embedding-3-small",
            "ollama_embed_model": "nomic-embed-text",
        })
        assert settings.embedding_model == "text-embedding-3-small"


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

    def test_v1_to_v2_preserves_explicit_llm_fields(self, tmp_path, monkeypatch):
        """A user who has already opted into a non-Ollama provider
        before the migration runs (e.g. by hand-editing config) must
        not have their choice reverted by the v1→v2 promotion."""
        import json

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "_config_version": 1,
                    "llm_provider": "openai_compatible",
                    "llm_base_url": "http://lmstudio:1234/v1",
                    "llm_chat_model": "lmstudio-community/gemma",
                    "ollama_base_url": "http://127.0.0.1:11434",
                    "ollama_chat_model": "gemma4:e2b",
                }
            )
        )
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings

        settings = load_settings()

        assert settings.llm_provider == "openai_compatible"
        assert settings.llm_base_url == "http://lmstudio:1234/v1"
        assert settings.llm_chat_model == "lmstudio-community/gemma"

    def test_v1_to_v2_fresh_install_with_no_ollama_fields(self, tmp_path, monkeypatch):
        """A v1 config that never carried ``ollama_*`` keys should
        upgrade to v2 cleanly and fall through to defaults — no crash
        on missing keys."""
        import json

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"_config_version": 1}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings

        settings = load_settings()

        assert settings.llm_provider == "ollama"
        # Defaults flow through when nothing was explicitly set.
        assert settings.llm_base_url == settings.ollama_base_url
        # Migration runs without touching keys that have no source.
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["_config_version"] == 2
        assert on_disk["llm_provider"] == "ollama"
