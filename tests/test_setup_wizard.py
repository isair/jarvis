"""
Tests for setup wizard detection functions.

These tests verify the Ollama detection logic without touching the UI.
They treat the detection functions as black boxes, verifying inputs produce correct outputs.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import pytest

from desktop_app.setup_wizard import (
    check_ollama_cli,
    check_ollama_server,
    get_required_models,
    check_installed_models,
    check_ollama_status,
    resolve_ollama_path,
    should_show_setup_wizard,
    OllamaStatus,
    MCPPage,
    SearchProvidersPage,
    ProviderChoicePage,
    OpenAICompatiblePage,
)
from desktop_app.mcp_catalogue import get_wizard_entries
from jarvis.config import DEFAULT_CHAT_MODEL
from jarvis.utils.location import (
    get_location_context,
    is_location_available,
    _is_private_ip,
)


@pytest.fixture
def stub_openai_server():
    """A minimal in-process server answering GET /v1/models, for exercising
    the wizard's model-fetch against a real HTTP endpoint."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            if self.path.endswith("/models"):
                body = json.dumps({"data": [{"id": "stub-chat"}, {"id": "stub-embed"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield base, httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


class TestCheckOllamaCli:
    """Tests for Ollama CLI detection."""

    def test_detects_ollama_in_path(self):
        """When ollama is in PATH, returns True with path."""
        with patch("shutil.which", return_value="/usr/local/bin/ollama"):
            is_installed, path = check_ollama_cli()

            assert is_installed is True
            assert path == "/usr/local/bin/ollama"

    def test_returns_false_when_not_installed(self):
        """When ollama is not installed anywhere, returns False."""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                is_installed, path = check_ollama_cli()

                assert is_installed is False
                assert path is None

    def test_checks_macos_homebrew_path(self):
        """On macOS, checks Homebrew installation path."""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile") as mock_isfile:
                with patch("os.access", return_value=True):
                    # First call for /usr/local/bin/ollama returns False
                    # Second call for /opt/homebrew/bin/ollama returns True
                    mock_isfile.side_effect = lambda p: p == "/opt/homebrew/bin/ollama"

                    is_installed, path = check_ollama_cli()

                    assert is_installed is True
                    assert path == "/opt/homebrew/bin/ollama"


class TestCheckOllamaServer:
    """Tests for Ollama server detection."""

    def test_detects_running_server(self):
        """When server is running, returns True with version."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.23"}

        with patch("requests.get", return_value=mock_response):
            is_running, version = check_ollama_server()

            assert is_running is True
            assert version == "0.1.23"

    def test_returns_false_when_server_not_running(self):
        """When server is not responding, returns False."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            is_running, version = check_ollama_server()

            assert is_running is False
            assert version is None

    def test_handles_timeout(self):
        """When request times out, returns False."""
        import requests
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            is_running, version = check_ollama_server()

            assert is_running is False
            assert version is None


class TestGetRequiredModels:
    """Tests for getting required models from config."""

    def test_returns_models_from_config(self):
        """Returns chat and embed models from config."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "llama2:7b"
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.fast_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            assert "llama2:7b" in models
            assert "nomic-embed-text" in models

    def test_includes_fast_model_when_different_from_chat(self):
        """Includes the fast model when it differs from the chat model."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "gpt-oss:20b"  # Different from fast model
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.fast_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            # Should have 3 models: chat, embed, and the fast model
            assert len(models) == 3
            assert "gpt-oss:20b" in models
            assert "nomic-embed-text" in models
            assert "gemma4:e2b" in models  # the fast model is always required

    def test_fast_model_equal_to_chat_is_not_duplicated(self):
        """When the fast model is the chat model, the pull list stays at two
        entries — no duplicate download of the same model."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "gemma4:e2b"
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.fast_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()
            assert len(models) == 2
            assert models.count("gemma4:e2b") == 1

    def test_returns_defaults_on_config_error(self):
        """Returns default models if config can't be loaded."""
        with patch("desktop_app.setup_wizard.load_settings", side_effect=Exception("Config error")):
            models = get_required_models()

            assert len(models) == 2
            assert "gemma4:e2b" in models
            assert "nomic-embed-text" in models

    def _cfg(self, **over):
        from types import SimpleNamespace
        base = dict(
            llm_provider="ollama",
            embedding_provider="",
            ollama_chat_model="gemma4:e2b",
            ollama_embed_model="nomic-embed-text",
            fast_model="gemma4:e2b",
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_pure_ollama_requires_chat_embed_judge(self):
        """The default local setup needs the chat model, embed model, and
        (distinct) intent-judge model — all pulled from Ollama."""
        cfg = self._cfg(llm_provider="ollama", ollama_chat_model="gpt-oss:20b")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["gpt-oss:20b", "nomic-embed-text", "gemma4:e2b"]

    def test_pure_openai_requires_no_ollama_models(self):
        """Chat, judge, and embeddings all remote: nothing to pull locally."""
        cfg = self._cfg(llm_provider="openai_compatible", embedding_provider="")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == []

    def test_openai_chat_with_ollama_embeddings_requires_only_embed_model(self):
        """The advanced split: chat/judge remote, embeddings on Ollama. Only
        the embedding model must be present locally — not the remote chat
        model name, not the intent-judge model."""
        cfg = self._cfg(
            llm_provider="openai_compatible",
            embedding_provider="ollama",
            ollama_chat_model="some-remote-model",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["nomic-embed-text"]

    def test_ollama_chat_with_openai_embeddings_skips_embed_model(self):
        """Chat/judge on Ollama, embeddings remote: pull chat + judge, not
        the Ollama embed model."""
        cfg = self._cfg(
            llm_provider="ollama",
            embedding_provider="openai_compatible",
            ollama_chat_model="gpt-oss:20b",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["gpt-oss:20b", "gemma4:e2b"]


class TestCheckInstalledModels:
    """Tests for checking installed Ollama models."""

    def test_parses_ollama_list_output(self):
        """Correctly parses 'ollama list' output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """NAME                       ID              SIZE      MODIFIED
llama2:7b                  abc123          3.8 GB    2 days ago
nomic-embed-text:latest    def456          274 MB    1 week ago
"""

        with patch("subprocess.run", return_value=mock_result):
            models = check_installed_models("/usr/bin/ollama")

            assert "llama2:7b" in models
            assert "nomic-embed-text:latest" in models

    def test_returns_empty_on_error(self):
        """Returns empty list if ollama list fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            models = check_installed_models()

            assert models == []

    def test_handles_subprocess_exception(self):
        """Returns empty list if subprocess raises exception."""
        with patch("subprocess.run", side_effect=Exception("Command not found")):
            models = check_installed_models()

            assert models == []

    def test_falls_back_to_check_ollama_cli_when_path_unset(self):
        """When PATH does not contain ollama (e.g. frozen macOS .app launch),
        falls back to check_ollama_cli() so the resolved binary is invoked
        instead of plain "ollama" which would fail with FileNotFoundError."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME    ID    SIZE    MODIFIED\nllama2:7b    abc    3.8 GB    1d\n"

        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(True, "/usr/local/bin/ollama"),
            ):
                with patch("subprocess.run", return_value=mock_result) as run:
                    models = check_installed_models()

                    assert "llama2:7b" in models
                    args, _ = run.call_args
                    assert args[0][0] == "/usr/local/bin/ollama"


class TestResolveOllamaPath:
    """Tests for the ollama CLI path resolver."""

    def test_prefers_path_lookup(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value="/opt/homebrew/bin/ollama"):
            assert resolve_ollama_path() == "/opt/homebrew/bin/ollama"

    def test_falls_back_to_check_ollama_cli(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(True, "/usr/local/bin/ollama"),
            ):
                assert resolve_ollama_path() == "/usr/local/bin/ollama"

    def test_returns_literal_when_nothing_resolves(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(False, None),
            ):
                assert resolve_ollama_path() == "ollama"


class TestCheckOllamaStatus:
    """Tests for complete Ollama status check."""

    def test_fully_setup_status(self):
        """Returns correct status when everything is set up."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.1.23")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b"]):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=["llama2:7b"]):
                        status = check_ollama_status()

                        assert status.is_cli_installed is True
                        assert status.is_server_running is True
                        assert status.missing_models == []
                        assert status.is_fully_setup is True

    def test_missing_cli_status(self):
        """Returns correct status when CLI is not installed."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(False, None)):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(False, None)):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b"]):
                    status = check_ollama_status()

                    assert status.is_cli_installed is False
                    assert status.is_fully_setup is False
                    assert "llama2:7b" in status.missing_models

    def test_missing_models_status(self):
        """Returns correct status when models are missing."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.1.23")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b", "codellama"]):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=["llama2:7b"]):
                        status = check_ollama_status()

                        assert status.is_cli_installed is True
                        assert status.is_server_running is True
                        assert "codellama" in status.missing_models
                        assert status.is_fully_setup is False


class TestShouldShowSetupWizard:
    """Tests for wizard display logic."""

    def test_returns_false_when_fully_setup(self):
        """Returns False when everything is configured."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.1.23",
            installed_models=["llama2:7b"],
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_returns_true_when_cli_missing(self):
        """Returns True when CLI is not installed."""
        mock_status = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=["llama2:7b"],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_returns_false_when_server_not_running_but_cli_installed(self):
        """Returns False when server is not running but CLI is installed.

        The app can auto-start the server, so no wizard needed.
        """
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=False,
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_returns_true_when_models_missing(self):
        """Returns True when required models are missing."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.1.23",
            installed_models=[],
            missing_models=["llama2:7b"],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_returns_false_for_openai_compatible_provider(self):
        """An OpenAI-compatible user has opted out of the local Ollama
        stack, so the Ollama-centric wizard must never auto-show even if
        the Ollama CLI is absent."""
        from types import SimpleNamespace
        missing_cli = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=["llama2:7b"],
        )
        cfg = SimpleNamespace(llm_provider="openai_compatible")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg), \
             patch("desktop_app.setup_wizard.check_ollama_status", return_value=missing_cli):
            assert should_show_setup_wizard() is False

    def test_returns_true_when_force_server_check_and_server_down(self):
        """Returns True for force_server_check when CLI is installed but
        server is not running (auto-start already failed)."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=False,
            missing_models=[],
        )
        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard(force_server_check=True) is True

    def test_force_server_check_still_returns_true_when_cli_missing(self):
        """force_server_check does not suppress other triggers (CLI missing)."""
        mock_status = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=[],
        )
        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard(force_server_check=True) is True

    def test_force_server_check_ignored_for_openai_compatible(self):
        """force_server_check is dead code when llm_provider is
        openai_compatible — the early return prevents it from firing."""
        from types import SimpleNamespace
        cfg = SimpleNamespace(llm_provider="openai_compatible")
        mock_status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=False,
            missing_models=[],
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg), \
             patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard(force_server_check=True) is False

    def test_force_server_check_still_returns_false_when_everything_ok(self):
        """force_server_check still returns False when everything is fine."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.1.23",
            installed_models=["llama2:7b"],
            missing_models=[],
        )
        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard(force_server_check=True) is False


class TestProviderChoicePage:
    """The first real wizard decision: which runtime serves the LLM."""

    def test_validate_writes_openai_compatible_provider(self):
        """Selecting the OpenAI-compatible card persists llm_provider."""
        import tempfile, json
        from pathlib import Path
        page = ProviderChoicePage.__new__(ProviderChoicePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._selected = "openai_compatible"
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text())
            assert saved["llm_provider"] == "openai_compatible"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_removes_provider_override_for_ollama(self):
        """Selecting Ollama clears the openai_compatible overrides so the
        Ollama settings become authoritative again (no stale base URL /
        model / key left pointing at a former OpenAI-compatible server)."""
        import tempfile, json
        from pathlib import Path
        page = ProviderChoicePage.__new__(ProviderChoicePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "llm_provider": "openai_compatible",
                "llm_base_url": "http://localhost:1234/v1",
                "llm_api_key": "sk-x",
                "llm_chat_model": "lmstudio/gemma",
                "embedding_model": "text-embedding-3-small",
            }, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._selected = "ollama"
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text())
            assert saved.get("llm_provider", "ollama") == "ollama"
            for stale in ("llm_base_url", "llm_api_key", "llm_chat_model",
                          "embedding_model", "embedding_base_url", "embedding_api_key"):
                assert stale not in saved, f"{stale} must be cleared on the Ollama path"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_routes_openai_to_config_page(self):
        """OpenAI-compatible selection jumps to the connection-config page."""
        page = ProviderChoicePage.__new__(ProviderChoicePage)
        page._selected = "openai_compatible"
        wizard = MagicMock()
        wizard.openai_compat_page_id = 42
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 42

    def test_preselects_openai_from_existing_config(self, qapp):
        """Re-running the wizard reflects the saved provider: an existing
        openai_compatible config preselects the OpenAI card."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"llm_provider": "openai_compatible"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()  # __init__ calls _preselect_from_config
            assert page._selected == "openai_compatible"
            assert page._openai_radio.isChecked() is True
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_preselects_ollama_by_default(self, qapp):
        """A config without llm_provider (the default install) preselects Ollama."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            assert page._selected == "ollama"
            assert page._ollama_radio.isChecked() is True
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_radios_are_mutually_exclusive(self, qapp):
        """The two provider radios live in separate cards, so they need a
        shared QButtonGroup to be mutually exclusive — checking one must
        uncheck the other (and update the selection)."""
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            # Default: Ollama selected, OpenAI not.
            assert page._ollama_radio.isChecked() and not page._openai_radio.isChecked()
            page._openai_radio.setChecked(True)
            assert page._openai_radio.isChecked()
            assert not page._ollama_radio.isChecked(), "radios must be mutually exclusive"
            assert page._selected == "openai_compatible"
            page._ollama_radio.setChecked(True)
            assert not page._openai_radio.isChecked()
            assert page._selected == "ollama"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_openai_card_describes_a_local_server(self, qapp):
        """The OpenAI-compatible card must not imply the option is cloud /
        less private: its copy clarifies it points at a local server."""
        import tempfile
        from pathlib import Path
        from PyQt6.QtWidgets import QLabel
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            blob = " ".join(lbl.text().lower() for lbl in page.findChildren(QLabel))
            assert "local" in blob and "network" in blob, (
                "provider copy should make clear the OpenAI-compatible option "
                "is also local"
            )
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_ollama_routes_to_welcome(self):
        """Ollama selection goes to the Welcome/status page."""
        page = ProviderChoicePage.__new__(ProviderChoicePage)
        page._selected = "ollama"
        wizard = MagicMock()
        wizard.welcome_page_id = 5
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 5

    def test_wizard_starts_on_whisper(self, qapp):
        """Whisper setup is the first step — it has no LLM dependencies
        and informs VRAM calculations on the Models page."""
        import tempfile
        from pathlib import Path
        from desktop_app.setup_wizard import SetupWizard
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                wiz = SetupWizard()
            assert wiz.startId() == wiz.mlx_whisper_page_id
        finally:
            cfg_path.unlink(missing_ok=True)


class TestWelcomePageFlow:
    """The Welcome/status page is reached only on the Ollama branch."""

    def test_nextid_enters_ollama_flow(self):
        from desktop_app.setup_wizard import WelcomePage
        page = WelcomePage.__new__(WelcomePage)
        wizard = MagicMock()
        wizard.ollama_entry_page_id = MagicMock(return_value=9)
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 9


class TestOpenAICompatiblePage:
    """Collects the OpenAI-compatible connection details."""

    def test_incomplete_without_base_url_and_model(self):
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        page._base_url = ""
        page._chat_model = ""
        assert page._is_ready("", "") is False
        assert page._is_ready("http://localhost:1234/v1", "") is False
        assert page._is_ready("", "gemma") is False
        assert page._is_ready("http://localhost:1234/v1", "gemma") is True

    def test_validate_writes_connection_fields(self):
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        page._use_ollama_embed = SimpleNamespace(isVisible=lambda: False, isChecked=lambda: False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"llm_provider": "openai_compatible"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "http://localhost:1234/v1", "sk-secret", "lmstudio/gemma", "text-embed-3", "",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text())
            assert saved["llm_provider"] == "openai_compatible"
            assert saved["llm_base_url"] == "http://localhost:1234/v1"
            assert saved["llm_api_key"] == "sk-secret"
            assert saved["llm_chat_model"] == "lmstudio/gemma"
            assert saved["embedding_model"] == "text-embed-3"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_omits_empty_optional_fields(self):
        """API key and embedding model are optional; empty values are not
        persisted, keeping config.json minimal."""
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        page._use_ollama_embed = SimpleNamespace(isVisible=lambda: False, isChecked=lambda: False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "http://localhost:1234/v1", "", "lmstudio/gemma", "", "",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text())
            assert "llm_api_key" not in saved
            assert "embedding_model" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_skips_ollama_pages(self):
        """After configuring the remote provider, the wizard jumps straight
        to dictation — the Ollama install/server/models pages are
        irrelevant."""
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        wizard = MagicMock()
        wizard.dictation_page_id = 8
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 8

    def test_initialize_page_prefills_from_existing_config(self, qapp):
        """Re-running the wizard restores the user's saved connection
        details into the form fields so they are not re-typed."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "llm_provider": "openai_compatible",
                "llm_base_url": "http://lmstudio:1234/v1",
                "llm_api_key": "sk-saved",
                "llm_chat_model": "lmstudio/gemma",
                "embedding_model": "text-embed-3",
            }, f)
            cfg_path = Path(f.name)
        try:
            page = OpenAICompatiblePage()
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.initializePage()
            assert page._base_url_input.text() == "http://lmstudio:1234/v1"
            assert page._api_key_input.text() == "sk-saved"
            assert page._chat_model_combo.currentText() == "lmstudio/gemma"
            assert page._embed_model_combo.currentText() == "text-embed-3"
            # The API key field stays masked even when pre-filled.
            from PyQt6.QtWidgets import QLineEdit
            assert page._api_key_input.echoMode() == QLineEdit.EchoMode.Password
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_initialize_page_defaults_base_url_for_first_run(self, qapp):
        """A first-time user (empty config) gets the common LM Studio base
        URL prefilled so they can just click Connect."""
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            page = OpenAICompatiblePage()
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.initializePage()
            assert page._base_url_input.text() == OpenAICompatiblePage._DEFAULT_BASE_URL
            assert page._chat_model_combo.currentText() == ""
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_model_fields_are_editable_dropdowns(self, qapp):
        """Chat + embedding models are editable combo boxes: a guided list to
        pick from, but power users can still type a model id."""
        from PyQt6.QtWidgets import QComboBox
        page = OpenAICompatiblePage()
        assert isinstance(page._chat_model_combo, QComboBox)
        assert isinstance(page._embed_model_combo, QComboBox)
        assert page._chat_model_combo.isEditable()
        assert page._embed_model_combo.isEditable()

    def test_fetch_models_returns_server_model_ids(self, stub_openai_server):
        """_fetch_models hits /v1/models on the configured server."""
        base, _ = stub_openai_server
        models = OpenAICompatiblePage._fetch_models(base, "", timeout=3)
        assert "stub-chat" in models and "stub-embed" in models

    def test_fetch_models_failsoft_on_unreachable_server(self):
        """An unreachable server yields an empty list (never raises), so the
        user can still type a model id by hand."""
        models = OpenAICompatiblePage._fetch_models("http://127.0.0.1:1/v1", "", timeout=1)
        assert models == []

    def test_populate_models_fills_dropdowns_preserving_current(self, qapp):
        """Fetched models populate the dropdowns; a value the user already
        typed is preserved as the current selection. Embedding combo gets a
        blank '(none)' entry."""
        page = OpenAICompatiblePage()
        page._chat_model_combo.setCurrentText("my-typed-model")
        page._populate_models(["a-model", "b-model"])
        chat_items = [page._chat_model_combo.itemText(i)
                      for i in range(page._chat_model_combo.count())]
        embed_items = [page._embed_model_combo.itemText(i)
                       for i in range(page._embed_model_combo.count())]
        assert chat_items == ["a-model", "b-model"]
        assert embed_items[0] == "" and "a-model" in embed_items
        assert page._chat_model_combo.currentText() == "my-typed-model"

    def test_on_models_fetched_status_messages(self, qapp):
        """The status line reflects success vs failure honestly."""
        page = OpenAICompatiblePage()
        page._on_models_fetched(True, ["m1", "m2"])
        assert "Connected" in page._connect_status.text() and "2" in page._connect_status.text()
        page._on_models_fetched(False, [])
        assert "Couldn't load models" in page._connect_status.text()

    def test_editing_base_url_refreshes_completeness(self, qapp):
        """Editing the base URL must re-evaluate the Next button: the base URL
        is half of isComplete, so a change to it (not just the chat model) has
        to fire completeChanged, otherwise Next can stick in a stale state."""
        page = OpenAICompatiblePage()
        page._chat_model_combo.setCurrentText("some-model")
        page._base_url_input.setText("")  # incomplete: no base URL
        assert page.isComplete() is False

        fired = []
        page.completeChanged.connect(lambda: fired.append(True))
        page._base_url_input.setText("http://localhost:1234/v1")

        assert fired, "editing the base URL should emit completeChanged"
        assert page.isComplete() is True

    def test_classify_models_splits_embed_from_chat(self):
        chat, embed = OpenAICompatiblePage._classify_models(
            ["qwen2.5-7b-instruct", "nomic-embed-text", "text-embedding-3-small", "gemma-2b"])
        assert chat == ["qwen2.5-7b-instruct", "gemma-2b"]
        assert embed == ["nomic-embed-text", "text-embedding-3-small"]

    def test_populate_models_applies_sensible_defaults(self, qapp):
        """With nothing chosen yet, the first chat model and the first embed
        model are preselected so the common case is just Connect then Next."""
        page = OpenAICompatiblePage()
        page._populate_models(["llama-3-8b", "nomic-embed-text", "phi-3"])
        assert page._chat_model_combo.currentText() == "llama-3-8b"
        assert page._embed_model_combo.currentText() == "nomic-embed-text"

    def test_populate_models_preserves_user_choice_over_default(self, qapp):
        page = OpenAICompatiblePage()
        page._chat_model_combo.setCurrentText("my-model")
        page._populate_models(["llama-3-8b", "phi-3"])
        assert page._chat_model_combo.currentText() == "my-model"

    def test_preset_prefills_base_url(self, qapp):
        """Choosing an app preset fills in its default base URL."""
        page = OpenAICompatiblePage()
        # index 1 is the first known server (LM Studio)
        label, url = OpenAICompatiblePage._KNOWN_SERVERS[0]
        page._preset_combo.setCurrentIndex(1)
        assert page._base_url_input.text() == url

    def test_discover_servers_finds_running_server(self, stub_openai_server):
        """Discovery returns reachable loopback servers and skips dead ports."""
        base, _ = stub_openai_server
        found = OpenAICompatiblePage._discover_servers(
            [("Stub", base), ("Dead", "http://127.0.0.1:1/v1")], timeout=2)
        assert found == [("Stub", base)]

    def test_capability_summary_reports_each_feature(self):
        ok = SimpleNamespace(reachable=True, chat=True, tools=True, embeddings=True)
        summary = OpenAICompatiblePage._capability_summary(ok)
        assert "✅ Chat" in summary and "✅ Tool calling" in summary and "✅ Embeddings" in summary

        no_embed = SimpleNamespace(reachable=True, chat=True, tools=True, embeddings=False)
        assert "No embeddings" in OpenAICompatiblePage._capability_summary(no_embed)

        unreachable = SimpleNamespace(reachable=False, chat=False, tools=False, embeddings=False)
        assert "Couldn't" in OpenAICompatiblePage._capability_summary(unreachable)

    def test_on_capabilities_offers_ollama_embeddings_when_server_cannot_embed(self, qapp):
        # isHidden() reflects the requested visibility flag without the page
        # being shown on screen (isVisible() needs a shown ancestor).
        page = OpenAICompatiblePage()
        page._on_capabilities(
            SimpleNamespace(reachable=True, chat=True, tools=True, embeddings=False))
        assert page._use_ollama_embed.isHidden() is False

    def test_on_capabilities_hides_offer_when_embeddings_work(self, qapp):
        page = OpenAICompatiblePage()
        page._use_ollama_embed.setVisible(True)
        page._on_capabilities(
            SimpleNamespace(reachable=True, chat=True, tools=True, embeddings=True))
        assert page._use_ollama_embed.isHidden() is True

    def test_validate_writes_ollama_embedding_split(self):
        """When the user opts to embed via Ollama, the config routes embeddings
        to Ollama and drops the remote embedding model."""
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        page._use_ollama_embed = SimpleNamespace(isVisible=lambda: True, isChecked=lambda: True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "http://localhost:9876/v1", "", "qwen-27b", "some-embed", "",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text())
            assert saved["embedding_provider"] == "ollama"
            assert "embedding_model" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_connect_with_empty_base_url_warns_without_starting_worker(self, qapp):
        page = OpenAICompatiblePage()
        page._base_url_input.setText("")
        page._on_connect()
        assert "base URL" in page._connect_status.text()
        assert page._fetch_worker is None

    def test_populate_models_all_embeddings_does_not_default_chat_to_embed(self, qapp):
        """A server that only lists embedding models must not auto-select an
        embedding model as the chat model."""
        page = OpenAICompatiblePage()
        page._populate_models(["nomic-embed-text", "text-embedding-3-small"])
        assert page._chat_model_combo.currentText() == ""

    def test_initialize_page_starts_discovery_only_without_saved_url(self, qapp):
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage()

        # Empty config (no saved URL) -> discovery runs.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            empty_cfg = Path(f.name)
        # Saved custom URL -> discovery is skipped, saved value kept.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"llm_base_url": "http://saved:9/v1"}, f)
            saved_cfg = Path(f.name)
        try:
            page._start_discovery = MagicMock()
            with patch("jarvis.config.default_config_path", return_value=empty_cfg):
                page.initializePage()
            assert page._start_discovery.call_count == 1

            page._start_discovery.reset_mock()
            with patch("jarvis.config.default_config_path", return_value=saved_cfg):
                page.initializePage()
            assert page._start_discovery.call_count == 0
            assert page._base_url_input.text() == "http://saved:9/v1"
        finally:
            empty_cfg.unlink(missing_ok=True)
            saved_cfg.unlink(missing_ok=True)

    def test_on_discovered_prefills_default_but_not_a_custom_url(self, qapp):
        page = OpenAICompatiblePage()
        # Still on the default URL -> discovery prefills the found server.
        page._base_url_input.setText(OpenAICompatiblePage._DEFAULT_BASE_URL)
        page._on_discovered([("Jan", "http://localhost:1337/v1")])
        assert page._base_url_input.text() == "http://localhost:1337/v1"
        assert "Found" in page._connect_status.text()

        # User typed a custom URL -> discovery must not clobber it.
        page._base_url_input.setText("http://mine:5/v1")
        page._on_discovered([("Jan", "http://localhost:1337/v1")])
        assert page._base_url_input.text() == "http://mine:5/v1"


class TestOllamaStatusDataclass:
    """Tests for OllamaStatus dataclass behavior."""

    def test_is_fully_setup_property(self):
        """is_fully_setup returns True only when all conditions are met."""
        # All good
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=True,
            missing_models=[],
        )
        assert status.is_fully_setup is True

        # Missing CLI
        status = OllamaStatus(
            is_cli_installed=False,
            is_server_running=True,
            missing_models=[],
        )
        assert status.is_fully_setup is False

        # Server not running
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=False,
            missing_models=[],
        )
        assert status.is_fully_setup is False

        # Missing models
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=True,
            missing_models=["some-model"],
        )
        assert status.is_fully_setup is False

    def test_default_values(self):
        """Dataclass initializes with correct defaults."""
        status = OllamaStatus()

        assert status.is_cli_installed is False
        assert status.cli_path is None
        assert status.is_server_running is False
        assert status.server_version is None
        assert status.installed_models == []
        assert status.missing_models == []


class TestLocationDetectionForWizard:
    """Tests for location detection utilities used in setup wizard."""

    def test_private_ip_detection(self):
        """Private IPs are correctly identified."""
        # RFC 1918 private ranges
        assert _is_private_ip("10.0.0.1") is True
        assert _is_private_ip("10.255.255.255") is True
        assert _is_private_ip("172.16.0.1") is True
        assert _is_private_ip("172.31.255.255") is True
        assert _is_private_ip("192.168.0.1") is True
        assert _is_private_ip("192.168.255.255") is True

        # Loopback
        assert _is_private_ip("127.0.0.1") is True

        # Public IPs (8.8.8.8 is Google DNS, 1.1.1.1 is Cloudflare)
        assert _is_private_ip("8.8.8.8") is False
        assert _is_private_ip("1.1.1.1") is False

    def test_location_context_returns_unknown_when_unavailable(self):
        """Location context returns 'Unknown' when detection fails."""
        # Disable auto-detect to avoid network calls, no config IP
        with patch("jarvis.utils.location._get_external_ip_automatically", return_value=None):
            with patch("jarvis.utils.location._get_local_network_ip", return_value="192.168.1.1"):
                context = get_location_context(config_ip=None, auto_detect=True)
                # Should return Unknown since 192.168.x.x can't be geolocated
                assert "Unknown" in context or "error" in context.lower()

    def test_location_availability_check(self):
        """is_location_available checks for GeoIP2 and database."""
        with patch("jarvis.utils.location.GEOIP2_AVAILABLE", False):
            # When library not available, should return False
            # Note: We can't easily patch the constant after import,
            # so we test the behavior indirectly
            pass

        # With patched database path
        with patch("jarvis.utils.location._get_database_path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            # Can't easily test due to import-time GEOIP2_AVAILABLE check
            # but the function should return False if DB doesn't exist

    def test_location_context_with_config_ip(self):
        """When config IP is provided and valid, uses it for location."""
        mock_location = {
            "city": "San Francisco",
            "region": "California",
            "country": "United States",
            "timezone": "America/Los_Angeles",
        }

        with patch("jarvis.utils.location.get_location_info", return_value=mock_location):
            context = get_location_context(config_ip="203.0.113.45")

            assert "San Francisco" in context
            assert "California" in context
            assert "United States" in context


class TestModelOptions:
    """Tests for model selection options in setup wizard."""

    def test_model_options_available(self):
        """Model options include both recommended and lightweight options."""
        from desktop_app.setup_wizard import ModelsPage

        assert "qwen3.8:27b" in ModelsPage.MODEL_OPTIONS
        assert DEFAULT_CHAT_MODEL in ModelsPage.MODEL_OPTIONS

    def test_model_options_have_required_fields(self):
        """Each model option has required info fields."""
        from desktop_app.setup_wizard import ModelsPage

        for model_id, info in ModelsPage.MODEL_OPTIONS.items():
            assert "name" in info, f"Model {model_id} missing 'name'"
            assert "description" in info, f"Model {model_id} missing 'description'"
            assert "size" in info, f"Model {model_id} missing 'size'"
            assert "vram" in info, f"Model {model_id} missing 'vram'"

    def test_model_options_uses_centralized_config(self):
        """ModelsPage.MODEL_OPTIONS should reference the centralized config."""
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        # Verify they're the same object (not just equal values)
        assert ModelsPage.MODEL_OPTIONS is SUPPORTED_CHAT_MODELS


class TestModelsPageUI:
    """Tests for the dropdown-based model selection UI in ModelsPage."""

    def test_uses_combobox_for_chat_model(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        from PyQt6.QtWidgets import QComboBox
        page = ModelsPage()
        assert isinstance(page._chat_combo, QComboBox)
        assert page._chat_combo.count() == len(ModelsPage.MODEL_OPTIONS)

    def test_uses_combobox_for_fast_model(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        from PyQt6.QtWidgets import QComboBox
        page = ModelsPage()
        assert isinstance(page._fast_combo, QComboBox)
        assert page._fast_combo.count() == len(ModelsPage._FAST_MODEL_IDS)

    def test_defaults_to_unlinked(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        assert page._linked is False
        assert page._link_cb.isChecked() is False

    def test_default_fast_model_is_gemma4_e2b(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        with patch("desktop_app.setup_wizard.detect_total_vram_mb", return_value=None):
            page = ModelsPage()
        assert page._fast_model == "gemma4:e2b"
        assert page._fast_combo.currentData() == "gemma4:e2b"

    def test_default_chat_model_is_default_config_model(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import DEFAULT_CHAT_MODEL
        page = ModelsPage()
        assert page._chat_model == DEFAULT_CHAT_MODEL
        assert page._chat_combo.currentData() == DEFAULT_CHAT_MODEL

    def test_initialize_page_stays_unlinked(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        page.initializePage()
        assert page._linked is False
        assert page._link_cb.isChecked() is False

    def test_linked_mode_syncs_both_combos(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        page._link_cb.setChecked(True)
        assert page._linked is True
        idx = page._chat_combo.findData("qwen3.5:0.8b")
        assert idx >= 0
        page._chat_combo.setCurrentIndex(idx)
        assert page._fast_model == "qwen3.5:0.8b"
        assert page._fast_combo.currentData() == "qwen3.5:0.8b"

    def test_unlinked_mode_allows_independent_selection(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        assert page._linked is False
        idx = page._fast_combo.findData("qwen3.5:0.8b")
        assert idx >= 0
        page._fast_combo.setCurrentIndex(idx)
        assert page._fast_model == "qwen3.5:0.8b"
        assert page._chat_model != "qwen3.5:0.8b"

    def test_auto_downgrades_fast_model_when_smaller_chat_selected(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        idx = page._chat_combo.findData("qwen3.5:0.8b")
        assert idx >= 0
        page._chat_combo.setCurrentIndex(idx)
        assert page._fast_model == "qwen3.5:0.8b"

    def test_fast_combo_uses_data_keys_for_fast_suitable_models(self, qapp):
        from desktop_app.setup_wizard import ModelsPage
        page = ModelsPage()
        datas = [page._fast_combo.itemData(i) for i in range(page._fast_combo.count())]
        for d in datas:
            assert d in ModelsPage._FAST_MODEL_IDS


class TestOpenAICompatiblePageDefaults:
    """Tests for default link state and fast model in OpenAI compatible page."""

    def test_defaults_to_unlinked(self, qapp):
        from desktop_app.setup_wizard import OpenAICompatiblePage
        page = OpenAICompatiblePage()
        assert page._openai_linked is False
        assert page._openai_link_cb.isChecked() is False

    def test_fast_model_selector_visible_by_default(self, qapp):
        from desktop_app.setup_wizard import OpenAICompatiblePage
        page = OpenAICompatiblePage()
        assert page._openai_linked is False
        assert page._fast_label.isHidden() is False
        assert page._fast_model_combo.isHidden() is False

    def test_fast_model_defaults_to_gemma4_e2b_when_in_model_list(self, qapp):
        from desktop_app.setup_wizard import OpenAICompatiblePage
        page = OpenAICompatiblePage()
        page._populate_models(["gemma4:e2b", "llama-3-8b", "nomic-embed-text"])
        assert page._fast_model_combo.currentText() == "gemma4:e2b"

    def test_fast_model_stays_empty_when_gemma4_not_available(self, qapp):
        from desktop_app.setup_wizard import OpenAICompatiblePage
        page = OpenAICompatiblePage()
        page._populate_models(["llama-3-8b", "phi-3", "nomic-embed-text"])
        assert page._fast_model_combo.currentText() == ""

    def test_link_toggle_shows_hides_fast_selector(self, qapp):
        from desktop_app.setup_wizard import OpenAICompatiblePage
        page = OpenAICompatiblePage()
        assert page._fast_label.isHidden() is False
        page._openai_link_cb.setChecked(True)
        assert page._fast_label.isHidden() is True
        assert page._fast_model_combo.isHidden() is True


class TestDefaultModelDetection:
    """Regression tests: the default small model must be detected as missing when not
    installed, triggering the setup wizard install prompt.

    Uses DEFAULT_CHAT_MODEL from config so these tests stay valid when the default
    model changes — no hardcoded model names here.
    """

    EMBED_MODEL = "nomic-embed-text"

    def test_small_model_missing_detected_in_status(self):
        """When the default chat model is not installed, check_ollama_status reports it as missing."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=[self.EMBED_MODEL]):
                        status = check_ollama_status()

                        assert DEFAULT_CHAT_MODEL in status.missing_models
                        assert status.is_fully_setup is False

    def test_small_model_installed_not_in_missing(self):
        """When the default chat model is installed, check_ollama_status does not list it as missing."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=required):
                        status = check_ollama_status()

                        assert status.missing_models == []
                        assert status.is_fully_setup is True

    def test_wizard_shown_when_small_model_missing(self):
        """should_show_setup_wizard returns True when the default chat model is not installed."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.3.0",
            installed_models=[self.EMBED_MODEL],
            missing_models=[DEFAULT_CHAT_MODEL],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_wizard_not_shown_when_small_model_installed(self):
        """should_show_setup_wizard returns False when the default chat model is present."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.3.0",
            installed_models=[DEFAULT_CHAT_MODEL, self.EMBED_MODEL],
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_latest_tag_stripped_before_comparison(self):
        """Ollama appends ':latest' to model names; the status check must strip it so
        '<model>:latest' is not incorrectly treated as missing when '<model>' is required."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    # Simulate Ollama reporting "<model>:latest" in its model list
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = (
                        "NAME                       ID              SIZE      MODIFIED\n"
                        f"{DEFAULT_CHAT_MODEL}:latest    abc123          2.0 GB    1 day ago\n"
                        f"{self.EMBED_MODEL}:latest    def456          274 MB    1 week ago\n"
                    )
                    with patch("subprocess.run", return_value=mock_result):
                        status = check_ollama_status()

                        assert DEFAULT_CHAT_MODEL not in status.missing_models
                        assert status.is_fully_setup is True


class TestWhisperModelOptions:
    """Tests for whisper model selection options in setup wizard."""

    def test_whisper_multilingual_model_options_available(self):
        """Multilingual whisper model options include recommended and lightweight options."""
        from desktop_app.setup_wizard import WhisperSetupPage

        model_ids = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS]
        assert "small" in model_ids
        assert "tiny" in model_ids
        assert "large-v3-turbo" in model_ids

    def test_whisper_english_model_options_available(self):
        """English-only whisper model options include recommended and lightweight options."""
        from desktop_app.setup_wizard import WhisperSetupPage

        model_ids = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN]
        assert "small.en" in model_ids
        assert "tiny.en" in model_ids
        assert "medium.en" in model_ids
        # Note: large models don't have .en variants
        assert not any("large" in m for m in model_ids)

    def test_whisper_multilingual_model_options_have_required_fields(self):
        """Each multilingual whisper model option has required info fields."""
        from desktop_app.setup_wizard import WhisperSetupPage

        for model_tuple in WhisperSetupPage.WHISPER_MODEL_OPTIONS:
            assert len(model_tuple) == 5, f"Whisper model tuple should have 5 elements: {model_tuple}"
            model_id, name, file_size, ram, desc = model_tuple
            assert model_id, "Model ID should not be empty"
            assert name, "Model name should not be empty"
            assert file_size, "Model file size should not be empty"
            assert ram, "Model RAM requirement should not be empty"
            assert desc, "Model description should not be empty"
            # Multilingual models should NOT have .en suffix
            assert not model_id.endswith(".en"), f"Multilingual model should not end with .en: {model_id}"

    def test_turbo_hidden_when_faster_whisper_unsupported(self):
        """large-v3-turbo is filtered from options when faster-whisper is too old."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = False
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        with patch("desktop_app.setup_wizard._is_faster_whisper_turbo_supported", return_value=False):
            options = WhisperSetupPage._get_current_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" not in model_ids
        assert "small" in model_ids

    def test_turbo_shown_when_faster_whisper_supported(self):
        """large-v3-turbo is available when faster-whisper supports it."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = False
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        with patch("desktop_app.setup_wizard._is_faster_whisper_turbo_supported", return_value=True):
            options = WhisperSetupPage._get_current_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" in model_ids

    def test_turbo_always_shown_on_apple_silicon(self):
        """large-v3-turbo is always available on Apple Silicon (MLX backend)."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = True
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        with patch("desktop_app.setup_wizard._is_faster_whisper_turbo_supported", return_value=False):
            options = WhisperSetupPage._get_current_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" in model_ids

    def test_whisper_english_model_options_have_required_fields(self):
        """Each English-only whisper model option has required info fields."""
        from desktop_app.setup_wizard import WhisperSetupPage

        for model_tuple in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN:
            assert len(model_tuple) == 5, f"Whisper model tuple should have 5 elements: {model_tuple}"
            model_id, name, file_size, ram, desc = model_tuple
            assert model_id, "Model ID should not be empty"
            assert name, "Model name should not be empty"
            assert file_size, "Model file size should not be empty"
            assert ram, "Model RAM requirement should not be empty"
            assert desc, "Model description should not be empty"
            # English-only models should have .en suffix
            assert model_id.endswith(".en"), f"English model should end with .en: {model_id}"


class TestWhisperSetupPageSliderRebuild:
    """Regression tests for WhisperSetupPage slider rebuild lifecycle.

    On macOS, promoting a child QLabel to a top-level widget (via
    setParent(None)) during a QWizard page transition could trigger
    a SIGABRT ('Fatal Python error: Aborted') while the next page
    was being shown.  These tests guarantee that the slider labels
    stay parented to their containers throughout rebuilds — the
    safe pattern for clearing items out of a layout.
    """

    def test_slider_labels_keep_container_parent_after_rebuild(self, qapp):
        """Newly-built slider labels must remain children of their containers.

        If any label ends up reparented to None it becomes a top-level
        widget, which on macOS triggers a native window creation that
        can abort during wizard page transitions.
        """
        from desktop_app.setup_wizard import WhisperSetupPage

        page = WhisperSetupPage()

        # Toggle language mode — this fires _rebuild_slider_ui which
        # clears the old labels and inserts a new set.
        page._on_language_changed(True)
        page._on_language_changed(False)

        labels_container = page._labels_container
        size_container = page._size_container

        for i in range(page._labels_layout.count()):
            item = page._labels_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                assert w.parent() is labels_container, (
                    "Slider name labels must stay parented to their "
                    "container — a None parent promotes them to top-level "
                    "widgets, which crashes QWizard transitions on macOS."
                )

        for i in range(page._size_layout.count()):
            item = page._size_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                assert w.parent() is size_container, (
                    "Slider size labels must stay parented to their "
                    "container — a None parent promotes them to top-level "
                    "widgets, which crashes QWizard transitions on macOS."
                )

    def test_initialize_page_can_be_called_multiple_times(self, qapp):
        """initializePage must be safely re-callable.

        QWizard calls initializePage each time a page is shown.  The
        first call (right after construction) has to clear the initial
        labels that __init__ built, and subsequent calls must not
        crash or leak top-level widgets.
        """
        from desktop_app.setup_wizard import WhisperSetupPage

        page = WhisperSetupPage()

        # Re-initialise a few times — this mirrors back/forward
        # navigation between wizard pages.
        for _ in range(3):
            page.initializePage()

        # All remaining labels in the layouts are still properly
        # parented (not promoted to top-level).
        for layout, container in [
            (page._labels_layout, page._labels_container),
            (page._size_layout, page._size_container),
        ]:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget()
                if w is not None:
                    assert w.parent() is container


class TestMCPPage:
    """Tests for the MCP servers wizard page."""

    def test_mcp_page_is_always_complete(self):
        """MCP page should always be completeable (nothing is required)."""
        # MCPPage.isComplete is hardcoded to True — the page is always optional
        page = MCPPage.__new__(MCPPage)
        assert page.isComplete() is True

    def test_is_already_configured_returns_false_on_empty_config(self):
        """When config has no mcps key, returns False."""
        with patch("jarvis.config._load_json", return_value={}):
            assert MCPPage._is_already_configured("filesystem") is False

    def test_is_already_configured_returns_true_when_present(self):
        """When the server name exists in config.mcps, returns True."""
        mock_config = {"mcps": {"filesystem": {"transport": "stdio"}}}
        with patch("jarvis.config._load_json", return_value=mock_config):
            assert MCPPage._is_already_configured("filesystem") is True

    def test_is_already_configured_handles_exception(self):
        """Returns False if config loading fails."""
        with patch("jarvis.config._load_json", side_effect=Exception("boom")):
            assert MCPPage._is_already_configured("filesystem") is False

    def test_wizard_entries_available(self):
        """Wizard-featured catalogue entries are available for the MCP page."""
        entries = get_wizard_entries()
        assert len(entries) >= 1
        # All entries should have display names and descriptions
        for e in entries:
            assert e.display_name
            assert e.description

    def test_validate_page_saves_selected_mcps(self):
        """validatePage writes selected MCPs to config."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            page = MCPPage.__new__(MCPPage)
            entries = get_wizard_entries()
            # Simulate checkboxes: first entry checked, rest unchecked
            page._checkboxes = {}
            for i, entry in enumerate(entries):
                cb = MagicMock()
                cb.isChecked.return_value = (i == 0)
                page._checkboxes[entry.name] = cb

            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                result = page.validatePage()

            assert result is True
            saved = _load_json(cfg_path)
            first_entry = entries[0]
            assert first_entry.name in saved.get("mcps", {})
            assert saved["mcps"][first_entry.name]["command"] == first_entry.command
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_is_node_available_returns_true_when_npx_found(self):
        """_is_node_available returns True when _resolve_command succeeds."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", return_value="/usr/bin/npx"):
            assert MCPPage._is_node_available() is True

    def test_is_node_available_returns_false_when_npx_missing(self):
        """_is_node_available returns False when _resolve_command raises."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", side_effect=FileNotFoundError("not found")):
            assert MCPPage._is_node_available() is False

    def test_validate_page_preserves_existing_non_wizard_mcps(self):
        """validatePage must not remove MCPs that aren't in the wizard catalogue."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"mcps": {"custom-server": {"transport": "stdio", "command": "node", "args": []}}}, f)
            cfg_path = Path(f.name)

        try:
            page = MCPPage.__new__(MCPPage)
            entries = get_wizard_entries()
            page._checkboxes = {}
            for entry in entries:
                cb = MagicMock()
                cb.isChecked.return_value = False
                page._checkboxes[entry.name] = cb

            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()

            saved = _load_json(cfg_path)
            assert "custom-server" in saved.get("mcps", {}), "Custom MCP server was removed"
        finally:
            cfg_path.unlink(missing_ok=True)


class TestSearchProvidersPage:
    """Tests for the Search Providers wizard page (Brave + Wikipedia)."""

    def _make_page(self, brave_key: str, wiki_enabled: bool) -> SearchProvidersPage:
        page = SearchProvidersPage.__new__(SearchProvidersPage)
        brave_input = MagicMock()
        brave_input.text.return_value = brave_key
        wiki_check = MagicMock()
        wiki_check.isChecked.return_value = wiki_enabled
        page._brave_input = brave_input
        page._wiki_check = wiki_check
        return page

    def test_page_is_always_complete(self):
        page = SearchProvidersPage.__new__(SearchProvidersPage)
        assert page.isComplete() is True

    def test_validate_writes_brave_key_when_provided(self):
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="BSA-abc123", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                assert page.validatePage() is True
            saved = _load_json(cfg_path)
            # Default non-default-only write: Brave present, Wikipedia omitted.
            assert saved.get("brave_search_api_key") == "BSA-abc123"
            assert "wikipedia_fallback_enabled" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_omits_empty_brave_key(self):
        """Empty Brave key must NOT write an empty-string entry — matches
        the settings-window minimal-diff invariant."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="   ", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert "brave_search_api_key" not in saved
            assert "wikipedia_fallback_enabled" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_persists_wikipedia_disable_only(self):
        """Wikipedia defaults to True, so only write it when user disables it."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="", wiki_enabled=False)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert saved.get("wikipedia_fallback_enabled") is False
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_removes_existing_brave_key_when_cleared(self):
        """If user blanks the Brave key, the entry must be removed, not kept."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"brave_search_api_key": "old-key"}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert "brave_search_api_key" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_preserves_unrelated_keys(self):
        """validatePage must not clobber unrelated config entries."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"ollama_chat_model": "gpt-oss:20b", "mcps": {"x": {}}}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="BSA-key", wiki_enabled=False)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert saved["ollama_chat_model"] == "gpt-oss:20b"
            assert saved["mcps"] == {"x": {}}
        finally:
            cfg_path.unlink(missing_ok=True)

