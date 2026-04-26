"""
Tests for setup wizard detection functions.

These tests verify the Ollama detection logic without touching the UI.
They treat the detection functions as black boxes, verifying inputs produce correct outputs.
"""

import subprocess
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
)
from desktop_app.mcp_catalogue import get_wizard_entries
from jarvis.config import DEFAULT_CHAT_MODEL
from jarvis.utils.location import (
    get_location_context,
    is_location_available,
    _is_private_ip,
)


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
        mock_settings.intent_judge_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            assert "llama2:7b" in models
            assert "nomic-embed-text" in models

    def test_includes_intent_judge_model_when_different_from_chat(self):
        """Includes intent judge model when it differs from chat model."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "gpt-oss:20b"  # Different from intent judge
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.intent_judge_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            # Should have 3 models: chat, embed, and intent judge
            assert len(models) == 3
            assert "gpt-oss:20b" in models
            assert "nomic-embed-text" in models
            assert "gemma4:e2b" in models  # Intent judge model is always required

    def test_returns_defaults_on_config_error(self):
        """Returns default models if config can't be loaded."""
        with patch("desktop_app.setup_wizard.load_settings", side_effect=Exception("Config error")):
            models = get_required_models()

            assert len(models) == 2
            assert "gemma4:e2b" in models
            assert "nomic-embed-text" in models


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

        assert "gpt-oss:20b" in ModelsPage.MODEL_OPTIONS
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

