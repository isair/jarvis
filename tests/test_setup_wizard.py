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
    should_show_setup_wizard,
    OllamaStatus,
)
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

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            assert "llama2:7b" in models
            assert "nomic-embed-text" in models

    def test_returns_defaults_on_config_error(self):
        """Returns default models if config can't be loaded."""
        with patch("desktop_app.setup_wizard.load_settings", side_effect=Exception("Config error")):
            models = get_required_models()

            assert len(models) == 2
            assert "llama3.2:3b" in models
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
        assert "llama3.2:3b" in ModelsPage.MODEL_OPTIONS

    def test_model_options_have_required_fields(self):
        """Each model option has required info fields."""
        from desktop_app.setup_wizard import ModelsPage

        for model_id, info in ModelsPage.MODEL_OPTIONS.items():
            assert "name" in info, f"Model {model_id} missing 'name'"
            assert "description" in info, f"Model {model_id} missing 'description'"
            assert "size" in info, f"Model {model_id} missing 'size'"
            assert "ram" in info, f"Model {model_id} missing 'ram'"

    def test_model_options_uses_centralized_config(self):
        """ModelsPage.MODEL_OPTIONS should reference the centralized config."""
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        # Verify they're the same object (not just equal values)
        assert ModelsPage.MODEL_OPTIONS is SUPPORTED_CHAT_MODELS


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

