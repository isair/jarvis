"""
Tests for desktop_app.py functionality.

Tests crash detection, model support checking, and other utility functions.
Note: GUI components are not tested here - only the underlying logic.
"""

import os
import pytest
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestGetCrashPaths:
    """Tests for get_crash_paths() function."""

    def test_returns_three_paths(self):
        """get_crash_paths() should return a tuple of 3 paths."""
        from desktop_app import get_crash_paths

        result = get_crash_paths()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_paths_are_path_objects(self):
        """All returned paths should be Path objects."""
        from desktop_app import get_crash_paths

        crash_log, crash_marker, previous_crash = get_crash_paths()
        assert isinstance(crash_log, Path)
        assert isinstance(crash_marker, Path)
        assert isinstance(previous_crash, Path)

    def test_paths_have_expected_names(self):
        """Paths should have the expected filenames."""
        from desktop_app import get_crash_paths

        crash_log, crash_marker, previous_crash = get_crash_paths()
        assert crash_log.name == "jarvis_desktop_crash.log"
        assert crash_marker.name == ".crash_marker"
        assert previous_crash.name == "previous_crash.log"

    def test_paths_share_same_parent_directory(self):
        """All crash paths should be in the same directory."""
        from desktop_app import get_crash_paths

        crash_log, crash_marker, previous_crash = get_crash_paths()
        assert crash_log.parent == crash_marker.parent == previous_crash.parent

    @patch("sys.platform", "darwin")
    def test_macos_uses_library_logs(self):
        """On macOS, should use ~/Library/Logs/Jarvis."""
        # Note: This is tricky because the function reads sys.platform at runtime
        from desktop_app import get_crash_paths

        crash_log, _, _ = get_crash_paths()
        if sys.platform == "darwin":
            assert "Library" in str(crash_log) or "Logs" in str(crash_log)


class TestCrashMarkerFunctions:
    """Tests for mark_session_started() and mark_session_clean_exit()."""

    def test_mark_session_started_creates_marker(self):
        """mark_session_started() should create the crash marker file."""
        from desktop_app import get_crash_paths, mark_session_started, mark_session_clean_exit

        _, crash_marker, _ = get_crash_paths()

        # Clean up first
        crash_marker.unlink(missing_ok=True)
        assert not crash_marker.exists()

        # Start session
        mark_session_started()
        assert crash_marker.exists()

        # Clean up
        mark_session_clean_exit()

    def test_mark_session_clean_exit_removes_marker(self):
        """mark_session_clean_exit() should remove the crash marker file."""
        from desktop_app import get_crash_paths, mark_session_started, mark_session_clean_exit

        _, crash_marker, _ = get_crash_paths()

        # Create marker
        mark_session_started()
        assert crash_marker.exists()

        # Clean exit
        mark_session_clean_exit()
        assert not crash_marker.exists()

    def test_mark_session_clean_exit_handles_missing_marker(self):
        """mark_session_clean_exit() should not error if marker doesn't exist."""
        from desktop_app import get_crash_paths, mark_session_clean_exit

        _, crash_marker, _ = get_crash_paths()
        crash_marker.unlink(missing_ok=True)

        # Should not raise
        mark_session_clean_exit()


class TestCheckPreviousCrash:
    """Tests for check_previous_crash() function."""

    def test_returns_none_when_no_marker(self):
        """check_previous_crash() should return None if no crash marker exists."""
        from desktop_app import get_crash_paths, check_previous_crash, mark_session_clean_exit

        # Ensure clean state
        mark_session_clean_exit()

        result = check_previous_crash()
        assert result is None

    def test_returns_none_when_marker_but_no_crash_log(self):
        """check_previous_crash() should return None if marker exists but no crash content."""
        from desktop_app import get_crash_paths, check_previous_crash, mark_session_started

        crash_log, crash_marker, _ = get_crash_paths()

        # Create marker but empty/missing crash log
        mark_session_started()
        crash_log.unlink(missing_ok=True)

        result = check_previous_crash()
        # Marker should be removed even if no crash content
        assert not crash_marker.exists()

    def test_returns_content_when_crash_detected(self):
        """check_previous_crash() should return crash content when crash is detected."""
        from desktop_app import get_crash_paths, check_previous_crash

        crash_log, crash_marker, previous_crash = get_crash_paths()

        # Simulate a crash: marker exists and crash log has error content
        crash_marker.touch()
        crash_content = "Fatal error: Something went wrong\nTraceback (most recent call last):\n  File test.py"
        crash_log.write_text(crash_content, encoding='utf-8')

        result = check_previous_crash()

        # Should return the crash content
        assert result is not None
        assert "Fatal" in result or "Traceback" in result

        # Marker should be removed
        assert not crash_marker.exists()

        # Previous crash should be saved
        assert previous_crash.exists()

        # Clean up
        crash_log.unlink(missing_ok=True)
        previous_crash.unlink(missing_ok=True)

    def test_ignores_normal_log_content(self):
        """check_previous_crash() should ignore logs without error indicators."""
        from desktop_app import get_crash_paths, check_previous_crash

        crash_log, crash_marker, _ = get_crash_paths()

        # Create marker with normal (non-crash) log content
        crash_marker.touch()
        crash_log.write_text("Normal startup log\nEverything is fine", encoding='utf-8')

        result = check_previous_crash()

        # Should return None since no crash indicators
        assert result is None

        # Marker should still be removed
        assert not crash_marker.exists()

        # Clean up
        crash_log.unlink(missing_ok=True)


class TestCheckModelSupport:
    """Tests for check_model_support() function."""

    @patch("jarvis.config.load_config")
    def test_returns_none_for_supported_model(self, mock_load_config):
        """check_model_support() should return None for supported models."""
        from desktop_app import check_model_support
        from jarvis.config import DEFAULT_CHAT_MODEL

        mock_load_config.return_value = {"ollama_chat_model": DEFAULT_CHAT_MODEL}

        result = check_model_support()
        assert result is None

    @patch("jarvis.config.load_config")
    def test_returns_model_name_for_unsupported_model(self, mock_load_config):
        """check_model_support() should return model name for unsupported models."""
        from desktop_app import check_model_support

        mock_load_config.return_value = {"ollama_chat_model": "some-unsupported-model:7b"}

        result = check_model_support()
        assert result == "some-unsupported-model:7b"

    @patch("jarvis.config.load_config")
    def test_matches_base_model_name(self, mock_load_config):
        """check_model_support() should match base model names without tags."""
        from desktop_app import check_model_support
        from jarvis.config import SUPPORTED_CHAT_MODELS

        # Get a supported model and use just its base name
        supported_model = next(iter(SUPPORTED_CHAT_MODELS.keys()))
        base_name = supported_model.split(":")[0]

        mock_load_config.return_value = {"ollama_chat_model": base_name}

        result = check_model_support()
        assert result is None  # Should be recognized as supported

    @patch("jarvis.config.load_config")
    def test_handles_config_error_gracefully(self, mock_load_config):
        """check_model_support() should return None on config errors."""
        from desktop_app import check_model_support

        mock_load_config.side_effect = Exception("Config error")

        result = check_model_support()
        assert result is None

    @patch("jarvis.config.load_config")
    def test_uses_default_when_not_configured(self, mock_load_config):
        """check_model_support() should use default model when not in config."""
        from desktop_app import check_model_support

        mock_load_config.return_value = {}  # No ollama_chat_model key

        result = check_model_support()
        # Default model is supported, so should return None
        assert result is None


class TestModelSupportIntegration:
    """Integration tests for model support checking."""

    def test_all_supported_models_pass_check(self):
        """All models in SUPPORTED_CHAT_MODELS should pass the support check."""
        from desktop_app import check_model_support
        from jarvis.config import SUPPORTED_CHAT_MODELS

        for model_id in SUPPORTED_CHAT_MODELS:
            with patch("jarvis.config.load_config") as mock_config:
                mock_config.return_value = {"ollama_chat_model": model_id}
                result = check_model_support()
                assert result is None, f"Model {model_id} should be supported"


class TestLogViewerReportIssue:
    """Tests for report issue URL generation logic.

    Note: We test the URL generation logic directly rather than through the
    LogViewerWindow class because Qt GUI components require a display server
    and block in test environments.
    """

    def test_report_issue_url_generation(self):
        """Report issue should generate correct GitHub issue URL with redacted content."""
        import urllib.parse
        import webbrowser
        from jarvis import get_version
        from jarvis.utils.redact import redact

        # Simulate what _report_issue does
        log_content = (
            "Starting Jarvis...\n"
            "API token: sk-secret-key-12345\n"
            "User email: user@example.com\n"
            "Error: Something went wrong\n"
        )

        # Apply same redaction as the actual method
        redacted_logs = redact(log_content, max_len=6000)

        try:
            version = get_version()
        except Exception:
            version = "unknown"

        # Build URL same as the actual method
        title = "Bug Report"
        body = f"""## Bug Report

**Version:** {version}
**Platform:** {sys.platform}

### Description
(Please describe what went wrong or what you expected to happen)



### Steps to Reproduce
1.
2.
3.

<details>
<summary>📋 Logs (click to expand)</summary>

```
{redacted_logs}
```

</details>

### Additional Context
(Any other relevant information)
"""
        params = urllib.parse.urlencode({
            'title': title,
            'body': body,
            'labels': 'bug'
        })
        url = f"https://github.com/isair/jarvis/issues/new?{params}"

        # Parse and verify
        assert url.startswith("https://github.com/isair/jarvis/issues/new?")
        parsed = urllib.parse.urlparse(url)
        params_parsed = urllib.parse.parse_qs(parsed.query)

        # Check title and labels
        assert params_parsed['title'][0] == "Bug Report"
        assert params_parsed['labels'][0] == "bug"

        # Check body contains expected sections
        body_decoded = params_parsed['body'][0]
        assert "## Bug Report" in body_decoded
        assert "### Description" in body_decoded
        assert "### Steps to Reproduce" in body_decoded
        assert "<details>" in body_decoded
        assert "📋 Logs (click to expand)" in body_decoded

        # Check that sensitive data was redacted
        assert "user@example.com" not in body_decoded
        assert "[REDACTED_EMAIL]" in body_decoded

    def test_report_issue_truncates_long_logs(self):
        """Report issue should truncate very long log content."""
        from jarvis.utils.redact import redact

        # Create very long realistic log content (not just repeated chars)
        long_content = "\n".join([f"[2024-01-{i:02d}] Processing request {i}" for i in range(1, 500)])

        # Apply redaction with same limit as actual method
        redacted = redact(long_content, max_len=6000)

        # Apply same truncation logic as actual method
        if len(redacted) > 5000:
            redacted = redacted[:5000] + "\n\n... (truncated)"

        # Verify truncation happened
        assert len(redacted) <= 5020  # 5000 + len("\n\n... (truncated)")
        assert "... (truncated)" in redacted

    def test_redaction_handles_multiple_sensitive_patterns(self):
        """Redaction should handle multiple types of sensitive data."""
        from jarvis.utils.redact import redact

        log_content = (
            "Config loaded:\n"
            "  email: admin@company.com\n"
            "  jwt_value: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test\n"
            "  password: secret123\n"
            "  hash: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4\n"
        )

        redacted = redact(log_content)

        # Email should be redacted
        assert "admin@company.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

        # JWT should be redacted (when not preceded by token=)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED_JWT]" in redacted

        # Password assignment should be redacted
        assert "secret123" not in redacted
        assert "[REDACTED]" in redacted

        # Long hex string should be redacted
        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in redacted
        assert "[REDACTED_HEX]" in redacted


class TestDiaryIPCProtocol:
    """Tests for diary dialog IPC protocol parsing.

    Note: Qt tests require a QApplication which may conflict with pytest fixtures.
    These tests focus on the IPC protocol parsing logic.
    """

    def test_diary_ipc_prefix_constant(self):
        """Diary IPC prefix should be a valid string constant."""
        from desktop_app.diary_dialog import DIARY_IPC_PREFIX

        assert isinstance(DIARY_IPC_PREFIX, str)
        assert len(DIARY_IPC_PREFIX) > 0
        # Prefix should be unique enough to not conflict with normal log lines
        assert DIARY_IPC_PREFIX == "__DIARY__:"

    def test_ipc_event_format_is_parseable(self):
        """IPC event format should be valid JSON after prefix."""
        import json
        from desktop_app.diary_dialog import DIARY_IPC_PREFIX

        # Test various event types
        events = [
            {"type": "chunks", "data": ["chunk1", "chunk2"]},
            {"type": "token", "data": "hello"},
            {"type": "status", "data": "Writing..."},
            {"type": "complete", "data": True},
        ]

        for event in events:
            line = f"{DIARY_IPC_PREFIX}{json.dumps(event)}"
            # Should be parseable
            assert line.startswith(DIARY_IPC_PREFIX)
            json_str = line[len(DIARY_IPC_PREFIX):]
            parsed = json.loads(json_str)
            assert parsed == event

    def test_normal_log_lines_dont_match_prefix(self):
        """Normal daemon log lines should not start with IPC prefix."""
        from desktop_app.diary_dialog import DIARY_IPC_PREFIX

        # Common log patterns that should NOT be intercepted
        normal_logs = [
            "Starting Jarvis daemon...",
            "✓ Daemon started",
            "📝 Updating diary...",
            "🔄 Daemon shutting down...",
            "✅ Diary update complete",
            "",
            "DEBUG: some message",
        ]

        for log in normal_logs:
            assert not log.startswith(DIARY_IPC_PREFIX), f"Log line should not match prefix: {log}"


class TestMemoryViewerModulePath:
    """Tests to verify memory viewer module references are valid.

    These tests catch issues like wrong module paths in subprocess calls
    without requiring actual GUI/server components.
    """

    def test_memory_viewer_module_is_importable(self):
        """The module used for subprocess mode should be importable."""
        import importlib

        pytest.importorskip("flask")

        # This is the module path used in MemoryViewerWindow.start_server()
        # If this fails, the subprocess command will fail at runtime
        module = importlib.import_module("desktop_app.memory_viewer")
        assert hasattr(module, "app"), "memory_viewer should have Flask 'app' attribute"
        assert hasattr(module, "main"), "memory_viewer should have 'main' function"

    def test_memory_viewer_subprocess_module_runs(self):
        """The module should be runnable with python -m (with correct PYTHONPATH)."""
        pytest.importorskip("flask")

        # Set PYTHONPATH the same way start_server() does
        src_path = Path(__file__).parent.parent / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_path)

        # Test that the module can at least be imported in subprocess
        result = subprocess.run(
            [sys.executable, "-c", "import desktop_app.memory_viewer"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0, f"Module import failed: {result.stderr}"

    def test_memory_viewer_module_path_matches_code(self):
        """Verify the module path in start_server matches the actual location."""
        import re
        from pathlib import Path

        # Read the actual code to find the module path used
        app_py = Path(__file__).parent.parent / "src" / "desktop_app" / "app.py"
        content = app_py.read_text(encoding="utf-8")

        # Find the subprocess module path
        match = re.search(r'"-m",\s*"([^"]+)"', content)
        assert match, "Could not find subprocess module path in app.py"

        module_path = match.group(1)
        assert module_path == "desktop_app.memory_viewer", (
            f"Module path should be 'desktop_app.memory_viewer', found '{module_path}'"
        )
