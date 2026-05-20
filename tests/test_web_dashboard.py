"""Tests for the Jarvis web command centre surface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestWebDashboard:
    """Coverage for the futuristic web status surface."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        from desktop_app import memory_viewer

        memory_viewer.app.config["TESTING"] = True
        memory_viewer._dashboard_logs.clear()
        self.client = memory_viewer.app.test_client()
        yield
        memory_viewer._dashboard_logs.clear()

    def test_dashboard_page_contains_core_panels(self):
        resp = self.client.get("/dashboard")

        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Jarvis Command Centre" in html
        assert "Presence" in html
        assert "Model Readiness" in html
        assert "Language Quality" in html
        assert "Memory & Tools" in html
        assert "Work Queue" in html
        assert "Data Roots" in html

    def test_status_api_reports_models_language_and_privacy(self):
        from desktop_app import memory_viewer

        settings = MagicMock()
        settings.ollama_chat_model = "gemma4:e2b"
        settings.intent_judge_model = "gemma4:e2b"
        settings.ollama_embed_model = "nomic-embed-text"
        settings.whisper_model = "medium"
        settings.hot_window_enabled = True
        settings.hot_window_seconds = 3.0
        settings.transcript_buffer_duration_sec = 120.0
        settings.wake_word = "jarvis"
        settings.mcps = {"github": {"command": "gh"}}
        settings.web_search_enabled = True
        settings.operator_name = "Mr. Johnson"
        settings.persona_style = "formal_majordomo"
        settings.operator_briefing_enabled = True
        settings.work_queue_enabled = True
        settings.data_live_roots = []
        settings.latvian_quality_enabled = False
        settings.ollama_latvian_model = ""
        settings.ledger_enabled = True
        settings.ledger_path = ""

        with patch.object(memory_viewer, "load_settings", return_value=settings):
            with patch.object(
                memory_viewer, "_get_installed_ollama_models", return_value=["gemma4:e2b"]
            ):
                resp = self.client.get("/api/dashboard/status")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["presence"]["mode"] == "passive_wake_word_monitoring"
        assert data["privacy"]["mic_control"] == "explicit_start_stop"
        assert data["models"]["chat"]["id"] == "gemma4:e2b"
        assert data["models"]["chat"]["ready"] is True
        assert data["models"]["embedding"]["ready"] is False
        assert data["language"]["mode"] == "multilingual"
        assert data["conversation"]["hot_window_enabled"] is True
        assert data["tools"]["mcp_count"] == 1
        assert data["operator"]["name"] == "Mr. Johnson"
        assert "work_queue" in data
        assert "data_live" in data
        assert "latvian" in data
        assert "ledger" in data
        assert "mcp" in data

    def test_dashboard_logs_are_redacted_and_bounded(self):
        from desktop_app.memory_viewer import append_dashboard_log

        append_dashboard_log("Token secret=abc123 password=hunter2")
        resp = self.client.get("/api/dashboard/logs")

        assert resp.status_code == 200
        logs = resp.get_json()["logs"]
        assert len(logs) == 1
        assert "hunter2" not in logs[0]
        assert "[REDACTED]" in logs[0]
