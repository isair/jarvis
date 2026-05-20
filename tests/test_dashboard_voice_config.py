"""Dashboard voice-config API for Jarvis shell."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_dashboard_voice_config_route():
    from desktop_app.memory_viewer import app

    fake = MagicMock(
        auto_start_listening=False,
        ptt_enabled=True,
        ptt_hotkey="ctrl+shift+j",
        continuous_listening=False,
        whisper_lazy_load=True,
        whisper_model="small",
        wake_word="Jarvis",
    )
    with patch("desktop_app.memory_viewer.load_settings", return_value=fake):
        client = app.test_client()
        resp = client.get("/api/dashboard/voice-config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["continuous_listening"] is False
        assert data["whisper_lazy_load"] is True
        assert "Ctrl" in data["ptt_hotkey_display"]


@pytest.mark.unit
def test_whisper_lazy_load_default_in_settings():
    from jarvis.config import get_default_config

    assert get_default_config()["whisper_lazy_load"] is False
