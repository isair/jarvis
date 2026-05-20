"""HTTP settings API for Jarvis shell native panel."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_export_settings_metadata_includes_voice_fields():
    from desktop_app.settings_api import export_settings_bundle

    bundle = export_settings_bundle()
    assert bundle["ok"] is True
    keys = [f["key"] for f in bundle["fields"]]
    assert "whisper_lazy_load" in keys
    assert "ptt_enabled" in keys
    cats = [c["id"] for c in bundle["categories"]]
    assert "voice_input" in cats
    assert "mcps" not in cats


@pytest.mark.unit
def test_merged_settings_values_uses_defaults():
    from desktop_app.settings_api import build_merged_values

    with patch("desktop_app.settings_api.load_config", return_value={}):
        values = build_merged_values()
    assert values["ptt_enabled"] is True
    assert values["whisper_lazy_load"] is False


@pytest.mark.unit
def test_save_settings_from_values_writes_non_default(tmp_path):
    from desktop_app.settings_api import save_settings_from_values

    cfg_path = tmp_path / "config.json"
    with patch("desktop_app.settings_api.default_config_path", return_value=cfg_path), patch(
        "desktop_app.settings_api.load_config", return_value={}
    ):
        ok, msg = save_settings_from_values(
            {"whisper_lazy_load": True, "continuous_listening": False}
        )
    assert ok is True
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["whisper_lazy_load"] is True
    assert data["continuous_listening"] is False
    assert "ptt_enabled" not in data


@pytest.mark.unit
def test_build_default_values_matches_merged_when_config_empty():
    from desktop_app.settings_api import build_default_values, build_merged_values

    with patch("desktop_app.settings_api.load_config", return_value={}):
        defaults = build_default_values()
        merged = build_merged_values()
    assert defaults["ptt_enabled"] == merged["ptt_enabled"]


@pytest.mark.unit
def test_settings_defaults_route():
    from desktop_app.memory_viewer import app

    client = app.test_client()
    resp = client.get("/api/settings/defaults")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "whisper_lazy_load" in data["values"]


@pytest.mark.unit
def test_settings_metadata_route():
    from desktop_app.memory_viewer import app

    client = app.test_client()
    resp = client.get("/api/settings/metadata")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert len(data["fields"]) > 10


@pytest.mark.unit
def test_settings_config_get_and_post_roundtrip(tmp_path):
    from desktop_app.memory_viewer import app

    cfg_path = tmp_path / "config.json"
    with patch("desktop_app.settings_api.default_config_path", return_value=cfg_path), patch(
        "desktop_app.settings_api.load_config", return_value={}
    ):
        client = app.test_client()
        get_resp = client.get("/api/settings/config")
        assert get_resp.status_code == 200
        body = get_resp.get_json()
        assert body["ok"] is True
        values = dict(body["values"])
        values["whisper_lazy_load"] = True

        post_resp = client.post(
            "/api/settings/config",
            data=json.dumps({"values": values}),
            content_type="application/json",
        )
        assert post_resp.status_code == 200
        assert post_resp.get_json()["ok"] is True

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["whisper_lazy_load"] is True
