"""auto_start_listening config default and load behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import get_default_config, load_settings


def test_default_config_auto_start_listening_is_false() -> None:
    assert get_default_config()["auto_start_listening"] is False


@pytest.mark.unit
def test_load_settings_auto_start_listening_false_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg)
    s = load_settings()
    assert s.auto_start_listening is False


@pytest.mark.unit
def test_load_settings_auto_start_listening_true_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"auto_start_listening": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg)
    s = load_settings()
    assert s.auto_start_listening is True
