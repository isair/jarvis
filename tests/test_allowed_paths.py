"""Tests for configured file access roots."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.utils.allowed_paths import is_path_allowed, roots_from_settings


@pytest.mark.unit
class TestAllowedPaths:
    def test_home_always_allowed(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("jarvis.utils.allowed_paths.Path.home", lambda: home)

        cfg = MagicMock()
        cfg.data_live_roots = []
        roots = roots_from_settings(cfg)
        target = home / "docs" / "a.txt"
        target.parent.mkdir(parents=True)
        target.write_text("hi", encoding="utf-8")

        assert is_path_allowed(target, roots) is True

    def test_data_live_root_allowed(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.setattr("jarvis.utils.allowed_paths.Path.home", lambda: home)

        cfg = MagicMock()
        cfg.data_live_roots = [{"path": str(work), "label": "Work"}]
        roots = roots_from_settings(cfg)
        file = work / "invoice.json"
        file.write_text("{}", encoding="utf-8")

        assert is_path_allowed(file, roots) is True

    def test_outside_roots_denied(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("jarvis.utils.allowed_paths.Path.home", lambda: home)

        cfg = MagicMock()
        cfg.data_live_roots = []
        roots = roots_from_settings(cfg)
        outside = tmp_path / "outside" / "secret.txt"
        outside.parent.mkdir()
        outside.write_text("x", encoding="utf-8")

        assert is_path_allowed(outside, roots) is False
