"""Tests for stale Command Centre detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestDashboardStaleRestart:
    def test_current_when_version_and_pulse_ok(self):
        from desktop_app.app import _dashboard_is_current

        payload = json.dumps({"dashboard_version": 6, "work_queue": {}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("desktop_app.memory_viewer.DASHBOARD_VERSION", 6):
            with patch("desktop_app.app._pulse_route_available", return_value=True):
                with patch("urllib.request.urlopen", return_value=mock_resp):
                    assert _dashboard_is_current(5050) is True

    def test_stale_when_version_mismatch(self):
        from desktop_app.app import _dashboard_is_current

        payload = json.dumps(
            {"dashboard_version": 2, "work_queue": {}, "operator": {}}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("desktop_app.memory_viewer.DASHBOARD_VERSION", 6):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                assert _dashboard_is_current(5050) is False

    def test_stale_when_pulse_missing(self):
        from desktop_app.app import _dashboard_is_current

        payload = json.dumps({"dashboard_version": 6, "work_queue": {}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("desktop_app.memory_viewer.DASHBOARD_VERSION", 6):
            with patch("desktop_app.app._pulse_route_available", return_value=False):
                with patch("urllib.request.urlopen", return_value=mock_resp):
                    assert _dashboard_is_current(5050) is False

    def test_stale_when_operator_fields_missing_and_no_version(self):
        from desktop_app.app import _dashboard_is_current

        payload = json.dumps({"presence": {}, "tools": {}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _dashboard_is_current(5050) is False
