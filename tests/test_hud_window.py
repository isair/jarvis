"""
Tests for HudFaceWindow (src/desktop_app/hud_window.py) - the QWebEngineView
face window that hosts face_hud.html and drives its JS hooks
(window.setJarvisState / window.addJarvisLogLine) with real assistant state.
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_window():
    """Build a HudFaceWindow without running Qt widget __init__."""
    from desktop_app.hud_window import HudFaceWindow

    window = HudFaceWindow.__new__(HudFaceWindow)
    window._view = MagicMock()
    window._last_state = None
    return window


class TestPollState:
    """_poll_state() reads the real JarvisStateManager state and forwards it
    to the HUD's window.setJarvisState JS hook, only on change."""

    def test_calls_set_jarvis_state_with_current_state(self):
        window = _make_window()
        mock_state_mgr = MagicMock()
        mock_state_mgr.state.value = "listening"

        with patch("desktop_app.hud_window.get_jarvis_state", return_value=mock_state_mgr):
            window._poll_state()

        window._view.page.return_value.runJavaScript.assert_called_once()
        script = window._view.page.return_value.runJavaScript.call_args[0][0]
        assert "setJarvisState" in script
        assert '"listening"' in script

    def test_skips_js_call_when_state_unchanged(self):
        window = _make_window()
        window._last_state = "idle"
        mock_state_mgr = MagicMock()
        mock_state_mgr.state.value = "idle"

        with patch("desktop_app.hud_window.get_jarvis_state", return_value=mock_state_mgr):
            window._poll_state()

        window._view.page.return_value.runJavaScript.assert_not_called()

    def test_updates_last_state_on_change(self):
        window = _make_window()
        window._last_state = "idle"
        mock_state_mgr = MagicMock()
        mock_state_mgr.state.value = "thinking"

        with patch("desktop_app.hud_window.get_jarvis_state", return_value=mock_state_mgr):
            window._poll_state()

        assert window._last_state == "thinking"


class TestAddLogLine:
    """add_log_line() forwards real daemon log lines to the HUD's
    window.addJarvisLogLine JS hook."""

    def test_forwards_non_empty_line(self):
        window = _make_window()
        window.add_log_line("daemon started")

        window._view.page.return_value.runJavaScript.assert_called_once()
        script = window._view.page.return_value.runJavaScript.call_args[0][0]
        assert "addJarvisLogLine" in script
        assert "daemon started" in script

    def test_skips_blank_line(self):
        window = _make_window()
        window.add_log_line("   \n")

        window._view.page.return_value.runJavaScript.assert_not_called()

    def test_escapes_quotes_and_special_characters_safely(self):
        """Log content is untrusted (daemon stdout) so it must be embedded as
        a properly-escaped JS string literal, not concatenated raw."""
        window = _make_window()
        window.add_log_line('bad "quote" and \\ backslash and \n newline')

        script = window._view.page.return_value.runJavaScript.call_args[0][0]
        # json.dumps produces a single valid JS/JSON string literal argument
        import re
        match = re.search(r"addJarvisLogLine\((.*)\);", script)
        assert match is not None
        import json
        parsed = json.loads(match.group(1))
        assert "quote" in parsed


class TestNoWebEngineFallback:
    """When QtWebEngine isn't available, _run_js must no-op instead of
    raising, since there's no page() to call."""

    def test_run_js_noop_when_view_is_none(self):
        window = _make_window()
        window._view = None
        # Should not raise
        window._run_js("window.setJarvisState('idle');")
