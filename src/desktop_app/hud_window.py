"""
HUD-style Jarvis face window.

Hosts desktop_assets/face_hud.html in a QWebEngineView and drives it with the
real assistant state (JarvisStateManager, see face_widget.py) and live daemon
log lines, rather than the HTML's own placeholder data. See the comments
around window.setJarvisState / window.addJarvisLogLine in face_hud.html for
the JS side of this contract.

Falls back to the original low-poly painted face (LowPolyFaceWidget) when
QtWebEngine isn't available, so the window still shows something useful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QApplication

from jarvis.debug import debug_log
from desktop_app.face_widget import LowPolyFaceWidget, get_jarvis_state

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    QWebEngineView = None
    HAS_WEBENGINE = False

# How often to check for a Jarvis state change. Polling (rather than only
# relying on JarvisStateManager's Qt signal) is required because in dev mode
# the daemon runs as a separate process and only communicates state via the
# cross-process state file that JarvisStateManager reads.
_STATE_POLL_INTERVAL_MS = 300


class HudFaceWindow(QWidget):
    """Standalone window hosting the HTML/JS Jarvis HUD."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Jarvis")
        self.setMinimumSize(900, 600)
        self.resize(1400, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._last_state: Optional[str] = None
        self._view: Optional["QWebEngineView"] = None

        if HAS_WEBENGINE:
            self._view = QWebEngineView()
            html_path = Path(__file__).parent / "desktop_assets" / "face_hud.html"
            self._view.load(QUrl.fromLocalFile(str(html_path)))
            layout.addWidget(self._view)
        else:
            debug_log("QtWebEngine unavailable, HUD falling back to low-poly face", "desktop")
            layout.addWidget(LowPolyFaceWidget())

        self._position_on_right()

        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._poll_state)
        self._state_timer.start(_STATE_POLL_INTERVAL_MS)
        self._poll_state()

    def _position_on_right(self) -> None:
        """Position the window on the right side of the screen, vertically centered."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return

        geo = screen.availableGeometry()
        x = geo.right() - self.width() - 20
        y = geo.top() + (geo.height() - self.height()) // 2
        self.move(max(geo.left(), x), max(geo.top(), y))

    def _poll_state(self) -> None:
        state = get_jarvis_state().state.value
        if state == self._last_state:
            return
        self._last_state = state
        self._run_js(f"window.setJarvisState && window.setJarvisState({json.dumps(state)});")

    def add_log_line(self, line: str) -> None:
        """Forward a real daemon log line into the HUD's system log panel."""
        line = line.strip()
        if not line:
            return
        self._run_js(f"window.addJarvisLogLine && window.addJarvisLogLine({json.dumps(line)});")

    def _run_js(self, script: str) -> None:
        if self._view is None:
            return
        try:
            self._view.page().runJavaScript(script)
        except RuntimeError as e:
            debug_log(f"HUD runJavaScript failed: {e}", "desktop")
