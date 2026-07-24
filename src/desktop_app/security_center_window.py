"""Qt shell for Cora Security Center (localhost Flask)."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from jarvis.debug import debug_log

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:
    QWebEngineView = None  # type: ignore
    HAS_WEBENGINE = False


class SecurityCenterWindow(QMainWindow):
    def __init__(self, port: int = 5051) -> None:
        super().__init__()
        self.port = int(port) if 1024 <= int(port) <= 65535 else 5051
        self.setWindowTitle("🛡️ Cora Security Center")
        self.setGeometry(160, 160, 1100, 820)
        self.is_server_running = False
        self.server_thread: Optional[threading.Thread] = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = None
        if HAS_WEBENGINE and not (sys.platform == "darwin" and getattr(sys, "frozen", False)):
            try:
                self.web_view = QWebEngineView()
                layout.addWidget(self.web_view)
            except Exception as exc:
                debug_log(f"security center webengine failed: {exc}", "desktop")
                self.web_view = None

        if self.web_view is None:
            label = QLabel("Opening Security Center in your browser…\n(READ-ONLY · localhost only)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size:16px;color:#fbbf24;padding:40px;")
            layout.addWidget(label)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self.start_server():
            url = f"http://127.0.0.1:{self.port}/"
            if self.web_view is not None:
                self.web_view.setUrl(__import__("PyQt6.QtCore", fromlist=["QUrl"]).QUrl(url))
            else:
                webbrowser.open(url)

    def start_server(self) -> bool:
        if self.is_server_running:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            in_use = sock.connect_ex(("127.0.0.1", self.port)) == 0
        finally:
            sock.close()
        if in_use:
            self.is_server_running = True
            return True

        try:
            from desktop_app.security_center import app as flask_app, get_service

            # Sync port into security config (localhost forced inside config)
            svc = get_service()
            svc.config.bind_host = "127.0.0.1"
            svc.config.bind_port = self.port

            def _run() -> None:
                import logging

                logging.getLogger("werkzeug").setLevel(logging.ERROR)
                flask_app.run(
                    host="127.0.0.1",
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True,
                )

            self.server_thread = threading.Thread(target=_run, name="cora-security-flask", daemon=True)
            self.server_thread.start()
            # Wait briefly for bind
            for _ in range(20):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    if s.connect_ex(("127.0.0.1", self.port)) == 0:
                        self.is_server_running = True
                        return True
                finally:
                    s.close()
                time.sleep(0.1)
            self.is_server_running = True  # best-effort
            return True
        except Exception as exc:
            debug_log(f"security center server failed: {exc}", "desktop")
            return False
