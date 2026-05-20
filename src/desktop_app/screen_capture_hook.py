"""Run screen-capture work on the Qt main thread (Windows + ImageGrab safety)."""

from __future__ import annotations

from typing import Callable


def install_screen_capture_hook() -> None:
    """Register a main-thread runner with ``jarvis.screen_capture``."""
    try:
        from PyQt6.QtCore import QEventLoop, QThread, QTimer
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return

    from jarvis.screen_capture import register_main_thread_capture_runner

    def run_on_main_thread(work: Callable[[], bool]) -> bool:
        app = QApplication.instance()
        if app is None:
            return bool(work())
        if QThread.currentThread() == app.thread():
            return bool(work())
        result = {"ok": False}
        loop = QEventLoop()

        def run() -> None:
            try:
                result["ok"] = bool(work())
            except Exception:
                result["ok"] = False
            loop.quit()

        QTimer.singleShot(0, app, run)
        loop.exec()
        return bool(result["ok"])

    register_main_thread_capture_runner(run_on_main_thread)
