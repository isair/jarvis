#!/usr/bin/env python3
"""Open Jarvis settings window (used by Tauri shell)."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from desktop_app.settings_window import SettingsWindow


def main() -> int:
    app = QApplication(sys.argv)
    dialog = SettingsWindow()
    dialog.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
