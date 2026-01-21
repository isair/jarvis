"""
Update notification and download progress dialogs.
"""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .themes import COLORS, JARVIS_THEME_STYLESHEET
from .updater import (
    DownloadSignals,
    DownloadWorker,
    ReleaseInfo,
    UpdateStatus,
    install_update,
    save_installed_asset_id,
)


class UpdateAvailableDialog(QDialog):
    """Dialog shown when an update is available."""

    def __init__(self, status: UpdateStatus, parent=None):
        super().__init__(parent)
        self.status = status
        self.release = status.latest_release
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Update Available")
        self.setMinimumSize(500, 450)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Update Available")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 20px; font-weight: 600; color: {COLORS['accent_secondary']};")
        layout.addWidget(title)

        # Version info card
        version_frame = QFrame()
        version_frame.setObjectName("card")
        version_layout = QVBoxLayout(version_frame)
        version_layout.setSpacing(8)

        current_label = QLabel(f"Current version: {self.status.current_version}")
        current_label.setObjectName("subtitle")
        version_layout.addWidget(current_label)

        new_label = QLabel(f"New version: {self.release.version}")
        new_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: 500;")
        version_layout.addWidget(new_label)

        if self.release.prerelease:
            prerelease_label = QLabel("(Development Build)")
            prerelease_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
            version_layout.addWidget(prerelease_label)

        layout.addWidget(version_frame)

        # Release notes section
        notes_label = QLabel("What's New")
        notes_label.setObjectName("section_title")
        layout.addWidget(notes_label)

        notes_text = QTextEdit()
        notes_text.setReadOnly(True)
        notes_text.setMarkdown(self.release.release_notes or "No release notes available.")
        notes_text.setMaximumHeight(180)
        layout.addWidget(notes_text)

        # Size info
        size_mb = self.release.asset_size / (1024 * 1024)
        size_label = QLabel(f"Download size: {size_mb:.1f} MB")
        size_label.setObjectName("subtitle")
        layout.addWidget(size_label)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()

        later_btn = QPushButton("Later")
        later_btn.clicked.connect(self.reject)
        button_layout.addWidget(later_btn)

        button_layout.addStretch()

        view_btn = QPushButton("View on GitHub")
        view_btn.clicked.connect(self._open_github)
        button_layout.addWidget(view_btn)

        update_btn = QPushButton("Update Now")
        update_btn.setObjectName("primary")
        update_btn.clicked.connect(self.accept)
        button_layout.addWidget(update_btn)

        layout.addLayout(button_layout)

    def _open_github(self):
        webbrowser.open(self.release.html_url)


class UpdateProgressDialog(QDialog):
    """Dialog showing download and installation progress."""

    def __init__(self, release: ReleaseInfo, parent=None):
        super().__init__(parent)
        self.release = release
        self.download_worker: Optional[DownloadWorker] = None
        self.download_signals = DownloadSignals()
        self.download_path: Optional[Path] = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("Updating Jarvis")
        self.setMinimumSize(450, 220)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        self.title_label = QLabel("Downloading Update")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {COLORS['accent_secondary']};")
        layout.addWidget(self.title_label)

        # Status
        self.status_label = QLabel("Preparing download...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(12)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_download)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _connect_signals(self):
        self.download_signals.progress.connect(self._on_progress)
        self.download_signals.completed.connect(self._on_completed)
        self.download_signals.error.connect(self._on_error)

    def start_download(self):
        """Start the download process."""
        self.download_path = Path(tempfile.mkdtemp()) / self.release.asset_name

        self.download_worker = DownloadWorker(
            self.release.download_url,
            self.download_path,
            self.download_signals,
        )
        self.download_worker.start()

    def _on_progress(self, downloaded: int, total: int):
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)

            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self.status_label.setText(f"Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB")

    def _on_completed(self, path: str):
        self.title_label.setText("Installing Update")
        self.status_label.setText("Installing update...")
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.cancel_btn.setEnabled(False)

        # Short delay before install
        QTimer.singleShot(500, lambda: self._install(Path(path)))

    def _install(self, download_path: Path):
        if install_update(download_path):
            # Save the asset ID so we don't prompt again for this version
            save_installed_asset_id(self.release.asset_id)

            self.title_label.setText("Update Complete")
            self.status_label.setText("Update installed! Restarting...")
            self.status_label.setStyleSheet(f"color: {COLORS['success']};")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

            # App will be relaunched by installer, exit current
            QTimer.singleShot(1500, lambda: self.done(QDialog.DialogCode.Accepted))
        else:
            self._on_error("Installation failed. Please try again or update manually.")

    def _on_error(self, error: str):
        self.title_label.setText("Update Failed")
        self.status_label.setText(f"Error: {error}")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.cancel_btn.setText("Close")
        self.cancel_btn.setEnabled(True)

    def _cancel_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_worker.wait()
        self.reject()

    def closeEvent(self, event):
        self._cancel_download()
        event.accept()


def show_no_update_dialog(current_version: str, parent=None) -> None:
    """Show a dialog indicating no updates are available."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowTitle("No Updates Available")
    msg.setText(f"You're running the latest version ({current_version})")
    msg.setStyleSheet(JARVIS_THEME_STYLESHEET)
    msg.exec()


def show_update_error_dialog(error: str, parent=None) -> None:
    """Show a dialog indicating an update check error."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Update Check Failed")
    msg.setText("Could not check for updates")
    msg.setInformativeText(error)
    msg.setStyleSheet(JARVIS_THEME_STYLESHEET)
    msg.exec()
