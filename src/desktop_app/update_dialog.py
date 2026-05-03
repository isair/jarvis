"""
Update notification and download progress dialogs.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
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

# ---------------------------------------------------------------------------
# Changelog parsing
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "feat":     ("✨", "New Features"),
    "feature":  ("✨", "New Features"),
    "fix":      ("🐛", "Bug Fixes"),
    "perf":     ("⚡", "Performance"),
    "refactor": ("♻️", "Improvements"),
    "improve":  ("♻️", "Improvements"),
    "security": ("🔒", "Security"),
    "docs":     ("📝", "Documentation"),
    "chore":    ("🔧", "Maintenance"),
    "ci":       ("🔧", "Maintenance"),
    "build":    ("🔧", "Maintenance"),
    "deps":     ("🔧", "Maintenance"),
    "test":     ("🧪", "Testing"),
    "style":    ("🎨", "Style"),
    "revert":   ("⏪", "Reverts"),
}

_CATEGORY_ORDER = [
    "New Features", "Bug Fixes", "Performance", "Improvements",
    "Security", "Documentation", "Maintenance", "Testing", "Style",
    "Reverts", "Changes",
]

_DEFAULT_CATEGORY = ("📋", "Changes")


@dataclass
class ChangelogEntry:
    text: str
    pr_number: Optional[int]
    category_emoji: str
    category_name: str


def _detect_category(raw: str) -> tuple[str, str, str]:
    """Return (emoji, category_name, cleaned_text) for a raw change line."""
    m = re.match(r'^(\w+)(?:\([^)]+\))?!?\s*:\s*(.+)$', raw.strip(), re.IGNORECASE)
    if m:
        ctype = m.group(1).lower()
        clean = m.group(2).strip()
        if ctype in _CATEGORY_MAP:
            emoji, name = _CATEGORY_MAP[ctype]
            return emoji, name, clean
    return _DEFAULT_CATEGORY[0], _DEFAULT_CATEGORY[1], raw.strip()


def parse_release_notes(notes: str) -> dict[str, list[ChangelogEntry]]:
    """Parse GitHub release markdown into categorised changelog entries.

    Handles both GitHub's auto-generated format
    (``* fix(x): desc by @user in https://.../pull/NNN``) and manually
    written conventional-commit bullets.  Returns an ordered dict keyed by
    category name.
    """
    # Strip "Full Changelog" footer
    notes = re.sub(r'\*\*Full Changelog\*\*.*$', '', notes, flags=re.MULTILINE).strip()

    entries: list[ChangelogEntry] = []
    for line in notes.splitlines():
        line = line.strip()
        if not re.match(r'^[*\-+]\s', line):
            continue

        text = line[2:].strip()

        # GitHub auto-generated: "... by @user in https://.../pull/NNN"
        m_gh = re.search(r'\s+by\s+@\w+\s+in\s+https?://\S+/pull/(\d+)\s*$', text)
        if m_gh:
            pr_number: Optional[int] = int(m_gh.group(1))
            text = text[: m_gh.start()].strip()
        else:
            pr_number = None
            # Plain attribution: "... by @user"
            text = re.sub(r'\s+by\s+@\w+\s*$', '', text).strip()
            # Inline PR ref: "... (#NNN)"
            m_pr = re.search(r'\s*\(#(\d+)\)\s*$', text)
            if m_pr:
                pr_number = int(m_pr.group(1))
                text = text[: m_pr.start()].strip()

        if not text:
            continue

        emoji, cat_name, clean_text = _detect_category(text)
        entries.append(ChangelogEntry(
            text=clean_text,
            pr_number=pr_number,
            category_emoji=emoji,
            category_name=cat_name,
        ))

    # Group preserving priority order
    buckets: dict[str, list[ChangelogEntry]] = {}
    for entry in entries:
        buckets.setdefault(entry.category_name, []).append(entry)

    return {name: buckets[name] for name in _CATEGORY_ORDER if name in buckets}


# ---------------------------------------------------------------------------
# Changelog widget
# ---------------------------------------------------------------------------

class _ClickableFrame(QFrame):
    """QFrame that calls a Python callable on left-click."""

    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click()
        super().mousePressEvent(event)


class _VersionCard(QFrame):
    """Collapsible card showing the changelog for one release version."""

    def __init__(
        self,
        release: ReleaseInfo,
        is_latest: bool,
        expanded: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._release = release
        self._expanded = expanded
        self._parsed = parse_release_notes(release.release_notes or "")
        self._setup_ui(is_latest)

    def _setup_ui(self, is_latest: bool) -> None:
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        # Clickable header
        header = _ClickableFrame(self._toggle)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
            QFrame:hover {{
                background-color: {COLORS['bg_hover']};
                border-color: {COLORS['border_glow']};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 10, 14, 10)
        h_layout.setSpacing(8)

        version_badge = QLabel(f" v{self._release.version} ")
        version_badge.setStyleSheet(f"""
            background-color: {COLORS['accent_glow']};
            color: {COLORS['accent_secondary']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            padding: 2px 6px;
        """)
        h_layout.addWidget(version_badge)

        name = self._release.name or ""
        redundant = {self._release.tag_name, f"v{self._release.version}", self._release.version}
        if name and name not in redundant:
            name_label = QLabel(name)
            name_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent;"
            )
            h_layout.addWidget(name_label)

        h_layout.addStretch()

        if is_latest:
            latest_badge = QLabel("  LATEST  ")
            latest_badge.setStyleSheet(f"""
                background-color: rgba(34, 197, 94, 0.12);
                color: {COLORS['success']};
                border: 1px solid rgba(34, 197, 94, 0.3);
                border-radius: 4px;
                font-size: 10px;
                font-weight: 700;
                padding: 2px 6px;
            """)
            h_layout.addWidget(latest_badge)

        if self._release.prerelease:
            dev_badge = QLabel("  DEV  ")
            dev_badge.setStyleSheet(f"""
                background-color: {COLORS['accent_glow']};
                color: {COLORS['warning']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 4px;
                font-size: 10px;
                font-weight: 700;
                padding: 2px 6px;
            """)
            h_layout.addWidget(dev_badge)

        self._arrow = QLabel("▾" if self._expanded else "▸")
        self._arrow.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; "
            f"padding-left: 4px; background: transparent;"
        )
        h_layout.addWidget(self._arrow)

        outer.addWidget(header)

        # Collapsible content
        self._content = QWidget()
        self._content.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-top: none;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
        """)
        c_layout = QVBoxLayout(self._content)
        c_layout.setSpacing(4)
        c_layout.setContentsMargins(16, 10, 16, 14)

        if self._parsed:
            first_cat = True
            for cat_name, cat_entries in self._parsed.items():
                if not cat_entries:
                    continue

                cat_row = QHBoxLayout()
                cat_row.setContentsMargins(0, 0 if first_cat else 8, 0, 2)

                emoji_lbl = QLabel(cat_entries[0].category_emoji)
                emoji_lbl.setStyleSheet("font-size: 13px; background: transparent;")
                cat_row.addWidget(emoji_lbl)

                cat_lbl = QLabel(cat_name)
                cat_lbl.setStyleSheet(
                    f"color: {COLORS['text_primary']}; font-size: 12px; "
                    f"font-weight: 600; background: transparent;"
                )
                cat_row.addWidget(cat_lbl)
                cat_row.addStretch()
                c_layout.addLayout(cat_row)
                first_cat = False

                for entry in cat_entries:
                    row = QHBoxLayout()
                    row.setContentsMargins(12, 0, 0, 0)
                    row.setSpacing(6)

                    bullet = QLabel("·")
                    bullet.setFixedWidth(10)
                    bullet.setStyleSheet(
                        f"color: {COLORS['accent_muted']}; font-size: 14px; background: transparent;"
                    )
                    row.addWidget(bullet)

                    text_lbl = QLabel(entry.text)
                    text_lbl.setTextFormat(Qt.TextFormat.PlainText)
                    text_lbl.setWordWrap(True)
                    text_lbl.setSizePolicy(
                        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
                    )
                    text_lbl.setStyleSheet(
                        f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;"
                    )
                    row.addWidget(text_lbl, 1)

                    if entry.pr_number:
                        pr_lbl = QLabel(f"#{entry.pr_number}")
                        pr_lbl.setStyleSheet(f"""
                            color: {COLORS['text_muted']};
                            background-color: {COLORS['bg_tertiary']};
                            border-radius: 3px;
                            font-size: 10px;
                            padding: 1px 5px;
                        """)
                        row.addWidget(pr_lbl)

                    c_layout.addLayout(row)
        else:
            placeholder = QLabel("No release notes available.")
            placeholder.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;"
            )
            c_layout.addWidget(placeholder)

        self._content.setVisible(self._expanded)
        outer.addWidget(self._content)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._arrow.setText("▾" if self._expanded else "▸")
        # Tell the scroll-area container to recompute its size
        p = self.parent()
        while p:
            if isinstance(p, QScrollArea):
                p.widget().adjustSize()
                break
            p = p.parent()


class ChangelogWidget(QScrollArea):
    """Scrollable accordion list of version changelog cards."""

    def __init__(self, releases: list[ReleaseInfo], parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 4, 0)

        for i, release in enumerate(releases):
            card = _VersionCard(
                release=release,
                is_latest=(i == 0),
                expanded=(i == 0),
            )
            layout.addWidget(card)

        layout.addStretch()
        self.setWidget(container)


# ---------------------------------------------------------------------------
# Main update dialog
# ---------------------------------------------------------------------------

class UpdateAvailableDialog(QDialog):
    """Dialog shown when an update is available."""

    def __init__(self, status: UpdateStatus, parent=None):
        super().__init__(parent)
        self.status = status
        self.release = status.latest_release
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Update Available")
        self.setMinimumSize(540, 520)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Update Available")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 600; color: {COLORS['accent_secondary']};"
        )
        layout.addWidget(title)

        # Version + download-size row
        info_frame = QFrame()
        info_frame.setObjectName("card")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(14, 10, 14, 10)

        ver_col = QVBoxLayout()
        ver_col.setSpacing(4)
        current_lbl = QLabel(f"Current version:  {self.status.current_version}")
        current_lbl.setObjectName("subtitle")
        ver_col.addWidget(current_lbl)
        new_lbl = QLabel(f"New version:  {self.release.version}")
        new_lbl.setStyleSheet(f"color: {COLORS['success']}; font-weight: 500;")
        ver_col.addWidget(new_lbl)
        if self.release.prerelease:
            dev_lbl = QLabel("Development build")
            dev_lbl.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
            ver_col.addWidget(dev_lbl)
        info_layout.addLayout(ver_col)

        info_layout.addStretch()

        size_mb = self.release.asset_size / (1024 * 1024)
        size_lbl = QLabel(f"{size_mb:.1f} MB")
        size_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        size_lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        info_layout.addWidget(size_lbl)

        layout.addWidget(info_frame)

        # Changelog section
        releases = self.status.releases_since_current or (
            [self.release] if self.release else []
        )
        section_title = (
            f"Changes since v{self.status.current_version}"
            if len(releases) > 1
            else "What's New"
        )
        notes_label = QLabel(section_title)
        notes_label.setObjectName("section_title")
        layout.addWidget(notes_label)

        changelog = ChangelogWidget(releases)
        changelog.setMinimumHeight(200)
        changelog.setMaximumHeight(340)
        layout.addWidget(changelog, 1)

        layout.addStretch(0)

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


# ---------------------------------------------------------------------------
# Progress dialog
# ---------------------------------------------------------------------------

class UpdateProgressDialog(QDialog):
    """Dialog showing download and installation progress."""

    def __init__(self, release: ReleaseInfo, pre_install_callback=None, parent=None):
        """Initialise the update progress dialog.

        Args:
            release: The release info to download and install.
            pre_install_callback: Optional callback called after download completes
                but before installation starts. Use this to save state (e.g., diary)
                before the update process begins. The callback should be synchronous.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.release = release
        self._pre_install_callback = pre_install_callback
        self.download_worker: Optional[DownloadWorker] = None
        self.download_signals = DownloadSignals()
        self.download_path: Optional[Path] = None
        self._temp_dir: Optional[Path] = None
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

        self.title_label = QLabel("Downloading Update")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {COLORS['accent_secondary']};"
        )
        layout.addWidget(self.title_label)

        self.status_label = QLabel("Preparing download...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(12)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_download)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _connect_signals(self):
        self.download_signals.progress.connect(self._on_progress)
        self.download_signals.completed.connect(self._on_completed)
        self.download_signals.error.connect(self._on_error)

    def start_download(self):
        """Start the download process."""
        self._temp_dir = Path(tempfile.mkdtemp())
        self.download_path = self._temp_dir / self.release.asset_name

        self.download_worker = DownloadWorker(
            self.release.download_url,
            self.download_path,
            self.download_signals,
        )
        self.download_worker.start()

    def _cleanup_temp_dir(self):
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None

    def _on_progress(self, downloaded: int, total: int):
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self.status_label.setText(
                f"Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB"
            )

    def _on_completed(self, path: str):
        self.cancel_btn.setEnabled(False)

        if self._pre_install_callback:
            self.title_label.setText("Preparing Update")
            self.status_label.setText("Saving your session...")
            self.progress_bar.setRange(0, 0)

            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

            try:
                self._pre_install_callback()
            except Exception as e:
                from jarvis.debug import debug_log
                debug_log(f"Pre-install callback failed: {e}", "updater")

        self.title_label.setText("Installing Update")
        self.status_label.setText("Installing update...")
        self.progress_bar.setRange(0, 0)

        QTimer.singleShot(500, lambda: self._install(Path(path)))

    def _install(self, download_path: Path):
        if install_update(download_path):
            save_installed_asset_id(self.release.asset_id)

            self.title_label.setText("Update Complete")
            self.status_label.setText("Update installed! Restarting...")
            self.status_label.setStyleSheet(f"color: {COLORS['success']};")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

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
        self._cleanup_temp_dir()

    def _cancel_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_worker.wait()
        self._cleanup_temp_dir()
        self.reject()

    def closeEvent(self, event):
        self._cancel_download()
        event.accept()


# ---------------------------------------------------------------------------
# Utility dialogs
# ---------------------------------------------------------------------------

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
