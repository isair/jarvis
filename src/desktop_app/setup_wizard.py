"""
Jarvis Setup Wizard

A setup wizard that checks for Ollama installation, running server, and required models.
Guides users through the setup process with automated actions where possible.
"""

from __future__ import annotations
import subprocess
import shutil
import sys
import os
import platform
import webbrowser
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto

import requests

from jarvis.config import SUPPORTED_CHAT_MODELS, DEFAULT_CHAT_MODEL


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def check_ffmpeg_installed() -> Tuple[bool, Optional[str]]:
    """Check if FFmpeg is installed (required for MLX Whisper)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return True, ffmpeg_path

    # Check common macOS paths
    macos_paths = [
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for path in macos_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return True, path

    return False, None


def check_mlx_whisper_installed() -> bool:
    """Check if mlx-whisper is installed."""
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False


@dataclass
class MLXWhisperStatus:
    """Status of MLX Whisper setup."""
    is_apple_silicon: bool = False
    is_ffmpeg_installed: bool = False
    ffmpeg_path: Optional[str] = None
    is_mlx_whisper_installed: bool = False

    @property
    def is_fully_setup(self) -> bool:
        """Check if MLX Whisper is fully set up."""
        if not self.is_apple_silicon:
            return True  # Not applicable on non-Apple Silicon
        return self.is_ffmpeg_installed and self.is_mlx_whisper_installed


def check_mlx_whisper_status() -> MLXWhisperStatus:
    """Check MLX Whisper setup status."""
    status = MLXWhisperStatus()
    status.is_apple_silicon = is_apple_silicon()

    if status.is_apple_silicon:
        status.is_ffmpeg_installed, status.ffmpeg_path = check_ffmpeg_installed()
        status.is_mlx_whisper_installed = check_mlx_whisper_installed()

    return status


# Import config early (no PyQt6 dependency) - needed for detection functions
from jarvis.config import load_settings, get_default_config


class SetupStatus(Enum):
    """Status of a setup check."""
    PENDING = auto()
    CHECKING = auto()
    SUCCESS = auto()
    FAILED = auto()
    INSTALLING = auto()


@dataclass
class OllamaStatus:
    """Current status of Ollama setup."""
    is_cli_installed: bool = False
    cli_path: Optional[str] = None
    is_server_running: bool = False
    server_version: Optional[str] = None
    installed_models: List[str] = None
    missing_models: List[str] = None

    def __post_init__(self):
        if self.installed_models is None:
            self.installed_models = []
        if self.missing_models is None:
            self.missing_models = []

    @property
    def is_fully_setup(self) -> bool:
        """Check if Ollama is fully set up and ready."""
        return (
            self.is_cli_installed
            and self.is_server_running
            and len(self.missing_models) == 0
        )


def check_ollama_cli() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama CLI is installed.
    Returns (is_installed, path_to_ollama).
    """
    # Check common installation paths
    ollama_path = shutil.which("ollama")
    if ollama_path:
        return True, ollama_path

    # Check macOS-specific paths
    macos_paths = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        os.path.expanduser("~/bin/ollama"),
    ]

    for path in macos_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return True, path

    # Check Windows paths
    if sys.platform == "win32":
        windows_paths = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Ollama", "ollama.exe"),
        ]
        for path in windows_paths:
            if os.path.isfile(path):
                return True, path

    return False, None


def check_ollama_server() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama server is running.
    Returns (is_running, version).
    """
    try:
        cfg = load_settings()
        base_url = cfg.ollama_base_url
    except Exception:
        base_url = "http://127.0.0.1:11434"

    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        if response.status_code == 200:
            data = response.json()
            version = data.get("version", "unknown")
            return True, version
    except Exception:
        pass

    return False, None


def get_required_models() -> List[str]:
    """Get list of required Ollama models from config.

    Always includes:
    - Chat model (user-selectable)
    - Embedding model
    - Intent judge model (llama3.2:3b - required for voice intent classification)
    """
    try:
        cfg = load_settings()
        models = []

        # Chat model
        if cfg.ollama_chat_model:
            models.append(cfg.ollama_chat_model)

        # Embedding model
        if cfg.ollama_embed_model:
            models.append(cfg.ollama_embed_model)

        # Intent judge model - always required for voice intent classification
        # This is separate from the chat model and cannot be changed by users
        intent_judge_model = getattr(cfg, "intent_judge_model", "llama3.2:3b")
        if intent_judge_model and intent_judge_model not in models:
            models.append(intent_judge_model)

        return models
    except Exception:
        # Default models if config can't be loaded
        # Note: DEFAULT_CHAT_MODEL is llama3.2:3b which is also the intent judge model,
        # so the default list is effectively just 2 unique models
        defaults = [DEFAULT_CHAT_MODEL, "nomic-embed-text"]
        if "llama3.2:3b" not in defaults:
            defaults.append("llama3.2:3b")
        return defaults


def check_installed_models(ollama_path: Optional[str] = None) -> List[str]:
    """
    Get list of installed Ollama models.
    Returns list of model names.
    """
    if ollama_path is None:
        ollama_path = shutil.which("ollama") or "ollama"

    try:
        # Hide console window on Windows
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        result = subprocess.run(
            [ollama_path, "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30,
            creationflags=creationflags
        )

        if result.returncode != 0:
            return []

        # Parse output - format is "NAME ID SIZE MODIFIED"
        lines = result.stdout.strip().split("\n")
        models = []

        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    # Model name is the first column, may include :tag
                    model_name = parts[0]
                    models.append(model_name)

        return models
    except Exception:
        return []


def check_ollama_status() -> OllamaStatus:
    """Perform a complete check of Ollama status."""
    status = OllamaStatus()

    # Check CLI
    is_installed, cli_path = check_ollama_cli()
    status.is_cli_installed = is_installed
    status.cli_path = cli_path

    # Check server
    is_running, version = check_ollama_server()
    status.is_server_running = is_running
    status.server_version = version

    # Check models (only if CLI is installed AND server is running)
    # Running 'ollama list' when server isn't running causes it to hang
    if is_installed and is_running:
        required = get_required_models()
        installed = check_installed_models(cli_path)

        # Normalize model names (remove :latest suffix for comparison)
        def normalize_model(name: str) -> str:
            return name.split(":")[0] if ":" in name and name.endswith(":latest") else name

        installed_normalized = {normalize_model(m) for m in installed}

        status.installed_models = installed
        status.missing_models = [
            m for m in required
            if normalize_model(m) not in installed_normalized and m not in installed
        ]
    else:
        status.missing_models = get_required_models()

    return status


def should_show_setup_wizard() -> bool:
    """
    Check if the setup wizard should be shown.

    Returns True only if user intervention is needed:
    - CLI not installed (user must install Ollama)
    - Models missing (user must download models)

    Does NOT return True just because server isn't running,
    since the app can auto-start the server if CLI is installed.
    """
    status = check_ollama_status()

    # If CLI not installed, user needs to install Ollama
    if not status.is_cli_installed:
        return True

    # If server is running and models are missing, user needs to download them
    if status.is_server_running and len(status.missing_models) > 0:
        return True

    # If CLI is installed but server not running, we can start it ourselves
    # No need for wizard in this case
    return False


# --- PyQt6 UI components below ---
# These imports are wrapped to avoid import errors when only detection functions are needed
# (e.g., on headless CI systems where system Qt libraries may be missing)

import sys as _sys

try:
    from PyQt6.QtWidgets import (
        QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QProgressBar, QTextEdit, QWidget, QFrame,
        QSizePolicy, QScrollArea, QLineEdit, QSlider
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QPainter

    from desktop_app.themes import JARVIS_THEME_STYLESHEET, COLORS

    # Import location utilities with crash protection for Windows native modules
    try:
        from jarvis.utils.location import (
            get_location_info,
            get_location_context,
            is_location_available,
            _get_database_path,
            GEOIP2_AVAILABLE,
        )
    except Exception as e:
        if _sys.platform == 'win32':
            print(f"  ⚠️  Location utilities import failed: {e}", flush=True)
        # Provide stubs so the wizard can still run without location features
        get_location_info = lambda *a, **k: {}
        get_location_context = lambda *a, **k: "Location: Unknown"
        is_location_available = lambda: False
        _get_database_path = lambda: None
        GEOIP2_AVAILABLE = False

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    # Define stubs so module can be imported for detection functions only
    # These stubs allow the class definitions to parse without errors
    QThread = object
    QWizard = object
    QWizardPage = object
    QWidget = object
    QFrame = object
    Qt = None
    QTimer = None
    QObject = None

    def pyqtSignal(*args, **kwargs):
        """Stub for pyqtSignal when PyQt6 is not available."""
        return None

    # Stub location utilities that depend on themes
    JARVIS_THEME_STYLESHEET = ""
    COLORS = {}
    get_location_info = lambda *a, **k: {}
    get_location_context = lambda *a, **k: "Location: Unknown"
    is_location_available = lambda: False
    _get_database_path = lambda: None
    GEOIP2_AVAILABLE = False


class StatusCheckWorker(QThread):
    """Worker thread for checking Ollama status."""
    finished = pyqtSignal(OllamaStatus)

    def run(self):
        status = check_ollama_status()
        self.finished.emit(status)


class CommandWorker(QThread):
    """Worker thread for running commands."""
    output = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, command: List[str], parent=None):
        super().__init__(parent)
        self.command = command

    def run(self):
        try:
            # Use UTF-8 encoding with error replacement for cross-platform compatibility
            # Windows defaults to cp1252 which can't handle Ollama's UTF-8 output
            # Hide console window on Windows
            creationflags = 0
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                creationflags=creationflags
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    self.output.emit(line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.finished.emit(True, "✅ Command completed successfully")
            else:
                self.finished.emit(False, f"❌ Command failed with exit code {process.returncode}")
        except Exception as e:
            self.finished.emit(False, f"❌ Error: {str(e)}")


class SetupWizard(QWizard):
    """Main setup wizard window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Jarvis Setup Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(700, 800)

        # Apply dark theme
        self._apply_theme()

        # Add pages and store their IDs
        self.welcome_page = WelcomePage(self)
        self.ollama_install_page = OllamaInstallPage(self)
        self.ollama_server_page = OllamaServerPage(self)
        self.models_page = ModelsPage(self)
        self.mlx_whisper_page = WhisperSetupPage(self)
        self.location_page = LocationPage(self)
        self.complete_page = CompletePage(self)

        self.welcome_page_id = self.addPage(self.welcome_page)
        self.ollama_install_page_id = self.addPage(self.ollama_install_page)
        self.ollama_server_page_id = self.addPage(self.ollama_server_page)
        self.models_page_id = self.addPage(self.models_page)
        self.mlx_whisper_page_id = self.addPage(self.mlx_whisper_page)
        self.location_page_id = self.addPage(self.location_page)
        self.complete_page_id = self.addPage(self.complete_page)

        # Custom button labels
        self.setButtonText(QWizard.WizardButton.NextButton, "Next →")
        self.setButtonText(QWizard.WizardButton.BackButton, "← Back")
        self.setButtonText(QWizard.WizardButton.FinishButton, "🎉 Start Jarvis")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Exit")

        # Store status for sharing between pages
        self.ollama_status: Optional[OllamaStatus] = None
        self.mlx_whisper_status: Optional[MLXWhisperStatus] = None
        self._location_working: Optional[bool] = None

    def is_location_working(self) -> bool:
        """Check if location detection is working (cached)."""
        if self._location_working is None:
            try:
                context = get_location_context(auto_detect=True, resolve_cgnat_public_ip=True)
                self._location_working = context != "Location: Unknown"
            except Exception:
                self._location_working = False
        return self._location_working

    def _apply_theme(self):
        """Apply the shared Jarvis theme."""
        self.setStyleSheet(JARVIS_THEME_STYLESHEET + """
            /* Additional wizard-specific overrides */
            QLabel#title {
                color: #fbbf24;
                font-size: 24px;
                font-weight: bold;
            }
            QLabel#subtitle {
                color: #a1a1aa;
                font-size: 16px;
            }
            QLabel#status-success {
                color: #4ade80;
                font-size: 14px;
            }
            QLabel#status-warning {
                color: #fbbf24;
                font-size: 14px;
            }
            QLabel#status-error {
                color: #f87171;
                font-size: 14px;
            }
            QPushButton#secondary {
                background-color: #1a1d26;
                color: #f4f4f5;
            }
            QPushButton#secondary:hover {
                background-color: #1e222c;
                border-color: #f59e0b;
                color: #fbbf24;
            }
            QPushButton#success {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #22c55e, stop:1 #16a34a);
                color: #0a0b0f;
                border: none;
            }
            QPushButton#success:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4ade80, stop:1 #22c55e);
            }
        """)


class WelcomePage(QWizardPage):
    """Welcome page with status overview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        header_layout = QVBoxLayout()

        title = QLabel("🤖 Welcome to Jarvis")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("Your AI-powered voice assistant")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)

        layout.addLayout(header_layout)
        layout.addSpacing(20)

        # Status card
        self.status_card = QFrame()
        self.status_card.setObjectName("card")
        status_layout = QVBoxLayout(self.status_card)
        status_layout.setContentsMargins(24, 24, 24, 24)
        status_layout.setSpacing(12)

        status_title = QLabel("📋 System Status")
        status_title.setObjectName("section_title")
        status_layout.addWidget(status_title)
        status_layout.addSpacing(8)

        # Status items
        self.cli_status = self._create_status_row("💻 Ollama CLI", "Checking...")
        self.server_status = self._create_status_row("🌐 Ollama Server", "Checking...")
        self.models_status = self._create_status_row("🧠 AI Models", "Checking...")
        self.location_status = self._create_status_row("📍 Location", "Checking...")

        # MLX Whisper status (only shown on Apple Silicon)
        self.mlx_whisper_status = self._create_status_row("🎤 MLX Whisper", "Checking...")
        self._is_apple_silicon = is_apple_silicon()

        status_layout.addWidget(self.cli_status)
        status_layout.addWidget(self.server_status)
        status_layout.addWidget(self.models_status)

        if self._is_apple_silicon:
            status_layout.addWidget(self.mlx_whisper_status)
        else:
            self.mlx_whisper_status.setVisible(False)

        status_layout.addWidget(self.location_status)

        layout.addWidget(self.status_card)

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh Status")
        self.refresh_btn.setObjectName("secondary")
        self.refresh_btn.clicked.connect(self._refresh_status)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

        # Info label
        info = QLabel("Click 'Next' to continue with the setup process.")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #a1a1aa;")
        layout.addWidget(info)

        self.setLayout(layout)

        # Worker for background status check
        self.worker: Optional[StatusCheckWorker] = None

    def _create_status_row(self, label_text: str, status_text: str) -> QWidget:
        """Create a status row widget."""
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 8, 0, 8)

        label = QLabel(label_text)
        label.setStyleSheet("font-size: 14px; background: transparent;")
        layout.addWidget(label)

        layout.addStretch()

        status = QLabel(status_text)
        status.setStyleSheet("font-size: 14px; color: #a1a1aa; background: transparent;")
        status.setObjectName("status_label")
        layout.addWidget(status)

        return row

    def _update_status_row(self, row: QWidget, status_text: str, is_success: bool):
        """Update a status row with new status."""
        status_label = row.findChild(QLabel, "status_label")
        if status_label:
            status_label.setText(status_text)
            if is_success:
                status_label.setStyleSheet("font-size: 14px; color: #4ade80; background: transparent;")
            else:
                status_label.setStyleSheet("font-size: 14px; color: #fbbf24; background: transparent;")

    def initializePage(self):
        """Called when page is shown."""
        self._refresh_status()

    def _refresh_status(self):
        """Refresh Ollama status."""
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("⏳ Checking...")

        # Reset status labels
        for row in [self.cli_status, self.server_status, self.models_status]:
            status_label = row.findChild(QLabel, "status_label")
            if status_label:
                status_label.setText("Checking...")
                status_label.setStyleSheet("font-size: 14px; color: #a1a1aa; background: transparent;")

        # Start background check
        self.worker = StatusCheckWorker()
        self.worker.finished.connect(self._on_status_checked)
        self.worker.start()

    def _on_status_checked(self, status: OllamaStatus):
        """Handle status check completion."""
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("🔄 Refresh Status")

        # Store status in wizard
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            wizard.ollama_status = status

        # Update CLI status
        if status.is_cli_installed:
            self._update_status_row(self.cli_status, f"✅ Installed ({status.cli_path})", True)
        else:
            self._update_status_row(self.cli_status, "❌ Not installed", False)

        # Update server status
        if status.is_server_running:
            self._update_status_row(self.server_status, f"✅ Running (v{status.server_version})", True)
        else:
            self._update_status_row(self.server_status, "❌ Not running", False)

        # Update models status
        if not status.missing_models:
            self._update_status_row(self.models_status, f"✅ All models ready ({len(status.installed_models)} installed)", True)
        else:
            self._update_status_row(self.models_status, f"⚠️ Missing: {', '.join(status.missing_models)}", False)

        # Update location status
        if not is_location_available():
            self._update_status_row(self.location_status, "⚠️ Database not installed", False)
        else:
            location_context = get_location_context(auto_detect=True, resolve_cgnat_public_ip=True)
            if location_context == "Location: Unknown":
                self._update_status_row(self.location_status, "⚠️ Not configured", False)
            else:
                # Extract just the location part after "Location: "
                loc_text = location_context.replace("Location: ", "")
                self._update_status_row(self.location_status, f"✅ {loc_text}", True)

        # Update MLX Whisper status (Apple Silicon only)
        if self._is_apple_silicon:
            mlx_status = check_mlx_whisper_status()
            if isinstance(wizard, SetupWizard):
                wizard.mlx_whisper_status = mlx_status

            if mlx_status.is_fully_setup:
                self._update_status_row(self.mlx_whisper_status, "✅ Ready (GPU acceleration)", True)
            elif not mlx_status.is_ffmpeg_installed:
                self._update_status_row(self.mlx_whisper_status, "⚠️ FFmpeg not installed", False)
            elif not mlx_status.is_mlx_whisper_installed:
                self._update_status_row(self.mlx_whisper_status, "⚠️ Not installed", False)
            else:
                self._update_status_row(self.mlx_whisper_status, "⚠️ Setup incomplete", False)

        # Enable/disable navigation based on status
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Page is always complete - user can proceed."""
        return True

    def nextId(self) -> int:
        """Determine next page based on status."""
        wizard = self.wizard()
        if not isinstance(wizard, SetupWizard) or wizard.ollama_status is None:
            return wizard.ollama_install_page_id

        status = wizard.ollama_status

        # Skip to appropriate page based on what's missing
        if not status.is_cli_installed:
            return wizard.ollama_install_page_id
        elif not status.is_server_running:
            return wizard.ollama_server_page_id
        else:
            # Always show models page so users can change their model selection
            return wizard.models_page_id


class OllamaInstallPage(QWizardPage):
    """Page for installing Ollama CLI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("💻 Install Ollama")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Ollama is required to run local AI models for Jarvis.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Instructions card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        instructions_title = QLabel("📥 Installation Instructions")
        instructions_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(instructions_title)
        card_layout.addSpacing(8)

        if sys.platform == "darwin":
            instructions = QLabel(
                "1. Click the button below to open the Ollama download page\n"
                "2. Download and install Ollama for macOS\n"
                "3. After installation, click 'Verify Installation' to continue"
            )
        elif sys.platform == "win32":
            instructions = QLabel(
                "1. Click the button below to open the Ollama download page\n"
                "2. Download and run the Windows installer\n"
                "3. After installation, click 'Verify Installation' to continue"
            )
        else:
            instructions = QLabel(
                "1. Open a terminal and run: curl -fsSL https://ollama.ai/install.sh | sh\n"
                "2. Or click the button below to open the download page\n"
                "3. After installation, click 'Verify Installation' to continue"
            )

        instructions.setWordWrap(True)
        instructions.setStyleSheet("line-height: 1.8;")
        card_layout.addWidget(instructions)

        layout.addWidget(card)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.download_btn = QPushButton("🌐 Open Download Page")
        self.download_btn.clicked.connect(self._open_download_page)
        btn_layout.addWidget(self.download_btn)

        self.verify_btn = QPushButton("✅ Verify Installation")
        self.verify_btn.setObjectName("success")
        self.verify_btn.clicked.connect(self._verify_installation)
        btn_layout.addWidget(self.verify_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.setLayout(layout)
        self._is_installed = False

    def _open_download_page(self):
        """Open Ollama download page in browser."""
        webbrowser.open("https://ollama.ai/download")
        self.status_label.setText("📝 Download page opened. Please install Ollama and then click 'Verify Installation'.")
        self.status_label.setStyleSheet("color: #a1a1aa;")

    def _verify_installation(self):
        """Verify Ollama installation."""
        self.verify_btn.setEnabled(False)
        self.verify_btn.setText("⏳ Checking...")

        is_installed, path = check_ollama_cli()

        if is_installed:
            self._is_installed = True
            self.status_label.setText(f"✅ Ollama is installed at: {path}")
            self.status_label.setStyleSheet("color: #4ade80;")

            # Update wizard status
            wizard = self.wizard()
            if isinstance(wizard, SetupWizard) and wizard.ollama_status:
                wizard.ollama_status.is_cli_installed = True
                wizard.ollama_status.cli_path = path
        else:
            self._is_installed = False
            self.status_label.setText("❌ Ollama not found. Please install it and try again.")
            self.status_label.setStyleSheet("color: #f87171;")

        self.verify_btn.setEnabled(True)
        self.verify_btn.setText("✅ Verify Installation")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Page is complete when Ollama is installed."""
        return self._is_installed

    def initializePage(self):
        """Check installation status when page is shown."""
        is_installed, path = check_ollama_cli()
        self._is_installed = is_installed

        if is_installed:
            self.status_label.setText(f"✅ Ollama is already installed at: {path}")
            self.status_label.setStyleSheet("color: #4ade80;")
        else:
            self.status_label.setText("")

        self.completeChanged.emit()

    def nextId(self) -> int:
        """Go to server page next."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.ollama_server_page_id
        return super().nextId()


class OllamaServerPage(QWizardPage):
    """Page for starting Ollama server."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("🌐 Start Ollama Server")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("The Ollama server needs to be running for Jarvis to use AI models.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Instructions card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        instructions_title = QLabel("🚀 Starting the Server")
        instructions_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(instructions_title)
        card_layout.addSpacing(8)

        if sys.platform == "darwin":
            instructions = QLabel(
                "The Ollama server should start automatically when you use it.\n\n"
                "If it's not running, you can:\n"
                "• Open the Ollama app from your Applications folder\n"
                "• Or run 'ollama serve' in a terminal\n"
                "• Or click the button below to start it automatically"
            )
        else:
            instructions = QLabel(
                "The Ollama server should start automatically when you use it.\n\n"
                "If it's not running, you can:\n"
                "• Run 'ollama serve' in a terminal\n"
                "• Or click the button below to start it automatically"
            )

        instructions.setWordWrap(True)
        instructions.setStyleSheet("line-height: 1.8;")
        card_layout.addWidget(instructions)

        layout.addWidget(card)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.start_btn = QPushButton("🚀 Start Server")
        self.start_btn.clicked.connect(self._start_server)
        btn_layout.addWidget(self.start_btn)

        self.verify_btn = QPushButton("✅ Verify Server")
        self.verify_btn.setObjectName("success")
        self.verify_btn.clicked.connect(self._verify_server)
        btn_layout.addWidget(self.verify_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.setLayout(layout)
        self._is_running = False

    def _start_server(self):
        """Start the Ollama server."""
        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ Starting...")
        self.status_label.setText("Starting Ollama server...")
        self.status_label.setStyleSheet("color: #a1a1aa;")

        try:
            # Get ollama path
            wizard = self.wizard()
            ollama_path = "ollama"
            if isinstance(wizard, SetupWizard) and wizard.ollama_status and wizard.ollama_status.cli_path:
                ollama_path = wizard.ollama_status.cli_path

            # Note: We intentionally detach the Ollama server process so it keeps
            # running after Jarvis exits. Ollama is a system service that should
            # persist. The serve command is idempotent - it won't spawn duplicates.
            if sys.platform == "darwin":
                # On macOS, try to open the Ollama app first
                try:
                    subprocess.Popen(
                        ["open", "-a", "Ollama"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception:
                    # Fall back to running serve command
                    subprocess.Popen(
                        [ollama_path, "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
            elif sys.platform == "win32":
                # On Windows, hide the console window
                subprocess.Popen(
                    [ollama_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                # On Linux and other platforms, run serve command
                subprocess.Popen(
                    [ollama_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

            # Wait a bit and then verify
            QTimer.singleShot(3000, self._verify_server)

        except Exception as e:
            self.status_label.setText(f"❌ Failed to start server: {str(e)}")
            self.status_label.setStyleSheet("color: #f87171;")
            self.start_btn.setEnabled(True)
            self.start_btn.setText("🚀 Start Server")

    def _verify_server(self):
        """Verify the server is running."""
        self.verify_btn.setEnabled(False)
        self.verify_btn.setText("⏳ Checking...")
        self.start_btn.setEnabled(False)

        is_running, version = check_ollama_server()

        if is_running:
            self._is_running = True
            self.status_label.setText(f"✅ Ollama server is running (version {version})")
            self.status_label.setStyleSheet("color: #4ade80;")

            # Update wizard status
            wizard = self.wizard()
            if isinstance(wizard, SetupWizard) and wizard.ollama_status:
                wizard.ollama_status.is_server_running = True
                wizard.ollama_status.server_version = version
        else:
            self._is_running = False
            self.status_label.setText("❌ Server not responding. Please try starting it again.")
            self.status_label.setStyleSheet("color: #f87171;")

        self.verify_btn.setEnabled(True)
        self.verify_btn.setText("✅ Verify Server")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 Start Server")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Page is complete when server is running."""
        return self._is_running

    def initializePage(self):
        """Check server status when page is shown."""
        is_running, version = check_ollama_server()
        self._is_running = is_running

        if is_running:
            self.status_label.setText(f"✅ Ollama server is already running (version {version})")
            self.status_label.setStyleSheet("color: #4ade80;")
        else:
            self.status_label.setText("")

        self.completeChanged.emit()

    def nextId(self) -> int:
        """Go to models page next."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.models_page_id
        return super().nextId()


class ModelsPage(QWizardPage):
    """Page for installing required AI models."""

    # Use the centralized model configuration from config.py
    MODEL_OPTIONS = SUPPORTED_CHAT_MODELS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("🧠 Install AI Models")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Jarvis needs specific AI models to work. Choose your model and install.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Model selection card
        selection_card = QFrame()
        selection_card.setObjectName("card")
        # Override card padding to prevent layout issues
        selection_card.setStyleSheet(selection_card.styleSheet() + "QFrame#card { padding: 0px; }")
        selection_layout = QVBoxLayout(selection_card)
        selection_layout.setContentsMargins(24, 24, 24, 24)
        selection_layout.setSpacing(16)

        selection_title = QLabel("🎯 Choose Chat Model")
        selection_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        selection_layout.addWidget(selection_title)
        selection_layout.addSpacing(8)

        # Model option buttons
        self._model_buttons: Dict[str, QPushButton] = {}
        self._selected_model: str = DEFAULT_CHAT_MODEL

        for model_id, info in self.MODEL_OPTIONS.items():
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setMinimumHeight(56)
            btn.setMaximumHeight(56)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setText(f"{info['name']}  —  {info['description']} • RAM: {info['ram']}")
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 12px 16px;
                    border: 2px solid #27272a;
                    border-radius: 8px;
                    background: #1a1d26;
                    color: #e4e4e7;
                    font-size: 13px;
                }
                QPushButton:hover {
                    border-color: #f59e0b;
                    background: #1e222c;
                }
                QPushButton:checked {
                    border-color: #f59e0b;
                    background: rgba(245, 158, 11, 0.1);
                }
            """)
            btn.clicked.connect(lambda checked, m=model_id: self._on_model_selected(m))
            self._model_buttons[model_id] = btn
            selection_layout.addWidget(btn)

        layout.addWidget(selection_card)

        # Model list card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        models_title = QLabel("📦 Required Models")
        models_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(models_title)
        card_layout.addSpacing(8)

        self.models_label = QLabel("Loading...")
        self.models_label.setWordWrap(True)
        self.models_label.setStyleSheet("line-height: 1.6;")
        card_layout.addWidget(self.models_label)

        layout.addWidget(card)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setVisible(False)
        self.log_output.setMaximumHeight(150)
        layout.addWidget(self.log_output)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.install_btn = QPushButton("📥 Install Missing Models")
        self.install_btn.clicked.connect(self._install_models)
        btn_layout.addWidget(self.install_btn)

        self.skip_btn = QPushButton("⏭️ Skip")
        self.skip_btn.setObjectName("secondary")
        self.skip_btn.clicked.connect(self._skip_models)
        btn_layout.addWidget(self.skip_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.setLayout(layout)

        self._is_complete = False
        self._missing_models: List[str] = []
        self._current_model_index = 0
        self._worker: Optional[CommandWorker] = None

    def _on_model_selected(self, model_id: str):
        """Handle model selection."""
        self._selected_model = model_id

        # Update button checked states
        for m_id, btn in self._model_buttons.items():
            btn.setChecked(m_id == model_id)

        # Update the models list display
        self._update_models_display()

    def _update_models_display(self):
        """Update the models display based on selected model."""
        wizard = self.wizard()

        # Get config values
        embed_model = "nomic-embed-text"
        intent_judge_model = "llama3.2:3b"
        try:
            cfg = load_settings()
            embed_model = cfg.ollama_embed_model
            intent_judge_model = getattr(cfg, "intent_judge_model", "llama3.2:3b")
        except Exception:
            pass

        # Get installed models
        installed: List[str] = []
        if isinstance(wizard, SetupWizard) and wizard.ollama_status:
            installed = wizard.ollama_status.installed_models

        # Required models: selected chat model + embed model + intent judge model
        # Intent judge (llama3.2:3b) is always required for voice intent classification
        required = [self._selected_model, embed_model]
        if intent_judge_model and intent_judge_model not in required:
            required.append(intent_judge_model)

        # Check which are missing
        def normalize_model(name: str) -> str:
            return name.split(":")[0] if ":" in name and name.endswith(":latest") else name

        installed_normalized = {normalize_model(m) for m in installed}
        self._missing_models = [
            m for m in required
            if normalize_model(m) not in installed_normalized and m not in installed
        ]

        # Update display
        if self._missing_models:
            missing_text = ", ".join(f"❌ {m}" for m in self._missing_models)
            installed_text = ", ".join(f"✅ {m}" for m in installed) if installed else "None"
            model_info = self.MODEL_OPTIONS.get(self._selected_model, {})
            size_info = model_info.get("size", "unknown size")
            self.models_label.setText(
                f"Installed: {installed_text}\n\n"
                f"Missing: {missing_text}\n\n"
                f"⚠️ Download size: {size_info}. Installation may take several minutes."
            )
            self._is_complete = False
            self.install_btn.setVisible(True)
            self.install_btn.setEnabled(True)
            self.skip_btn.setVisible(True)
        else:
            self.models_label.setText(f"✅ All required models are installed: {', '.join(installed)}")
            self._is_complete = True
            self.install_btn.setVisible(False)
            self.skip_btn.setVisible(False)

        self.completeChanged.emit()

    def _save_model_to_config(self):
        """Save the selected chat model to config file."""
        try:
            xdg = os.environ.get("XDG_CONFIG_HOME")
            if xdg:
                config_path = Path(xdg) / "jarvis" / "config.json"
            else:
                config_path = Path.home() / ".config" / "jarvis" / "config.json"

            config_path.parent.mkdir(parents=True, exist_ok=True)

            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {}

            config["ollama_chat_model"] = self._selected_model

            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            return True
        except Exception:
            return False

    def initializePage(self):
        """Initialize page with current model status."""
        # Load the currently configured chat model
        current_chat_model = DEFAULT_CHAT_MODEL
        try:
            cfg = load_settings()
            current_chat_model = cfg.ollama_chat_model
        except Exception:
            pass

        # Pre-select the model if it's one of our options, otherwise default
        if current_chat_model in self.MODEL_OPTIONS:
            self._selected_model = current_chat_model
        else:
            self._selected_model = DEFAULT_CHAT_MODEL

        # Update button states
        for m_id, btn in self._model_buttons.items():
            btn.setChecked(m_id == self._selected_model)

        # Update the models display
        self._update_models_display()

    def _install_models(self):
        """Start installing missing models."""
        # Save the selected model to config first
        if not self._save_model_to_config():
            self.status_label.setText("⚠️ Could not save model selection to config. Continuing with installation...")
            self.status_label.setStyleSheet("color: #fbbf24;")

        if not self._missing_models:
            self._is_complete = True
            self.completeChanged.emit()
            return

        self._current_model_index = 0
        self._install_next_model()

    def _install_next_model(self):
        """Install the next model in the queue."""
        if self._current_model_index >= len(self._missing_models):
            # All models installed
            self._is_complete = True
            self.progress.setVisible(False)
            self.status_label.setText("✅ All models installed successfully!")
            self.status_label.setStyleSheet("color: #4ade80;")
            self.install_btn.setEnabled(False)
            self.skip_btn.setVisible(False)
            self.completeChanged.emit()
            return

        model = self._missing_models[self._current_model_index]

        self.install_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.log_output.setVisible(True)

        self.status_label.setText(f"📥 Installing {model}... ({self._current_model_index + 1}/{len(self._missing_models)})")
        self.status_label.setStyleSheet("color: #a1a1aa;")

        # Get ollama path
        wizard = self.wizard()
        ollama_path = "ollama"
        if isinstance(wizard, SetupWizard) and wizard.ollama_status and wizard.ollama_status.cli_path:
            ollama_path = wizard.ollama_status.cli_path

        self._worker = CommandWorker([ollama_path, "pull", model])
        self._worker.output.connect(self._on_install_output)
        self._worker.finished.connect(self._on_install_finished)
        self._worker.start()

    def _on_install_output(self, text: str):
        """Handle installation output."""
        self.log_output.append(text)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_install_finished(self, success: bool, message: str):
        """Handle installation completion."""
        if success:
            self._current_model_index += 1
            self._install_next_model()
        else:
            self.progress.setVisible(False)
            self.status_label.setText(f"❌ Failed to install model. {message}")
            self.status_label.setStyleSheet("color: #f87171;")
            self.install_btn.setEnabled(True)
            self.skip_btn.setEnabled(True)

    def _skip_models(self):
        """Skip model installation."""
        self._is_complete = True
        self.status_label.setText("⚠️ Skipped model installation. Jarvis may not work correctly without all models.")
        self.status_label.setStyleSheet("color: #fbbf24;")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Page is complete when all models are installed or skipped."""
        return self._is_complete

    def validatePage(self) -> bool:
        """Save model selection when leaving the page."""
        self._save_model_to_config()
        return True

    def nextId(self) -> int:
        """Go to Whisper setup page next."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            # Always show whisper setup page (for model selection on all platforms)
            return wizard.mlx_whisper_page_id
        return super().nextId()


class WhisperSetupPage(QWizardPage):
    """Page for setting up Whisper speech recognition (all platforms)."""

    # Multilingual models - support ~99 languages
    # Sizes from OpenAI: https://github.com/openai/whisper
    # (id, name, file_size, vram_required, description)
    WHISPER_MODEL_OPTIONS = [
        ("tiny", "Tiny", "~75MB", "~1GB VRAM", "Fastest, lower accuracy"),
        ("base", "Base", "~150MB", "~1GB VRAM", "Fast, decent accuracy"),
        ("small", "Small", "~500MB", "~2GB VRAM", "Good balance (Recommended)"),
        ("medium", "Medium", "~1.5GB", "~5GB VRAM", "Better accuracy, slower"),
        ("large-v3-turbo", "Large V3 Turbo", "~1.6GB", "~6GB VRAM", "Best accuracy, 8x faster than large"),
    ]

    # English-only models - optimized for English, slightly better accuracy
    # Note: large/turbo models don't have .en variants
    WHISPER_MODEL_OPTIONS_EN = [
        ("tiny.en", "Tiny", "~75MB", "~1GB VRAM", "Fastest, English optimized"),
        ("base.en", "Base", "~150MB", "~1GB VRAM", "Fast, English optimized"),
        ("small.en", "Small", "~500MB", "~2GB VRAM", "Good balance (Recommended)"),
        ("medium.en", "Medium", "~1.5GB", "~5GB VRAM", "Better accuracy, English optimized"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self._is_apple_silicon = is_apple_silicon()
        self._is_english_only = True  # Default to English-only for better accuracy

        # Main layout with scroll area for overflow
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(30, 20, 30, 20)

        # Header - different text based on platform
        if self._is_apple_silicon:
            title = QLabel("🎤 MLX Whisper Setup")
            subtitle_text = (
                "GPU-accelerated speech recognition. Choose language and model size."
            )
        else:
            title = QLabel("🎤 Whisper Model Selection")
            subtitle_text = "Choose language mode and model size for speech recognition."

        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Language selection card
        lang_card = QFrame()
        lang_card.setObjectName("card")
        lang_layout = QVBoxLayout(lang_card)
        lang_layout.setContentsMargins(16, 12, 16, 12)
        lang_layout.setSpacing(8)

        lang_title = QLabel("🌍 Language Support")
        lang_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fbbf24; background: transparent;")
        lang_layout.addWidget(lang_title)

        # Language toggle buttons
        lang_btn_layout = QHBoxLayout()
        lang_btn_layout.setSpacing(8)

        self._english_btn = QPushButton("🇬🇧 English Only")
        self._english_btn.setCheckable(True)
        self._english_btn.setChecked(True)
        self._english_btn.setFixedHeight(36)
        self._english_btn.clicked.connect(lambda: self._on_language_changed(True))

        self._multilingual_btn = QPushButton("🌐 Multilingual (99 langs)")
        self._multilingual_btn.setCheckable(True)
        self._multilingual_btn.setFixedHeight(36)
        self._multilingual_btn.clicked.connect(lambda: self._on_language_changed(False))

        lang_btn_style = """
            QPushButton {
                text-align: center;
                padding: 6px 12px;
                border: 2px solid #27272a;
                border-radius: 6px;
                background: #1a1d26;
                color: #e4e4e7;
                font-size: 12px;
            }
            QPushButton:hover {
                border-color: #f59e0b;
                background: #1e222c;
            }
            QPushButton:checked {
                border-color: #f59e0b;
                background: rgba(245, 158, 11, 0.15);
                color: #fbbf24;
            }
        """
        self._english_btn.setStyleSheet(lang_btn_style)
        self._multilingual_btn.setStyleSheet(lang_btn_style)

        lang_btn_layout.addWidget(self._english_btn)
        lang_btn_layout.addWidget(self._multilingual_btn)
        lang_layout.addLayout(lang_btn_layout)

        # Language info label
        self._lang_info_label = QLabel()
        self._lang_info_label.setWordWrap(True)
        self._lang_info_label.setStyleSheet("font-size: 10px; color: #71717a; background: transparent;")
        lang_layout.addWidget(self._lang_info_label)

        layout.addWidget(lang_card)

        # Model selection card with slider
        selection_card = QFrame()
        selection_card.setObjectName("card")
        selection_layout = QVBoxLayout(selection_card)
        selection_layout.setContentsMargins(16, 12, 16, 12)
        selection_layout.setSpacing(4)

        selection_title = QLabel("🎯 Choose Model Size")
        selection_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fbbf24; background: transparent;")
        selection_layout.addWidget(selection_title)

        # Container for slider labels (will be rebuilt on language change)
        self._labels_container = QWidget()
        self._labels_container.setStyleSheet("background: transparent;")
        self._labels_layout = QHBoxLayout(self._labels_container)
        self._labels_layout.setContentsMargins(0, 4, 0, 0)
        self._labels_layout.setSpacing(0)
        selection_layout.addWidget(self._labels_container)

        # Slider with proper padding for handle visibility
        slider_container = QWidget()
        slider_container.setStyleSheet("background: transparent;")
        slider_container.setFixedHeight(36)
        slider_inner = QHBoxLayout(slider_container)
        slider_inner.setContentsMargins(0, 0, 0, 0)

        self._model_slider = QSlider(Qt.Orientation.Horizontal)
        self._model_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._model_slider.setTickInterval(1)
        self._model_slider.setStyleSheet("""
            QSlider {
                background: transparent;
                height: 32px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #27272a;
                height: 4px;
                background: #1a1d26;
                border-radius: 2px;
                margin: 0;
            }
            QSlider::handle:horizontal {
                background: #f59e0b;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #fbbf24;
            }
            QSlider::sub-page:horizontal {
                background: rgba(245, 158, 11, 0.4);
                border-radius: 2px;
            }
            QSlider::tick-mark {
                background: #71717a;
            }
        """)
        self._model_slider.valueChanged.connect(self._on_slider_changed)
        slider_inner.addWidget(self._model_slider)
        selection_layout.addWidget(slider_container)

        # Container for size labels (will be rebuilt on language change)
        self._size_container = QWidget()
        self._size_container.setStyleSheet("background: transparent;")
        self._size_layout = QHBoxLayout(self._size_container)
        self._size_layout.setContentsMargins(0, 0, 0, 4)
        self._size_layout.setSpacing(0)
        selection_layout.addWidget(self._size_container)

        # Selected model info
        self._model_info_label = QLabel()
        self._model_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_info_label.setWordWrap(True)
        self._model_info_label.setFixedHeight(32)
        self._model_info_label.setStyleSheet("""
            font-size: 11px;
            color: #e4e4e7;
            padding: 6px 10px;
            background: #1a1d26;
            border-radius: 6px;
        """)
        selection_layout.addWidget(self._model_info_label)

        layout.addWidget(selection_card)

        # Store selected model (default to tiny for fast first-time experience)
        self._selected_whisper_model: str = "tiny.en"

        # Build initial slider UI
        self._rebuild_slider_ui()
        self._update_language_info()

        # MLX-specific installation section (only for Apple Silicon)
        self._mlx_section = QFrame()
        self._mlx_section.setObjectName("card")
        mlx_layout = QVBoxLayout(self._mlx_section)
        mlx_layout.setContentsMargins(16, 12, 16, 12)
        mlx_layout.setSpacing(6)

        status_title = QLabel("📋 Requirements")
        status_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fbbf24; background: transparent;")
        mlx_layout.addWidget(status_title)

        self.ffmpeg_status = self._create_status_row("🎬 FFmpeg", "Checking...")
        self.mlx_status = self._create_status_row("🧠 MLX Whisper", "Checking...")

        mlx_layout.addWidget(self.ffmpeg_status)
        mlx_layout.addWidget(self.mlx_status)

        # Progress bar for installations
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(16)
        mlx_layout.addWidget(self.progress)

        # Log output for installations
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setVisible(False)
        self.log_output.setMaximumHeight(60)
        self.log_output.setStyleSheet("font-size: 10px;")
        mlx_layout.addWidget(self.log_output)

        # Installation buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.install_ffmpeg_btn = QPushButton("🎬 FFmpeg")
        self.install_ffmpeg_btn.setFixedHeight(32)
        self.install_ffmpeg_btn.clicked.connect(self._install_ffmpeg)
        btn_layout.addWidget(self.install_ffmpeg_btn)

        self.install_mlx_btn = QPushButton("🧠 MLX Whisper")
        self.install_mlx_btn.setFixedHeight(32)
        self.install_mlx_btn.clicked.connect(self._install_mlx_whisper)
        btn_layout.addWidget(self.install_mlx_btn)

        btn_layout.addStretch()
        mlx_layout.addLayout(btn_layout)

        layout.addWidget(self._mlx_section)

        # Hide MLX section on non-Apple Silicon
        if not self._is_apple_silicon:
            self._mlx_section.setVisible(False)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 11px; background: transparent;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self._is_complete = True  # Always complete - model selection can always proceed
        self._worker: Optional[CommandWorker] = None

    def _get_current_model_options(self) -> list:
        """Get the model options list based on current language mode."""
        return self.WHISPER_MODEL_OPTIONS_EN if self._is_english_only else self.WHISPER_MODEL_OPTIONS

    def _on_language_changed(self, is_english: bool):
        """Handle language mode change."""
        self._is_english_only = is_english
        self._english_btn.setChecked(is_english)
        self._multilingual_btn.setChecked(not is_english)

        # Update the language info text
        self._update_language_info()

        # Rebuild slider with new model options
        self._rebuild_slider_ui()

    def _update_language_info(self):
        """Update the language info label based on current selection."""
        if self._is_english_only:
            self._lang_info_label.setText(
                "English-only models are optimized for English and may have slightly better accuracy."
            )
        else:
            self._lang_info_label.setText(
                "Multilingual models support 99 languages including: Spanish, French, German, Chinese, "
                "Japanese, Korean, Arabic, Hindi, Portuguese, Russian, and many more."
            )

    def _rebuild_slider_ui(self):
        """Rebuild the slider labels based on current language mode."""
        options = self._get_current_model_options()
        n = len(options)

        # Clear existing labels
        while self._labels_layout.count():
            item = self._labels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.spacerItem():
                pass  # Spacers are automatically cleaned up

        while self._size_layout.count():
            item = self._size_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add labels aligned with slider tick positions
        # Slider ticks are at 0, 1/(n-1), 2/(n-1), ..., 1 of the groove width
        # We achieve this by: label[0], stretch, label[1], stretch, ..., label[n-1]
        # First label left-aligned, last label right-aligned, middle labels centered
        for i, (model_id, name, file_size, vram, desc) in enumerate(options):
            # Model name label
            label = QLabel(name)
            if i == 0:
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            elif i == n - 1:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 11px; color: #e4e4e7; background: transparent;")
            label.setFixedHeight(18)
            self._labels_layout.addWidget(label)

            # Size/VRAM label - single line to save space
            size_label = QLabel(f"{file_size} / {vram}")
            if i == 0:
                size_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            elif i == n - 1:
                size_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            size_label.setStyleSheet("font-size: 9px; color: #71717a; background: transparent;")
            size_label.setFixedHeight(16)
            self._size_layout.addWidget(size_label)

            # Add stretch after each label except the last
            if i < n - 1:
                self._labels_layout.addStretch(1)
                self._size_layout.addStretch(1)

        # Update slider range
        self._model_slider.setMinimum(0)
        self._model_slider.setMaximum(len(options) - 1)

        # Find best matching position for current selection or default to "tiny"
        model_ids = [m[0] for m in options]
        current_base = self._selected_whisper_model.replace(".en", "")

        # Try to find matching model
        if self._is_english_only:
            target = f"{current_base}.en" if not current_base.endswith(".en") else current_base
        else:
            target = current_base.replace(".en", "")

        if target in model_ids:
            slider_pos = model_ids.index(target)
        elif "tiny.en" in model_ids:
            slider_pos = model_ids.index("tiny.en")
        elif "tiny" in model_ids:
            slider_pos = model_ids.index("tiny")
        else:
            slider_pos = 0  # Default to first (smallest) model

        self._model_slider.setValue(slider_pos)
        self._selected_whisper_model = options[slider_pos][0]
        self._update_model_info()

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        options = self._get_current_model_options()
        if 0 <= value < len(options):
            model_id, name, file_size, ram, desc = options[value]
            self._selected_whisper_model = model_id
            self._update_model_info()

    def _update_model_info(self):
        """Update the model info label based on current selection."""
        options = self._get_current_model_options()
        for model_id, name, file_size, ram, desc in options:
            if model_id == self._selected_whisper_model:
                lang_note = "English only" if self._is_english_only else "99 languages"
                self._model_info_label.setText(f"Selected: {name} ({file_size}, {ram}) — {desc} [{lang_note}]")
                break

    def _create_status_row(self, label_text: str, status_text: str) -> QWidget:
        """Create a status row widget."""
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row.setFixedHeight(28)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)

        label = QLabel(label_text)
        label.setStyleSheet("font-size: 12px; background: transparent;")
        row_layout.addWidget(label)

        row_layout.addStretch()

        status = QLabel(status_text)
        status.setStyleSheet("font-size: 12px; color: #a1a1aa; background: transparent;")
        status.setObjectName("status_label")
        row_layout.addWidget(status)

        return row

    def _update_status_row(self, row: QWidget, status_text: str, is_success: bool):
        """Update a status row with new status."""
        status_label = row.findChild(QLabel, "status_label")
        if status_label:
            status_label.setText(status_text)
            if is_success:
                status_label.setStyleSheet("font-size: 12px; color: #4ade80; background: transparent;")
            else:
                status_label.setStyleSheet("font-size: 12px; color: #fbbf24; background: transparent;")

    def _save_whisper_model_to_config(self):
        """Save the selected whisper model to config file."""
        try:
            xdg = os.environ.get("XDG_CONFIG_HOME")
            if xdg:
                config_path = Path(xdg) / "jarvis" / "config.json"
            else:
                config_path = Path.home() / ".config" / "jarvis" / "config.json"

            config_path.parent.mkdir(parents=True, exist_ok=True)

            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {}

            config["whisper_model"] = self._selected_whisper_model

            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            return True
        except Exception:
            return False

    def initializePage(self):
        """Check status when page is shown."""
        # Load the currently configured whisper model
        current_whisper_model = "tiny.en"  # Default to tiny for fast first-time experience
        try:
            cfg = load_settings()
            current_whisper_model = cfg.whisper_model
        except Exception:
            pass

        # Detect language mode from the model name
        self._is_english_only = current_whisper_model.endswith(".en")
        self._english_btn.setChecked(self._is_english_only)
        self._multilingual_btn.setChecked(not self._is_english_only)
        self._update_language_info()

        # Set the selected model and rebuild slider
        self._selected_whisper_model = current_whisper_model
        self._rebuild_slider_ui()

        # Refresh MLX status only on Apple Silicon
        if self._is_apple_silicon:
            self._refresh_mlx_status()

    def _refresh_mlx_status(self):
        """Refresh MLX Whisper installation status (Apple Silicon only)."""
        status = check_mlx_whisper_status()

        # Update wizard status
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            wizard.mlx_whisper_status = status

        # Update FFmpeg status
        if status.is_ffmpeg_installed:
            self._update_status_row(self.ffmpeg_status, f"✅ Installed ({status.ffmpeg_path})", True)
            self.install_ffmpeg_btn.setEnabled(False)
            self.install_ffmpeg_btn.setText("✅ FFmpeg Installed")
        else:
            self._update_status_row(self.ffmpeg_status, "❌ Not installed", False)
            self.install_ffmpeg_btn.setEnabled(True)
            self.install_ffmpeg_btn.setText("🎬 Install FFmpeg")

        # Update MLX Whisper status
        if status.is_mlx_whisper_installed:
            self._update_status_row(self.mlx_status, "✅ Installed", True)
            self.install_mlx_btn.setEnabled(False)
            self.install_mlx_btn.setText("✅ MLX Whisper Installed")
        else:
            self._update_status_row(self.mlx_status, "❌ Not installed", False)
            self.install_mlx_btn.setEnabled(True)
            self.install_mlx_btn.setText("🧠 Install MLX Whisper")

        # Update status message based on setup state
        if status.is_fully_setup:
            self.status_label.setText("✅ MLX Whisper is ready! GPU-accelerated speech recognition enabled.")
            self.status_label.setStyleSheet("color: #4ade80;")
        else:
            if not status.is_ffmpeg_installed:
                self.status_label.setText(
                    "💡 Install FFmpeg for audio processing, or continue to save your model selection."
                )
            elif not status.is_mlx_whisper_installed:
                self.status_label.setText(
                    "💡 Install MLX Whisper for GPU acceleration, or continue to save your model selection."
                )
            self.status_label.setStyleSheet("color: #a1a1aa;")

        self.completeChanged.emit()

    def _install_ffmpeg(self):
        """Install FFmpeg via Homebrew."""
        # Check if Homebrew is installed
        brew_path = shutil.which("brew")
        if not brew_path:
            self.status_label.setText(
                "❌ Homebrew not found. Please install Homebrew first:\n"
                "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            )
            self.status_label.setStyleSheet("color: #f87171;")
            return

        self.install_ffmpeg_btn.setEnabled(False)
        self.install_ffmpeg_btn.setText("⏳ Installing...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log_output.setVisible(True)
        self.log_output.clear()

        self._worker = CommandWorker([brew_path, "install", "ffmpeg"])
        self._worker.output.connect(self._on_output)
        self._worker.finished.connect(self._on_ffmpeg_installed)
        self._worker.start()

    def _install_mlx_whisper(self):
        """Install MLX Whisper via pip."""
        self.install_mlx_btn.setEnabled(False)
        self.install_mlx_btn.setText("⏳ Installing...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log_output.setVisible(True)
        self.log_output.clear()

        # Use the current Python interpreter
        python_path = sys.executable
        self._worker = CommandWorker([python_path, "-m", "pip", "install", "mlx-whisper"])
        self._worker.output.connect(self._on_output)
        self._worker.finished.connect(self._on_mlx_installed)
        self._worker.start()

    def _on_output(self, text: str):
        """Handle command output."""
        self.log_output.append(text)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_ffmpeg_installed(self, success: bool, message: str):
        """Handle FFmpeg installation completion."""
        self.progress.setVisible(False)
        self.install_ffmpeg_btn.setEnabled(True)
        self.install_ffmpeg_btn.setText("🎬 Install FFmpeg")

        if success:
            self._refresh_mlx_status()
        else:
            self.status_label.setText(f"❌ Failed to install FFmpeg: {message}")
            self.status_label.setStyleSheet("color: #f87171;")

    def _on_mlx_installed(self, success: bool, message: str):
        """Handle MLX Whisper installation completion."""
        self.progress.setVisible(False)
        self.install_mlx_btn.setEnabled(True)
        self.install_mlx_btn.setText("🧠 Install MLX Whisper")

        if success:
            self._refresh_mlx_status()
        else:
            self.status_label.setText(f"❌ Failed to install MLX Whisper: {message}")
            self.status_label.setStyleSheet("color: #f87171;")

    def isComplete(self) -> bool:
        """Page is complete when setup is done or skipped."""
        return self._is_complete

    def validatePage(self) -> bool:
        """Save whisper model selection when leaving the page."""
        self._save_whisper_model_to_config()
        return True

    def nextId(self) -> int:
        """Go to next incomplete step, or complete page if all done."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            # Check if location needs setup
            if not wizard.is_location_working():
                return wizard.location_page_id
            # All done
            return wizard.complete_page_id
        return super().nextId()


class LocationPage(QWizardPage):
    """Page for configuring location detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        # Main layout with scroll area
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollArea > QWidget#qt_scrollarea_viewport { background: transparent; }
        """)

        # Content widget inside scroll area
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("📍 Location Configuration")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Location helps Jarvis provide weather, local services, and time-aware responses.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Status card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        status_title = QLabel("🔍 Detection Status")
        status_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(status_title)
        card_layout.addSpacing(8)

        self.status_label = QLabel("Checking location detection...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("line-height: 1.6;")
        card_layout.addWidget(self.status_label)

        layout.addWidget(card)

        # IP configuration section
        config_card = QFrame()
        config_card.setObjectName("card")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(24, 24, 24, 24)
        config_layout.setSpacing(12)

        config_title = QLabel("⚙️ Manual Configuration (Optional)")
        config_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        config_layout.addWidget(config_title)
        config_layout.addSpacing(8)

        config_info = QLabel("If automatic detection fails, you can manually enter your public IP address.")
        config_info.setWordWrap(True)
        config_info.setStyleSheet("color: #a1a1aa;")
        config_layout.addWidget(config_info)

        config_layout.addSpacing(8)

        # IP input row
        ip_layout = QHBoxLayout()
        ip_layout.setSpacing(12)

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Enter your public IP (e.g., 203.0.113.45)")
        self.ip_input.setMinimumHeight(44)
        ip_layout.addWidget(self.ip_input, stretch=1)

        self.test_btn = QPushButton("🧪 Test")
        self.test_btn.clicked.connect(self._test_ip)
        self.test_btn.setMinimumHeight(44)
        ip_layout.addWidget(self.test_btn)

        config_layout.addLayout(ip_layout)

        layout.addWidget(config_card)

        # Test result label
        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        layout.addWidget(self.test_result_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.open_ip_btn = QPushButton("🌐 Find My IP")
        self.open_ip_btn.setObjectName("secondary")
        self.open_ip_btn.setMinimumHeight(44)
        self.open_ip_btn.clicked.connect(self._open_ip_lookup)
        btn_layout.addWidget(self.open_ip_btn)

        self.save_btn = QPushButton("💾 Save IP to Config")
        self.save_btn.setObjectName("success")
        self.save_btn.setMinimumHeight(44)
        self.save_btn.clicked.connect(self._save_ip_to_config)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Save status label
        self.save_status_label = QLabel("")
        self.save_status_label.setWordWrap(True)
        layout.addWidget(self.save_status_label)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self._validated_ip: Optional[str] = None

    def initializePage(self):
        """Check location status when page is shown."""
        self._check_location_status()

    def _check_location_status(self):
        """Check current location detection status."""
        status_parts = []

        if not GEOIP2_AVAILABLE:
            status_parts.append("❌ GeoIP2 library not installed (pip install geoip2)")
        elif not is_location_available():
            db_path = _get_database_path()
            status_parts.append("❌ GeoLite2 database not found")
            status_parts.append(f"   Expected location: {db_path}")
            status_parts.append("")
            status_parts.append("   To set up:")
            status_parts.append("   1. Register at: maxmind.com/en/geolite2/signup")
            status_parts.append("   2. Download GeoLite2-City (MMDB format)")
            status_parts.append(f"   3. Save as: {db_path}")
        else:
            status_parts.append("✅ GeoLite2 database found")
            location_context = get_location_context(auto_detect=True, resolve_cgnat_public_ip=True)

            if location_context == "Location: Unknown":
                status_parts.append("❌ Could not detect public IP address")
                status_parts.append("")
                status_parts.append("   Your network likely uses NAT without UPnP support.")
                status_parts.append("   Enter your public IP below to enable location features.")
            else:
                status_parts.append(f"✅ {location_context}")
                status_parts.append("")
                status_parts.append("   Location is working! You can skip this step.")

        self.status_label.setText("\n".join(status_parts))

    def _open_ip_lookup(self):
        """Open IP lookup website."""
        webbrowser.open("https://whatismyipaddress.com")

    def _test_ip(self):
        """Test the entered IP address."""
        ip = self.ip_input.text().strip()

        if not ip:
            self.test_result_label.setText("❌ Please enter an IP address")
            self.test_result_label.setStyleSheet("color: #f87171;")
            self.save_btn.setEnabled(False)
            self._validated_ip = None
            return

        import re
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(ip_pattern, ip):
            self.test_result_label.setText("❌ Invalid IP format. Use format: 203.0.113.45")
            self.test_result_label.setStyleSheet("color: #f87171;")
            self.save_btn.setEnabled(False)
            self._validated_ip = None
            return

        octets = ip.split('.')
        for octet in octets:
            if int(octet) > 255:
                self.test_result_label.setText("❌ Invalid IP: octets must be 0-255")
                self.test_result_label.setStyleSheet("color: #f87171;")
                self.save_btn.setEnabled(False)
                self._validated_ip = None
                return

        first_octet = int(octets[0])
        second_octet = int(octets[1])
        if (first_octet == 10 or
            (first_octet == 172 and 16 <= second_octet <= 31) or
            (first_octet == 192 and second_octet == 168) or
            first_octet == 127):
            self.test_result_label.setText("⚠️ This appears to be a private IP. Use your public IP instead.")
            self.test_result_label.setStyleSheet("color: #fbbf24;")
            self.save_btn.setEnabled(False)
            self._validated_ip = None
            return

        if not is_location_available():
            self.test_result_label.setText("⚠️ Cannot test: GeoLite2 database not installed")
            self.test_result_label.setStyleSheet("color: #fbbf24;")
            self.save_btn.setEnabled(True)
            self._validated_ip = ip
            return

        location_info = get_location_info(ip_address=ip)

        if "error" in location_info:
            self.test_result_label.setText("⚠️ IP not found in database. It may still work.")
            self.test_result_label.setStyleSheet("color: #fbbf24;")
            self.save_btn.setEnabled(True)
            self._validated_ip = ip
        else:
            city = location_info.get("city", "Unknown")
            country = location_info.get("country", "Unknown")
            self.test_result_label.setText(f"✅ Location: {city}, {country}")
            self.test_result_label.setStyleSheet("color: #4ade80;")
            self.save_btn.setEnabled(True)
            self._validated_ip = ip

    def _save_ip_to_config(self):
        """Save the validated IP to config file."""
        if not self._validated_ip:
            self.save_status_label.setText("❌ Please test an IP address first")
            self.save_status_label.setStyleSheet("color: #f87171;")
            return

        try:
            import json
            from pathlib import Path

            xdg = os.environ.get("XDG_CONFIG_HOME")
            if xdg:
                config_path = Path(xdg) / "jarvis" / "config.json"
            else:
                config_path = Path.home() / ".config" / "jarvis" / "config.json"

            config_path.parent.mkdir(parents=True, exist_ok=True)

            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {}

            config["location_ip_address"] = self._validated_ip

            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            self.save_status_label.setText(f"✅ Saved to {config_path}")
            self.save_status_label.setStyleSheet("color: #4ade80;")
            self._check_location_status()

        except Exception as e:
            self.save_status_label.setText(f"❌ Error saving config: {e}")
            self.save_status_label.setStyleSheet("color: #f87171;")

    def isComplete(self) -> bool:
        """Page is always complete - location is optional."""
        return True

    def nextId(self) -> int:
        """Go to complete page next."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.complete_page_id
        return super().nextId()


class CompletePage(QWizardPage):
    """Final page showing setup is complete."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self.setFinalPage(True)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 60, 40, 40)

        # Big success icon
        success_icon = QLabel("🎉")
        success_icon.setStyleSheet("font-size: 72px;")
        success_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(success_icon)

        # Header
        title = QLabel("Setup Complete!")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Jarvis is ready to use. Click 'Start Jarvis' to launch the voice assistant.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        # Tips card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        tips_title = QLabel("💡 Quick Tips")
        tips_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(tips_title)
        card_layout.addSpacing(8)

        tips = QLabel(
            "• Say 'Jarvis' followed by your question to activate the assistant\n"
            "• Jarvis will appear in your system tray (menu bar on macOS)\n"
            "• Right-click the tray icon to access settings and controls\n"
            "• View logs by clicking '📝 View Logs' in the tray menu"
        )
        tips.setWordWrap(True)
        tips.setStyleSheet("line-height: 1.8;")
        card_layout.addWidget(tips)

        # Memory viewer tip with special styling
        brain_tip = QLabel("🧠  Peek inside Jarvis's brain — open the Memory Viewer to see what he remembers")
        brain_tip.setWordWrap(True)
        brain_tip.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(245, 158, 11, 0.15), stop:1 rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 8px;
            color: #fbbf24;
            font-style: italic;
        """)
        card_layout.addWidget(brain_tip)

        layout.addWidget(card)

        layout.addStretch()

        self.setLayout(layout)

    def initializePage(self):
        """Hide Cancel button on final page - user can use window close if needed."""
        wizard = self.wizard()
        if wizard:
            wizard.button(QWizard.WizardButton.CancelButton).setVisible(False)

    def nextId(self) -> int:
        """No next page."""
        return -1


def run_setup_wizard() -> bool:
    """
    Run the setup wizard.
    Returns True if setup completed successfully, False if cancelled.
    """
    if not _PYQT6_AVAILABLE:
        raise ImportError(
            "PyQt6 is not available. Install it with: pip install PyQt6\n"
            "On Linux, you may also need: apt-get install libegl1"
        )

    # Create app if not exists
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    wizard = SetupWizard()
    result = wizard.exec()

    return result == QWizard.DialogCode.Accepted


if __name__ == "__main__":
    # For testing
    app = QApplication(sys.argv)
    wizard = SetupWizard()
    result = wizard.exec()
    print(f"Wizard result: {result}")
    sys.exit(0)

