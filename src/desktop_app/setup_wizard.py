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
from typing import ClassVar, Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto

import requests

from jarvis.config import SUPPORTED_CHAT_MODELS, DEFAULT_CHAT_MODEL
from jarvis.utils.vram import (
    detect_total_vram_mb,
    get_recommended_model_id,
    required_vram_mb,
)


def check_funasr_installed() -> bool:
    """Check if the FunASR runtime (needed by SenseVoice) is importable."""
    try:
        from jarvis.listening.sensevoice import is_available
        return is_available()
    except Exception:
        return False


# Import config early (no PyQt6 dependency) - needed for detection functions
from jarvis.config import load_settings, get_default_config, default_config_path


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

    from jarvis.llm import check_version
    return check_version(base_url, timeout=5.0)


def get_required_models() -> List[str]:
    """Get the Ollama models that must be present locally, given the active
    providers.

    Only models that actually run on Ollama are required:
    - Chat model + intent-judge model — when the chat provider is Ollama
      (both run through the chat backend). Skipped for an OpenAI-compatible
      chat provider, where those are remote model names, not Ollama pulls.
    - Embedding model — when the effective embedding provider is Ollama
      (covers the advanced split where chat is remote but embeddings are
      local). Skipped when embeddings are remote.

    A pure OpenAI-compatible setup therefore requires nothing locally.
    """
    try:
        cfg = load_settings()
        llm_provider = getattr(cfg, "llm_provider", "ollama") or "ollama"
        embed_provider = getattr(cfg, "embedding_provider", "") or llm_provider
        models = []

        # Chat model runs on the chat provider's backend.
        if llm_provider != "openai_compatible":
            if cfg.ollama_chat_model:
                models.append(cfg.ollama_chat_model)

        # Embedding model runs on the embedding provider's backend.
        if embed_provider != "openai_compatible":
            if cfg.ollama_embed_model and cfg.ollama_embed_model not in models:
                models.append(cfg.ollama_embed_model)

        # The fast model powers voice intent classification and the other
        # real-time passes, but is only an Ollama pull when the chat
        # provider is Ollama (config load resolves it per provider).
        if llm_provider != "openai_compatible":
            fast_model = getattr(cfg, "fast_model", "gemma4:e2b")
            if fast_model and fast_model not in models:
                models.append(fast_model)

        return models
    except Exception:
        # Default models if config can't be loaded
        # Note: DEFAULT_CHAT_MODEL is gemma4:e2b which is also the intent judge model,
        # so the default list is effectively just 2 unique models
        defaults = [DEFAULT_CHAT_MODEL, "nomic-embed-text"]
        if "gemma4:e2b" not in defaults:
            defaults.append("gemma4:e2b")
        return defaults


def resolve_ollama_path() -> str:
    """Resolve the ollama CLI path for subprocess invocation.

    PATH first, then platform-specific install locations via check_ollama_cli,
    then a literal "ollama" as last resort. Frozen .app launches on macOS get
    a sanitised PATH that excludes /usr/local/bin and /opt/homebrew/bin, so
    shutil.which alone is not enough.
    """
    path = shutil.which("ollama")
    if path:
        return path
    _, resolved = check_ollama_cli()
    return resolved or "ollama"


def check_installed_models(ollama_path: Optional[str] = None) -> List[str]:
    """
    Get list of installed Ollama models.
    Returns list of model names.
    """
    if ollama_path is None:
        ollama_path = resolve_ollama_path()

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
            return name[:-len(":latest")] if name.endswith(":latest") else name

        installed_normalized = {normalize_model(m) for m in installed}

        status.installed_models = installed
        status.missing_models = [
            m for m in required
            if normalize_model(m) not in installed_normalized and m not in installed
        ]
    else:
        status.missing_models = get_required_models()

    return status


def should_show_setup_wizard(force_server_check: bool = False) -> bool:
    """
    Check if the setup wizard should be shown.

    Returns True only if user intervention is needed:
    - CLI not installed (user must install Ollama)
    - Models missing (user must download models)
    - Server unreachable after auto-start already failed (force_server_check)

    Does NOT return True just because server isn't running,
    since the app can auto-start the server if CLI is installed.
    Pass ``force_server_check=True`` after auto-start has already been
    attempted and failed to re-evaluate the unreachable-server case.
    """
    # An OpenAI-compatible user has opted out of the local Ollama stack,
    # so the Ollama-centric prerequisites don't apply — never auto-show.
    # (The wizard can still be opened manually from the tray to switch back.)
    try:
        cfg = load_settings()
        if getattr(cfg, "llm_provider", "ollama") == "openai_compatible":
            return False
    except Exception:
        pass

    status = check_ollama_status()

    # If CLI not installed, user needs to install Ollama
    if not status.is_cli_installed:
        return True

    # If server is running and models are missing, user needs to download them
    if status.is_server_running and len(status.missing_models) > 0:
        return True

    # If auto-start already failed and server is still unreachable,
    # the user needs to intervene to diagnose the problem.
    if force_server_check and not status.is_server_running:
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
        QSizePolicy, QScrollArea, QLineEdit, QSlider, QComboBox, QCheckBox,
        QRadioButton, QButtonGroup, QStackedWidget
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QPainter

    from desktop_app.themes import JARVIS_THEME_STYLESHEET, COLORS, _ensure_icons, _ICON_STYLESHEET_TEMPLATE
    from desktop_app.mcp_catalogue import get_wizard_entries, MCPEntry

    # Import location utilities with crash protection for Windows native modules
    try:
        from jarvis.utils.location import (
            get_location_info,
            get_location_context,
            is_location_available,
            _get_database_path,
            _is_private_ip,
            _is_cgnat_ip,
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
        _is_private_ip = lambda ip: True
        _is_cgnat_ip = lambda ip: False
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
    _is_private_ip = lambda ip: True
    _is_cgnat_ip = lambda ip: False
    GEOIP2_AVAILABLE = False


class _KeepAliveWorker(QThread):
    """QThread that keeps itself referenced until its OS thread has fully
    finished.

    Wizard pages rebind their worker attribute inside completion slots
    (model install chains, refresh buttons, test-connection buttons). The
    completion signal is emitted at the end of run(), so the slot can run
    while the OS thread is still winding down; dropping the last Python
    reference at that point destroys a running QThread and Qt aborts the
    whole app ("Fatal Python error: Aborted" — #509, #407, #239).

    Subclasses must NOT shadow the built-in ``finished`` signal — the
    keep-alive registry relies on it to know when release is safe.
    """

    _active: ClassVar[set] = set()

    def start(self, *args, **kwargs):
        _KeepAliveWorker._active.add(self)
        self.finished.connect(self._retire)
        super().start(*args, **kwargs)

    def _retire(self) -> None:
        _KeepAliveWorker._active.discard(self)


class StatusCheckWorker(_KeepAliveWorker):
    """Worker thread for checking Ollama status."""
    status_ready = pyqtSignal(OllamaStatus)

    def run(self):
        status = check_ollama_status()
        self.status_ready.emit(status)


class CommandWorker(_KeepAliveWorker):
    """Worker thread for running commands."""
    output = pyqtSignal(str)
    completed = pyqtSignal(bool, str)

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
                self.completed.emit(True, "✅ Command completed successfully")
            else:
                self.completed.emit(False, f"❌ Command failed with exit code {process.returncode}")
        except Exception as e:
            self.completed.emit(False, f"❌ Error: {str(e)}")


class SetupWizard(QWizard):
    """Main setup wizard window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Jarvis Setup Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(700, 875)

        # Apply dark theme
        self._apply_theme()

        # Add pages and store their IDs
        self.welcome_page = WelcomePage(self)
        self.provider_choice_page = ProviderChoicePage(self)
        self.openai_compat_page = OpenAICompatiblePage(self)
        self.ollama_install_page = OllamaInstallPage(self)
        self.ollama_server_page = OllamaServerPage(self)
        self.models_page = ModelsPage(self)
        self.sensevoice_page = SenseVoiceSetupPage(self)
        self.dictation_page = DictationPage(self)
        self.mcp_page = MCPPage(self)
        self.search_providers_page = SearchProvidersPage(self)
        self.location_page = LocationPage(self)
        self.complete_page = CompletePage(self)

        self.welcome_page_id = self.addPage(self.welcome_page)
        self.sensevoice_page_id = self.addPage(self.sensevoice_page)
        self.provider_choice_page_id = self.addPage(self.provider_choice_page)
        self.openai_compat_page_id = self.addPage(self.openai_compat_page)
        self.ollama_install_page_id = self.addPage(self.ollama_install_page)
        self.ollama_server_page_id = self.addPage(self.ollama_server_page)
        self.models_page_id = self.addPage(self.models_page)
        self.dictation_page_id = self.addPage(self.dictation_page)
        self.mcp_page_id = self.addPage(self.mcp_page)
        self.search_providers_page_id = self.addPage(self.search_providers_page)
        self.location_page_id = self.addPage(self.location_page)
        self.complete_page_id = self.addPage(self.complete_page)

        # The provider choice is the first step: Ollama is optional now, so
        # the wizard must ask which runtime the user wants before running any
        # Ollama-specific checks. The Welcome/status page and the Ollama
        # install/server/models pages are only reached on the Ollama branch.
        self.setStartId(self.sensevoice_page_id)

        # Custom button labels
        self.setButtonText(QWizard.WizardButton.NextButton, "Next →")
        self.setButtonText(QWizard.WizardButton.BackButton, "← Back")
        self.setButtonText(QWizard.WizardButton.FinishButton, "🎉 Start Jarvis")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Exit")

        # Store status for sharing between pages
        self.ollama_status: Optional[OllamaStatus] = None
        self.sensevoice_status: Optional[bool] = None
        self._location_working: Optional[bool] = None

    def ollama_entry_page_id(self) -> int:
        """First Ollama-flow page to show, based on detection status:
        install (CLI missing) → server (not running) → models. Shared by the
        provider-choice page so the Ollama branch lands on the right step."""
        status = self.ollama_status
        if status is None or not status.is_cli_installed:
            return self.ollama_install_page_id
        if not status.is_server_running:
            return self.ollama_server_page_id
        return self.models_page_id

    def is_location_working(self) -> bool:
        """Check if location detection is working (cached)."""
        if self._location_working is None:
            try:
                cfg = load_settings()
                # If location is disabled, treat as "working" so we skip the page
                if not getattr(cfg, 'location_enabled', True):
                    self._location_working = True
                else:
                    context = get_location_context(
                        config_ip=cfg.location_ip_address,
                        auto_detect=cfg.location_auto_detect,
                        resolve_cgnat_public_ip=cfg.location_cgnat_resolve_public_ip,
                    )
                    self._location_working = context != "Location: Unknown"
            except Exception:
                self._location_working = False
        return self._location_working

    def _apply_theme(self):
        """Apply the shared Jarvis theme with SVG indicator icons."""
        icons = _ensure_icons()
        icon_css = _ICON_STYLESHEET_TEMPLATE.format(**icons)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET + icon_css + """
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

        # SenseVoice (FunASR) status — cross-platform
        self.sensevoice_status = self._create_status_row("🎤 SenseVoice (FunASR)", "Checking...")

        status_layout.addWidget(self.cli_status)
        status_layout.addWidget(self.server_status)
        status_layout.addWidget(self.models_status)
        status_layout.addWidget(self.sensevoice_status)

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
        self.worker.status_ready.connect(self._on_status_checked)
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
            try:
                cfg = load_settings()
                location_context = get_location_context(
                    config_ip=cfg.location_ip_address,
                    auto_detect=cfg.location_auto_detect,
                    resolve_cgnat_public_ip=cfg.location_cgnat_resolve_public_ip,
                )
            except Exception:
                location_context = get_location_context(auto_detect=True, resolve_cgnat_public_ip=True)
            if location_context == "Location: Unknown":
                self._update_status_row(self.location_status, "⚠️ Not configured", False)
            else:
                # Extract just the location part after "Location: "
                loc_text = location_context.replace("Location: ", "")
                self._update_status_row(self.location_status, f"✅ {loc_text}", True)

        # Update SenseVoice (FunASR) status
        sensevoice_ok = check_funasr_installed()
        if isinstance(wizard, SetupWizard):
            wizard.sensevoice_status = sensevoice_ok

        if sensevoice_ok:
            self._update_status_row(self.sensevoice_status, "✅ Installed", True)
        else:
            self._update_status_row(self.sensevoice_status, "❌ Not installed", False)

        # Enable/disable navigation based on status
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Page is always complete - user can proceed."""
        return True

    def nextId(self) -> int:
        """The Welcome/status page is reached only on the Ollama branch (after
        the provider choice), so it leads into the Ollama install/server/models
        flow based on the detected status."""
        wizard = self.wizard()
        if not isinstance(wizard, SetupWizard):
            return super().nextId()
        return wizard.ollama_entry_page_id()


class ProviderChoicePage(QWizardPage):
    """Choose which local runtime serves the LLM: Ollama (the bundled
    default) or an OpenAI-compatible server (LM Studio, oMLX, llama.cpp's
    ``llama-server``, vLLM, LocalAI). The choice branches the rest of the
    wizard — Ollama continues to the install/server/models flow, while
    OpenAI-compatible jumps to a connection-config page and skips the
    Ollama-specific pages entirely."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self._selected = "ollama"

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("🔌 Choose Your LLM Provider")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Welcome to Jarvis. Choose how it runs its language model. Both "
            "options keep everything on machines you control, never a "
            "third-party cloud."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # A QButtonGroup makes the radios mutually exclusive even though each
        # lives in its own card (Qt's auto-exclusivity only applies to radios
        # sharing a direct parent, which these do not).
        self._button_group = QButtonGroup(self)

        self._ollama_radio = QRadioButton("  🦙  Ollama (recommended)")
        self._ollama_radio.setChecked(True)
        self._button_group.addButton(self._ollama_radio)
        ollama_card = self._provider_card(
            self._ollama_radio,
            "Runs open models locally on this machine. The wizard installs "
            "Ollama and downloads the models for you. Best if you have no "
            "model server already.",
        )
        layout.addWidget(ollama_card)

        self._openai_radio = QRadioButton("  🔗  OpenAI-compatible server")
        self._button_group.addButton(self._openai_radio)
        openai_card = self._provider_card(
            self._openai_radio,
            "Point Jarvis at a server that speaks the OpenAI API. This is "
            "usually another local app (LM Studio, oMLX, llama.cpp, vLLM, "
            "LocalAI) running on your own machine or network. You provide its "
            "URL and model name on the next step.",
        )
        layout.addWidget(openai_card)

        self._ollama_radio.toggled.connect(self._on_toggle)
        self._openai_radio.toggled.connect(self._on_toggle)

        layout.addStretch()
        self.setLayout(layout)

        self._preselect_from_config()

    def _provider_card(self, radio, desc_text):
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)
        radio.setStyleSheet("font-size: 15px; font-weight: bold;")
        card_layout.addWidget(radio)
        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        card_layout.addWidget(desc)
        return card

    def _preselect_from_config(self):
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path()) or {}
            provider = str(config.get("llm_provider", "ollama") or "ollama")
        except Exception:
            provider = "ollama"
        if provider == "openai_compatible":
            self._openai_radio.setChecked(True)
            self._selected = "openai_compatible"
        else:
            self._ollama_radio.setChecked(True)
            self._selected = "ollama"

    def _on_toggle(self):
        self._selected = (
            "openai_compatible" if self._openai_radio.isChecked() else "ollama"
        )

    def validatePage(self) -> bool:
        """Persist the provider choice. Selecting Ollama clears any
        OpenAI-compatible overrides so the Ollama settings become
        authoritative again — no stale base URL / key / model is left
        pointing at a former remote server."""
        try:
            from jarvis.config import default_config_path, _load_json, _save_json
            config_path = default_config_path()
            config = _load_json(config_path) or {}

            if self._selected == "openai_compatible":
                config["llm_provider"] = "openai_compatible"
            else:
                # Ollama is the default; omit the key and drop the
                # OpenAI-compatible connection overrides.
                config.pop("llm_provider", None)
                for stale in (
                    "llm_base_url", "llm_api_key", "llm_chat_model",
                    "embedding_provider", "embedding_base_url",
                    "embedding_api_key", "embedding_model",
                ):
                    config.pop(stale, None)

            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def isComplete(self) -> bool:
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        if not isinstance(wizard, SetupWizard):
            return super().nextId()
        if self._selected == "openai_compatible":
            return wizard.openai_compat_page_id
        return wizard.welcome_page_id


class _ModelFetchWorker(_KeepAliveWorker):
    """Fetches the model list from an OpenAI-compatible server off the UI
    thread so the wizard never freezes while connecting."""

    done = pyqtSignal(bool, list)  # (reached, model_ids)

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key

    def run(self):
        models = OpenAICompatiblePage._fetch_models(self._base_url, self._api_key)
        self.done.emit(bool(models), models)


class _DiscoveryWorker(QThread):
    """Probes well-known local ports for a running OpenAI-compatible server so
    the wizard can offer a one-click pick instead of asking for a URL."""

    done = pyqtSignal(list)  # list of (label, url)

    def __init__(self, candidates: list):
        super().__init__()
        self._candidates = candidates

    def run(self):
        self.done.emit(OpenAICompatiblePage._discover_servers(self._candidates))


class _CapabilityWorker(QThread):
    """Probes what the chosen server+model can actually do (chat, tools,
    embeddings) off the UI thread, so the wizard catches a dud model or a
    missing embeddings endpoint before setup finishes rather than at runtime."""

    done = pyqtSignal(object)  # ServerCapabilities

    def __init__(self, base_url: str, api_key: str, chat_model: str, embed_model: str):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._chat_model = chat_model
        self._embed_model = embed_model

    def run(self):
        try:
            from jarvis.llm import OpenAICompatibleBackend, ServerCapabilities
            backend = OpenAICompatibleBackend(self._base_url, api_key=self._api_key or None)
            caps = backend.check_capabilities(self._chat_model, self._embed_model or None)
        except Exception:
            from jarvis.llm import ServerCapabilities
            caps = ServerCapabilities()
        self.done.emit(caps)


class OpenAICompatiblePage(QWizardPage):
    """Collect the OpenAI-compatible server's connection details. Shown only
    on the OpenAI-compatible branch; it writes the ``llm_*`` /
    ``embedding_model`` config keys and then skips straight to speech-recognition setup.

    Guided rather than freeform: the page auto-discovers running local
    servers, offers a one-click app preset, and (after Connect) fetches the
    server's actual model list into editable dropdowns with sensible defaults.
    A single Connect then probes the chosen model so the user learns up front
    whether chat, tool calling, and embeddings work, and is offered the
    Ollama-embeddings fallback when the server can't embed. Power users can
    still type any base URL or model id by hand.
    """

    _DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio default

    # Well-known local OpenAI-compatible servers, used both for the app preset
    # picker and for auto-discovery. All loopback, so probing never leaves the
    # machine.
    _KNOWN_SERVERS = [
        ("LM Studio", "http://localhost:1234/v1"),
        ("Ollama (OpenAI API)", "http://localhost:11434/v1"),
        ("Jan", "http://localhost:1337/v1"),
        ("llama.cpp / LocalAI", "http://localhost:8080/v1"),
        ("vLLM", "http://localhost:8000/v1"),
        ("oMLX (ol.mlx)", "http://localhost:9876/v1"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self._fetch_worker = None
        self._discovery_worker = None
        self._cap_worker = None

        layout = QVBoxLayout()
        layout.setSpacing(14)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("🔗 OpenAI-compatible Server")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Point Jarvis at a local server (LM Studio, Ollama, Jan, llama.cpp, "
            "vLLM, …). Pick your app or let Jarvis find it, then Connect to load "
            "its models. Only the base URL and chat model are required."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        form_card = QFrame()
        form_card.setObjectName("card")
        form = QVBoxLayout(form_card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        # App preset: prefills the base URL for a known server so the user
        # never has to remember a port.
        preset_label = QLabel("Your app")
        preset_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        form.addWidget(preset_label)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Select your app (optional)…")
        for label, _url in self._KNOWN_SERVERS:
            self._preset_combo.addItem(label)
        self._preset_combo.addItem("Other / custom")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addWidget(self._preset_combo)

        self._base_url_input = self._labelled_edit(
            form, "Base URL",
            "e.g. http://localhost:1234/v1 (LM Studio default)")
        self._api_key_input = self._labelled_edit(
            form, "API key (optional)", "leave empty if your server needs none",
            password=True)

        # Connect button + status: fetch the model list, then probe the model.
        self._connect_btn = QPushButton("🔌 Connect & load models")
        self._connect_btn.setObjectName("secondary")
        self._connect_btn.clicked.connect(self._on_connect)
        form.addWidget(self._connect_btn)
        self._connect_status = QLabel("")
        self._connect_status.setWordWrap(True)
        self._connect_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']};")
        form.addWidget(self._connect_status)

        self._chat_model_combo = self._labelled_combo(
            form, "Chat model", "pick after connecting, or type the model id")
        self._embed_model_combo = self._labelled_combo(
            form, "Embedding model (optional)",
            "leave empty to skip embeddings (memory uses keyword search)")

        # Fast model link toggle + selector
        self._openai_linked = False
        self._openai_link_cb = QCheckBox(
            "\u2699\ufe0f Use same model for fast tasks (voice, routing)")
        self._openai_link_cb.setChecked(False)
        self._openai_link_cb.setStyleSheet("font-size: 13px; color: #e4e4e7; padding: 4px 0;")
        self._openai_link_cb.toggled.connect(self._on_openai_link_toggled)
        form.addWidget(self._openai_link_cb)

        # Fast model selector: label + combo stored for visibility toggling
        self._fast_label = QLabel("Fast model (voice intent, tool routing)")
        self._fast_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        form.addWidget(self._fast_label)
        self._fast_model_combo = QComboBox()
        self._fast_model_combo.setEditable(True)
        self._fast_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fast_model_combo.lineEdit().setPlaceholderText(
            "leave empty to use the chat model")
        self._fast_model_combo.currentTextChanged.connect(
            lambda *_: self.completeChanged.emit())
        form.addWidget(self._fast_model_combo)
        self._fast_label.setVisible(True)
        self._fast_model_combo.setVisible(True)

        # Shown only when the probe finds the server can't embed: a one-click
        # way to keep full semantic memory by routing embeddings to Ollama.
        self._use_ollama_embed = QCheckBox(
            "Use Ollama for embeddings instead (keeps full semantic memory)")
        self._use_ollama_embed.setVisible(False)
        self._use_ollama_embed.toggled.connect(lambda *_: self.completeChanged.emit())
        form.addWidget(self._use_ollama_embed)

        layout.addWidget(form_card)

        tip = QLabel(
            "💡  Memory search uses embeddings. If your server has no "
            "embeddings endpoint, leave the embedding model empty and Jarvis "
            "falls back to keyword search."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "background: rgba(245, 158, 11, 0.10);"
            "border: 1px solid rgba(245, 158, 11, 0.25);"
            "border-radius: 8px; padding: 12px 16px; color: #fbbf24; font-size: 13px;"
        )
        layout.addWidget(tip)

        layout.addStretch()
        self.setLayout(layout)

    def _labelled_edit(self, form, label_text, placeholder, password=False):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 13px; font-weight: bold;")
        form.addWidget(label)
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        if password:
            field.setEchoMode(QLineEdit.EchoMode.Password)
        # Re-evaluate Next: the base URL is half of isComplete, so editing it
        # must refresh the button (the chat-model combo does the same).
        field.textChanged.connect(lambda *_: self.completeChanged.emit())
        form.addWidget(field)
        return field

    def _labelled_combo(self, form, label_text, placeholder):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 13px; font-weight: bold;")
        form.addWidget(label)
        combo = QComboBox()
        combo.setEditable(True)  # power users can type a model the listing omits
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.lineEdit().setPlaceholderText(placeholder)
        combo.currentTextChanged.connect(lambda *_: self.completeChanged.emit())
        form.addWidget(combo)
        return combo

    @staticmethod
    def _fetch_models(base_url: str, api_key: str, timeout: float = 6.0) -> list:
        """Return the model ids the server advertises at ``/v1/models``, or
        an empty list if it is unreachable. Fail-soft: never raises, so the
        user can still type a model id by hand."""
        base_url = (base_url or "").strip()
        if not base_url:
            return []
        try:
            from jarvis.llm import OpenAICompatibleBackend
            backend = OpenAICompatibleBackend(base_url, api_key=(api_key or "").strip() or None)
            return list(backend.list_models(timeout_sec=timeout))
        except Exception:
            return []

    @staticmethod
    def _discover_servers(candidates: list, timeout: float = 1.5) -> list:
        """Probe well-known local ports for a running OpenAI-compatible server.
        Only loopback addresses are probed, so discovery never touches the
        network. Returns the reachable ``(label, url)`` pairs."""
        found = []
        for label, url in candidates:
            if OpenAICompatiblePage._fetch_models(url, "", timeout=timeout):
                found.append((label, url))
        return found

    @staticmethod
    def _classify_models(models: list) -> tuple:
        """Split advertised ids into ``(chat, embed)`` by an id heuristic.
        Model ids are vendor tokens rather than natural language, so matching
        ``embed`` in the id stays language-agnostic."""
        embed = [m for m in models if "embed" in m.lower()]
        chat = [m for m in models if "embed" not in m.lower()]
        return chat, embed

    @staticmethod
    def _preferred_fast_default(models: list) -> str:
        """Return ``gemma4:e2b`` when it appears in the model list, otherwise
        empty string (user picks or types)."""
        for m in models:
            if m == "gemma4:e2b":
                return m
        return ""

    def _on_preset_changed(self, idx: int):
        # idx 0 is the placeholder and the last item is "Other / custom"; the
        # ones in between map to _KNOWN_SERVERS and prefill the base URL.
        if 1 <= idx <= len(self._KNOWN_SERVERS):
            _label, url = self._KNOWN_SERVERS[idx - 1]
            self._base_url_input.setText(url)

    def _on_openai_link_toggled(self, linked: bool):
        """Show/hide the fast model selector when toggling the link checkbox."""
        self._openai_linked = linked
        self._fast_label.setVisible(not linked)
        self._fast_model_combo.setVisible(not linked)
        if linked:
            self._fast_model_combo.setCurrentText("")
        # Let the wizard recalculate its size from the current page's content
        wizard = self.wizard()
        if wizard:
            wizard.adjustSize()
        self.completeChanged.emit()

    def _on_connect(self):
        base_url = (self._base_url_input.text() or "").strip()
        if not base_url:
            self._connect_status.setText("⚠️ Enter a base URL first.")
            return
        self._connect_btn.setEnabled(False)
        self._connect_status.setText("⏳ Connecting…")
        worker = _ModelFetchWorker(base_url, (self._api_key_input.text() or "").strip())
        worker.done.connect(self._on_models_fetched)
        self._fetch_worker = worker  # keep a reference so it isn't GC'd
        worker.start()

    def _on_models_fetched(self, reached: bool, models: list):
        self._populate_models(models)
        if not (reached and models):
            self._connect_btn.setEnabled(True)
            self._connect_status.setText(
                "⚠️ Couldn't load models. Check the URL/key and that the server "
                "is running, or type the model id manually below.")
            self.completeChanged.emit()
            return
        # Models loaded and a sensible chat default is selected — probe the
        # model so the user learns up front what works.
        chat = (self._chat_model_combo.currentText() or "").strip()
        base = (self._base_url_input.text() or "").strip()
        if base and chat:
            self._connect_status.setText(
                f"✅ Connected — {len(models)} model(s). Checking {chat}…")
            self._start_capability_probe()
        else:
            self._connect_btn.setEnabled(True)
            self._connect_status.setText(f"✅ Connected — {len(models)} model(s) found.")
        self.completeChanged.emit()

    def _start_capability_probe(self):
        base = (self._base_url_input.text() or "").strip()
        chat = (self._chat_model_combo.currentText() or "").strip()
        if not (base and chat):
            self._connect_btn.setEnabled(True)
            return
        self._connect_btn.setEnabled(False)
        worker = _CapabilityWorker(
            base, (self._api_key_input.text() or "").strip(),
            chat, (self._embed_model_combo.currentText() or "").strip())
        worker.done.connect(self._on_capabilities)
        self._cap_worker = worker  # keep a reference so it isn't GC'd
        worker.start()

    def _on_capabilities(self, caps):
        self._connect_btn.setEnabled(True)
        self._connect_status.setText(self._capability_summary(caps))
        # Offer the Ollama-embeddings split only when the server clearly works
        # for chat but cannot embed.
        needs_split = bool(getattr(caps, "reachable", False)
                           and getattr(caps, "chat", False)
                           and not getattr(caps, "embeddings", False))
        self._use_ollama_embed.setVisible(needs_split)
        if not needs_split:
            self._use_ollama_embed.setChecked(False)
        self.completeChanged.emit()

    @staticmethod
    def _capability_summary(caps) -> str:
        """Honest one-line verdict on what the chosen server+model can do."""
        if not getattr(caps, "reachable", False):
            return ("⚠️ Couldn't get a response with that model. Check the URL, "
                    "key, and that the model id is loaded on the server.")
        mark = lambda ok: "✅" if ok else "⚠️"
        parts = [f"{mark(caps.chat)} Chat", f"{mark(caps.tools)} Tool calling"]
        parts.append("✅ Embeddings" if caps.embeddings
                     else "⚠️ No embeddings (memory uses keyword search)")
        return "   ".join(parts)

    def _populate_models(self, models: list):
        """Fill the dropdowns with fetched model ids. Embedding-named ids go to
        the embedding box and the rest to chat; if the heuristic finds none of
        a kind, both boxes get the full list. A value the user already
        typed/selected is preserved, otherwise a sensible default is applied so
        the common case is just Connect then Next."""
        chat_models, embed_models = self._classify_models(models)
        # The chat box lists chat models (or the full list if the heuristic
        # found none), but only auto-selects a real chat model — never an
        # embedding model, which would be a wrong default.
        self._fill_combo(self._chat_model_combo, chat_models or models, blank=False,
                         default=(chat_models[0] if chat_models else ""))
        self._fill_combo(self._embed_model_combo, embed_models or models, blank=True,
                         default=(embed_models[0] if embed_models else ""))
        self._fill_combo(self._fast_model_combo, models, blank=True,
                         default=self._preferred_fast_default(models))

    def _fill_combo(self, combo, items, *, blank: bool, default: str):
        current = (combo.currentText() or "").strip()
        combo.blockSignals(True)
        combo.clear()
        if blank:
            combo.addItem("")  # "(none)" — embeddings optional
        for it in items:
            combo.addItem(it)
        combo.setCurrentText(current or default)
        combo.blockSignals(False)

    def initializePage(self):
        """Pre-fill from any existing config so re-running the wizard keeps the
        user's values. With no saved URL, default to the common LM Studio
        address and kick off auto-discovery of running local servers."""
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path()) or {}
        except Exception:
            config = {}
        saved_url = str(config.get("llm_base_url", "") or "")
        self._base_url_input.setText(saved_url or self._DEFAULT_BASE_URL)
        self._api_key_input.setText(str(config.get("llm_api_key", "") or ""))
        self._chat_model_combo.setCurrentText(str(config.get("llm_chat_model", "") or ""))
        self._embed_model_combo.setCurrentText(str(config.get("embedding_model", "") or ""))
        saved_fast = str(config.get("fast_model", "") or "")
        self._fast_model_combo.setCurrentText(saved_fast)
        if saved_fast:
            self._openai_linked = False
            self._openai_link_cb.setChecked(False)
            self._fast_label.setVisible(True)
            self._fast_model_combo.setVisible(True)
        self._use_ollama_embed.setVisible(False)
        self._connect_status.setText("")
        # Only auto-discover when the user hasn't already saved a custom URL.
        if not saved_url:
            self._start_discovery()
        # Force the wizard to recalculate its height for this page's content.
        # Without this, Qt compresses widgets to fit the wizard's current size
        # instead of growing the window (see CLAUDE.md Qt Layout section).
        wizard = self.wizard()
        if wizard:
            QTimer.singleShot(0, wizard.adjustSize)

    def _start_discovery(self):
        self._connect_status.setText("🔍 Looking for local servers…")
        worker = _DiscoveryWorker(list(self._KNOWN_SERVERS))
        worker.done.connect(self._on_discovered)
        self._discovery_worker = worker  # keep a reference so it isn't GC'd
        worker.start()

    def _on_discovered(self, found: list):
        if not found:
            self._connect_status.setText("")  # nothing running; user enters details
            return
        label, url = found[0]
        # Prefill the first hit unless the user already changed the default.
        if (self._base_url_input.text() or "").strip() in ("", self._DEFAULT_BASE_URL):
            self._base_url_input.setText(url)
        if len(found) == 1:
            self._connect_status.setText(
                f"🔍 Found {label} at {url} — click Connect to load its models.")
        else:
            names = ", ".join(l for l, _ in found)
            self._connect_status.setText(
                f"🔍 Found {len(found)} servers ({names}). Pick one above, then Connect.")

    @staticmethod
    def _is_ready(base_url: str, chat_model: str) -> bool:
        return bool((base_url or "").strip()) and bool((chat_model or "").strip())

    def _read_inputs(self):
        return (
            (self._base_url_input.text() or "").strip(),
            (self._api_key_input.text() or "").strip(),
            (self._chat_model_combo.currentText() or "").strip(),
            (self._embed_model_combo.currentText() or "").strip(),
            (self._fast_model_combo.currentText() or "").strip(),
        )

    def isComplete(self) -> bool:
        base_url, _, chat_model, _, _ = self._read_inputs()
        return self._is_ready(base_url, chat_model)

    def validatePage(self) -> bool:
        """Persist the connection details. Required fields are always
        written; optional ones (API key, embedding model) are omitted when
        empty to keep config.json minimal."""
        base_url, api_key, chat_model, embed_model, fast_model = self._read_inputs()
        if not self._is_ready(base_url, chat_model):
            return False
        try:
            from jarvis.config import default_config_path, _load_json, _save_json
            config_path = default_config_path()
            config = _load_json(config_path) or {}

            config["llm_provider"] = "openai_compatible"
            config["llm_base_url"] = base_url
            config["llm_chat_model"] = chat_model
            if api_key:
                config["llm_api_key"] = api_key
            else:
                config.pop("llm_api_key", None)

            # Embeddings: when the server can't embed and the user opted for the
            # Ollama fallback, route embeddings to Ollama and drop the remote
            # embedding model (Ollama's default applies). Otherwise keep
            # embeddings on this provider, writing the model only when set.
            if self._use_ollama_embed.isVisible() and self._use_ollama_embed.isChecked():
                config["embedding_provider"] = "ollama"
                config.pop("embedding_model", None)
            else:
                config.pop("embedding_provider", None)
                if embed_model:
                    config["embedding_model"] = embed_model
                else:
                    config.pop("embedding_model", None)

            # Save fast model: when linked it's left empty (defaults to chat)
            # Save fast model: write only when unlinked and non-empty
            is_linked = True  # safe default (backward compat for __new__-constructed pages)
            try:
                is_linked = self._openai_linked
            except Exception:
                pass
            config["fast_model"] = fast_model if (not is_linked and fast_model) else ""

            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.dictation_page_id
        return super().nextId()


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
    """Page for installing required AI models — dual-category (fast + chat)."""

    MODEL_OPTIONS = SUPPORTED_CHAT_MODELS
    _ALL_MODELS = MODEL_OPTIONS
    _FAST_MODEL_IDS = ["qwen3.5:0.8b", "gemma4:e2b"]

    # VRAM overhead for always-running companion models (MB).
    # nomic-embed-text: ~1 GB for ~1.5K dim semantic search.
    # SenseVoiceSmall: ~1 GB in float32 (the wizard balance default).
    _EMBED_VRAM_MB = 1024

    def _sensevoice_vram_mb(self) -> int:
        """VRAM in MB used by the SenseVoice speech-recognition model."""
        return SenseVoiceSetupPage.get_sensevoice_vram_mb()

    _WIZARD_HEIGHT_BASE = 875
    _WIZARD_HEIGHT_WITH_BUTTONS = 955
    _WIZARD_HEIGHT_INSTALLING = 1170

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self._linked = False
        self._chat_model = DEFAULT_CHAT_MODEL
        self._fast_model = "gemma4:e2b"
        self._detected_vram_mb = None

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("🧠 Install AI Models")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Jarvis needs a chat model (conversations) and a fast model "
            "(voice intent, tool routing). Pick them separately for "
            "best VRAM usage."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        # Link toggle (off by default — separate models recommended)
        self._link_cb = QCheckBox("\u2699\ufe0f Use same model for both roles")
        self._link_cb.setChecked(False)
        self._link_cb.setStyleSheet("font-size: 14px; color: #e4e4e7; padding: 4px 0;")
        self._link_cb.toggled.connect(self._on_link_toggled)
        layout.addWidget(self._link_cb)

        hint = QLabel(
            "When linked, one model handles both chat and fast tasks "
            "(shares VRAM). Separate models let you pick a lightweight "
            "fast model."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: #71717a; padding: 0 0 0 24px;")
        layout.addWidget(hint)
        layout.addSpacing(8)

        # Model selection card with dropdowns
        selection_card = QFrame()
        selection_card.setObjectName("card")
        card_layout = QVBoxLayout(selection_card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(10)

        # Chat model dropdown
        chat_label = QLabel("🎯 Chat Model (conversations)")
        chat_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #fbbf24;")
        card_layout.addWidget(chat_label)

        self._chat_combo = QComboBox()
        self._chat_combo.setMinimumHeight(36)
        self._chat_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mid in self._ALL_MODELS:
            info = self._ALL_MODELS[mid]
            self._chat_combo.addItem(f"{info['name']}  •  VRAM: {info['vram']}", mid)
        self._chat_combo.setCurrentIndex(self._chat_combo.findData(self._chat_model))
        self._chat_combo.currentIndexChanged.connect(self._on_chat_combo_changed)
        card_layout.addWidget(self._chat_combo)

        card_layout.addSpacing(8)

        # Fast model dropdown
        fast_label = QLabel("⚡ Fast Model (voice intent, tool routing)")
        fast_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #a78bfa;")
        card_layout.addWidget(fast_label)

        self._fast_combo = QComboBox()
        self._fast_combo.setMinimumHeight(36)
        self._fast_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mid in self._FAST_MODEL_IDS:
            info = self._ALL_MODELS[mid]
            self._fast_combo.addItem(f"{info['name']}  •  VRAM: {info['vram']}", mid)
        self._fast_combo.setCurrentIndex(self._fast_combo.findData(self._fast_model))
        self._fast_combo.currentIndexChanged.connect(self._on_fast_combo_changed)
        card_layout.addWidget(self._fast_combo)

        layout.addWidget(selection_card)

        # VRAM bar
        self._detected_vram_mb = detect_total_vram_mb()
        self._vram_bar = QFrame()
        self._vram_bar.setObjectName("card")
        self._vram_bar.setStyleSheet("QFrame#card { padding: 12px 20px; }")
        vl = QVBoxLayout(self._vram_bar)
        vl.setContentsMargins(24, 16, 24, 16)
        vl.setSpacing(4)
        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e4e4e7;")
        vl.addWidget(self._vram_label)
        self._vram_detail = QLabel("")
        self._vram_detail.setWordWrap(True)
        self._vram_detail.setStyleSheet("font-size: 12px; color: #71717a;")
        vl.addWidget(self._vram_detail)
        layout.addWidget(self._vram_bar)

        # Required models card
        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(12)
        mt = QLabel("📦 Required Models")
        mt.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        cl.addWidget(mt)
        cl.addSpacing(8)
        self.models_label = QLabel("Loading...")
        self.models_label.setWordWrap(True)
        self.models_label.setStyleSheet("line-height: 1.6;")
        cl.addWidget(self.models_label)
        layout.addWidget(card)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setVisible(False)
        self.log_output.setMaximumHeight(150)
        layout.addWidget(self.log_output)

        # Buttons
        bl = QHBoxLayout()
        bl.setSpacing(12)
        self.install_btn = QPushButton("📥 Install Missing Models")
        self.install_btn.clicked.connect(self._install_models)
        bl.addWidget(self.install_btn)
        self.skip_btn = QPushButton("⏭️ Skip")
        self.skip_btn.setObjectName("secondary")
        self.skip_btn.clicked.connect(self._skip_models)
        bl.addWidget(self.skip_btn)
        bl.addStretch()
        layout.addLayout(bl)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        self.setLayout(layout)

        self._is_complete = False
        self._missing_models = []
        self._current_model_index = 0
        self._worker = None

        if self._detected_vram_mb is not None:
            # Account for companion-model overhead so the recommendation
            # leaves room for embeddings + sensevoice alongside the chat model.
            overhead = self._EMBED_VRAM_MB + self._sensevoice_vram_mb()
            usable_mb = self._detected_vram_mb - overhead
            rec = get_recommended_model_id(usable_mb if usable_mb > 0 else None)
            if rec in self._ALL_MODELS:
                self._chat_model = rec
                # Fast model stays gemma4:e2b unless VRAM constrains it
                cv = required_vram_mb(rec) or 0
                fv = required_vram_mb(self._fast_model) or 0
                if cv + fv + overhead > self._detected_vram_mb:
                    for c in self._FAST_MODEL_IDS:
                        rc = required_vram_mb(c) or 0
                        if rc <= cv and cv + rc + overhead <= self._detected_vram_mb:
                            self._fast_model = c
                            break
                self._sync_combo_states()
        self._refresh_vram_display()
        self._update_models_display()

    def _build_linked_view(self):
        """No-op — kept for backward compat. Selection uses dropdowns."""
        pass

    def _build_unlinked_view(self):
        """No-op — kept for backward compat. Selection uses dropdowns."""
        pass

    def _make_button(self, info, compact=False):
        """No-op — kept for backward compat. Selection uses dropdowns."""
        pass

    def _on_link_toggled(self, linked):
        self._linked = linked
        if linked:
            self._fast_model = self._chat_model
        self._sync_combo_states()
        self._refresh_vram_display()
        self._update_models_display()

    def _on_linked_selected(self, mid):
        """No-op — selection uses dropdowns."""
        pass

    def _on_fast_combo_changed(self, idx):
        """Handle fast model combo change."""
        mid = self._fast_combo.itemData(idx)
        if mid:
            self._fast_model = mid
            if self._linked:
                self._chat_model = mid
                self._chat_combo.setCurrentIndex(self._chat_combo.findData(mid))
            self._refresh_vram_display()
            self._update_models_display()

    def _on_chat_combo_changed(self, idx):
        """Handle chat model combo change with auto-downgrade for fast model."""
        mid = self._chat_combo.itemData(idx)
        if not mid:
            return
        self._chat_model = mid
        if self._linked:
            self._fast_model = mid
            self._fast_combo.setCurrentIndex(self._fast_combo.findData(mid))
        else:
            # Auto-downgrade: if fast model needs more VRAM than chat model,
            # or the total (chat + fast + embed + sensevoice) exceeds our GPU,
            # pick the smallest fast-suitable model that fits.
            overhead = self._EMBED_VRAM_MB + self._sensevoice_vram_mb()
            cv = required_vram_mb(mid) or 0
            fv = required_vram_mb(self._fast_model) or 0
            exceeds_vram = (
                self._detected_vram_mb is not None
                and cv + fv + overhead > self._detected_vram_mb
            )
            if fv > cv or exceeds_vram:
                for c in self._FAST_MODEL_IDS:
                    rc = required_vram_mb(c) or 0
                    fits_vram = (
                        self._detected_vram_mb is None
                        or cv + rc + overhead <= self._detected_vram_mb
                    )
                    if rc <= cv and fits_vram:
                        self._fast_model = c
                        self._fast_combo.setCurrentIndex(self._fast_combo.findData(c))
                        break
        self._refresh_vram_display()
        self._update_models_display()

    def _sync_combo_states(self):
        """Sync combo selections to reflect current model choices."""
        ci = self._chat_combo.findData(self._chat_model)
        if ci >= 0:
            self._chat_combo.setCurrentIndex(ci)
        fi = self._fast_combo.findData(self._fast_model)
        if fi >= 0:
            self._fast_combo.setCurrentIndex(fi)
        self._refresh_vram_display()
        self._update_models_display()

    def _refresh_vram_display(self):
        overhead = self._EMBED_VRAM_MB + self._sensevoice_vram_mb()
        fv = required_vram_mb(self._fast_model) or 0
        cv = required_vram_mb(self._chat_model) or 0
        if self._linked or self._fast_model == self._chat_model:
            total = cv + overhead
            detail = f"(chat {cv // 1024} GB + embed+sensevoice {overhead // 1024} GB — shared VRAM)"
        else:
            total = fv + cv + overhead
            detail = (f"(fast {fv // 1024} GB + chat {cv // 1024} GB "
                      f"+ embed+sensevoice {overhead // 1024} GB)")
        tg = total / 1024
        if self._detected_vram_mb is not None:
            dg = self._detected_vram_mb / 1024
            self._vram_label.setText(
                f"Total VRAM Required: {tg:.1f} GB    "
                f"Your GPU: {dg:.1f} GB"
            )
            if total > self._detected_vram_mb:
                sg = (total - self._detected_vram_mb) / 1024
                self._vram_detail.setText(
                    f"Your GPU has {dg:.1f} GB VRAM but the selected "
                    f"models need {tg:.1f} GB ({sg:.1f} GB over). "
                    "Switch to smaller models or use CPU fallback."
                )
                self._vram_detail.setStyleSheet(
                    "font-size: 12px; color: #f87171; padding-top: 2px;"
                )
            else:
                self._vram_detail.setText(detail)
                self._vram_detail.setStyleSheet("font-size: 12px; color: #71717a; padding-top: 2px;")
        else:
            self._vram_label.setText(f"Total VRAM Required: {tg:.1f} GB")
            self._vram_detail.setText(detail)

    def _update_models_display(self):
        wiz = self.wizard()
        em = "nomic-embed-text"
        try:
            em = load_settings().ollama_embed_model
        except Exception:
            pass
        req = [self._chat_model]
        if self._fast_model not in req:
            req.append(self._fast_model)
        if em not in req:
            req.append(em)
        installed = []
        if isinstance(wiz, SetupWizard) and wiz.ollama_status:
            installed = wiz.ollama_status.installed_models
        def norm(n):
            return n[:-(len(":latest"))] if n.endswith(":latest") else n
        inorm = {norm(m) for m in installed}
        self._missing_models = [m for m in req if norm(m) not in inorm and m not in installed]
        rinst = [m for m in req if norm(m) in inorm or m in installed]
        if self._missing_models:
            self.models_label.setText(
                f"Missing: {', '.join('X ' + m for m in self._missing_models)}"
            )
            self._is_complete = False
            self.install_btn.setVisible(True)
            self.install_btn.setEnabled(True)
            self.skip_btn.setVisible(True)
            if not self.progress.isVisible():
                self._set_wizard_height(self._WIZARD_HEIGHT_WITH_BUTTONS)
        else:
            self.models_label.setText(f"All required models are installed: {', '.join(rinst)}")
            self._is_complete = True
            self.install_btn.setVisible(False)
            self.skip_btn.setVisible(False)
            if not self.progress.isVisible():
                self._set_wizard_height(self._WIZARD_HEIGHT_BASE)
        self.completeChanged.emit()

    def _save_model_to_config(self):
        try:
            from jarvis.config import _load_json, _save_json
            cp = default_config_path()
            cp.parent.mkdir(parents=True, exist_ok=True)
            cfg = _load_json(cp) or {}
            cfg["ollama_chat_model"] = self._chat_model
            cfg["fast_model"] = self._fast_model
            return _save_json(cp, cfg)
        except Exception:
            return False

    def initializePage(self):
        cc = DEFAULT_CHAT_MODEL
        fc = "gemma4:e2b"
        try:
            c = load_settings()
            cc = c.ollama_chat_model
            fc = getattr(c, "fast_model", "gemma4:e2b")
        except Exception:
            pass
        self._chat_model = cc if cc in self._ALL_MODELS else DEFAULT_CHAT_MODEL
        self._fast_model = fc if fc in self._ALL_MODELS else "gemma4:e2b"
        overhead = self._EMBED_VRAM_MB + self._sensevoice_vram_mb()
        cv = required_vram_mb(self._chat_model) or 0
        fv = required_vram_mb(self._fast_model) or 0
        exceeds_vram = (
            self._detected_vram_mb is not None
            and cv + fv + overhead > self._detected_vram_mb
        )
        if fv > cv or exceeds_vram:
            for c in self._FAST_MODEL_IDS:
                rc = required_vram_mb(c) or 0
                fits_vram = (
                    self._detected_vram_mb is None
                    or cv + rc + overhead <= self._detected_vram_mb
                )
                if rc <= cv and fits_vram:
                    self._fast_model = c
                    break
        # Default to unlinked — separate fast model is the recommended layout
        # even when both happen to be the same model ID.
        self._linked = False
        self._link_cb.setChecked(False)
        self._sync_combo_states()
        self._refresh_vram_display()
        self._update_models_display()
        # Force the wizard to recalculate its height for this page's content.
        wiz = self.wizard()
        if wiz:
            QTimer.singleShot(0, wiz.adjustSize)

    def _install_models(self):
        if not self._save_model_to_config():
            self.status_label.setText("Could not save model selection. Continuing...")
            self.status_label.setStyleSheet("color: #fbbf24;")
        if not self._missing_models:
            self._is_complete = True
            self.completeChanged.emit()
            return
        self._current_model_index = 0
        self._install_next_model()

    def _install_next_model(self):
        if self._current_model_index >= len(self._missing_models):
            self.progress.setVisible(False)
            self.log_output.setVisible(False)
            self.log_output.clear()
            self._update_models_display()
            self.status_label.setText("All models installed!")
            self.status_label.setStyleSheet("color: #4ade80;")
            return
        m = self._missing_models[self._current_model_index]
        self.install_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log_output.setVisible(True)
        self._set_wizard_height(self._WIZARD_HEIGHT_INSTALLING)
        self.status_label.setText(f"Installing {m}... ({self._current_model_index + 1}/{len(self._missing_models)})")
        self.status_label.setStyleSheet("color: #a1a1aa;")
        op = "ollama"
        w = self.wizard()
        if isinstance(w, SetupWizard) and w.ollama_status and w.ollama_status.cli_path:
            op = w.ollama_status.cli_path
        self._worker = CommandWorker([op, "pull", m])
        self._worker.output.connect(self._on_install_output)
        self._worker.completed.connect(self._on_install_finished)
        self._worker.start()

    def _on_install_output(self, text):
        self.log_output.append(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def _on_install_finished(self, success, message):
        if success:
            m = self._missing_models[self._current_model_index]
            w = self.wizard()
            if isinstance(w, SetupWizard) and w.ollama_status:
                if m not in w.ollama_status.installed_models:
                    w.ollama_status.installed_models.append(m)
            self._current_model_index += 1
            self._install_next_model()
        else:
            self.progress.setVisible(False)
            self.status_label.setText(f"Failed to install model. {message}")
            self.status_label.setStyleSheet("color: #f87171;")
            self.install_btn.setEnabled(True)
            self.skip_btn.setEnabled(True)

    def _skip_models(self):
        self._is_complete = True
        self.status_label.setText("Skipped model installation. Jarvis may not work correctly.")
        self.status_label.setStyleSheet("color: #fbbf24;")
        self.completeChanged.emit()

    def isComplete(self):
        return self._is_complete

    def validatePage(self):
        self._save_model_to_config()
        return True

    def nextId(self):
        w = self.wizard()
        if isinstance(w, SetupWizard):
            return w.dictation_page_id
        return super().nextId()

    def _set_wizard_height(self, height):
        w = self.wizard()
        if w:
            w.setMinimumHeight(height)
            w.resize(w.width(), height)
class SenseVoiceSetupPage(QWizardPage):
    """Page for setting up SenseVoice speech recognition (FunASR).

    SenseVoiceSmall is the single bundled model — there is no size ladder
    or language toggle. The page surfaces the FunASR runtime status (with
    an on-demand install when missing) and lets the user pick the compute
    device. The model weights ship with the app, so installed users never
    download anything at runtime.
    """

    # SenseVoiceSmall: 234M params, ~940 MB weights, ~1 GB VRAM in float32.
    _SENSEVOICE_VRAM_MB = 1024
    DEFAULT_MODEL_ID = "FunAudioLLM/SenseVoiceSmall"

    @staticmethod
    def get_sensevoice_vram_mb() -> int:
        """VRAM in MB used by the SenseVoice model (ModelsPage VRAM budget)."""
        return SenseVoiceSetupPage._SENSEVOICE_VRAM_MB

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        self._is_bundled = getattr(sys, 'frozen', False)

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

        title = QLabel("🎤 Speech Recognition (SenseVoice)")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Offline, fast, and multilingual (Mandarin, Cantonese, English, "
            "Japanese, Korean). The model is bundled with the app — no "
            "downloads, no accounts."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Runtime status card
        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(6)

        status_title = QLabel("⚙️ Speech Recognition Engine")
        status_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fbbf24; background: transparent;")
        status_layout.addWidget(status_title)

        self.funasr_status = self._create_status_row("🧠 FunASR runtime", "Checking...")
        status_layout.addWidget(self.funasr_status)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        status_layout.addWidget(self.progress)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        self.log_output.setVisible(False)
        status_layout.addWidget(self.log_output)

        self.install_btn = QPushButton("📦 Install FunASR")
        self.install_btn.setFixedHeight(32)
        self.install_btn.clicked.connect(self._install_funasr)
        status_layout.addWidget(self.install_btn)

        layout.addWidget(status_card)

        # Device selection card
        device_card = QFrame()
        device_card.setObjectName("card")
        device_layout = QVBoxLayout(device_card)
        device_layout.setContentsMargins(16, 12, 16, 12)
        device_layout.setSpacing(8)

        device_title = QLabel("⚡ Compute Device")
        device_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fbbf24; background: transparent;")
        device_layout.addWidget(device_title)

        self._device_combo = QComboBox()
        self._device_combo.addItem("Auto (CUDA → MPS → CPU)", "auto")
        self._device_combo.addItem("CUDA (NVIDIA GPU)", "cuda")
        self._device_combo.addItem("MPS (Apple Silicon)", "mps")
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.setFixedHeight(34)
        device_layout.addWidget(self._device_combo)

        self._device_info = QLabel(
            "Auto picks the fastest available device. CPU is plenty for "
            "SenseVoiceSmall (≈17× realtime)."
        )
        self._device_info.setWordWrap(True)
        self._device_info.setStyleSheet("font-size: 10px; color: #71717a; background: transparent;")
        device_layout.addWidget(self._device_info)

        layout.addWidget(device_card)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #a1a1aa; background: transparent;")
        layout.addWidget(self.status_label)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self._is_complete = False
        self._worker: Optional[CommandWorker] = None
        self._load_configured_device()
        QTimer.singleShot(0, self._refresh_status)

    def _load_configured_device(self) -> None:
        """Restore the configured device into the combo box."""
        try:
            cfg = load_settings()
            device = str(getattr(cfg, "sensevoice_device", "auto") or "auto").lower()
        except Exception:
            device = "auto"
        index = self._device_combo.findData(device)
        self._device_combo.setCurrentIndex(index if index >= 0 else 0)

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

    def _refresh_status(self) -> None:
        from jarvis.listening.sensevoice import is_available, bundled_model_dir

        available = is_available()
        bundled = bundled_model_dir()
        if available:
            self._update_status_row(self.funasr_status, "✅ Installed", True)
            self.install_btn.setEnabled(False)
            self.install_btn.setText("✅ FunASR Installed")
            if bundled:
                self.status_label.setText(
                    f"✅ Speech recognition ready — model weights bundled at:\n{bundled}"
                )
            else:
                self.status_label.setText(
                    "✅ FunASR installed. The model will be downloaded on first use "
                    "(or bundle it with the app via scripts/fetch_sensevoice_model.py)."
                )
        else:
            self._update_status_row(self.funasr_status, "❌ Not installed", False)
            self.install_btn.setEnabled(True)
            self.install_btn.setText("📦 Install FunASR")
            self.status_label.setText(
                "💡 FunASR is required for speech recognition. Install it now, "
                "or continue — you can install it later from the Settings window."
            )
        self._is_complete = True
        self.completeChanged.emit()

    def _install_funasr(self):
        """Install FunASR via pip (with the torch/torchaudio it requires)."""
        self.install_btn.setEnabled(False)
        self.install_btn.setText("⏳ Installing...")
        self.progress.setVisible(True)
        self.log_output.setVisible(True)
        self._worker = CommandWorker(
            [sys.executable, "-m", "pip", "install", "funasr", "torch", "torchaudio"]
        )
        self._worker.output.connect(self.log_output.append)
        self._worker.completed.connect(self._on_funasr_installed)
        self._worker.start()

    def _on_funasr_installed(self, success: bool, message: str):
        """Handle FunASR installation completion."""
        self.install_btn.setEnabled(True)
        self.install_btn.setText("📦 Install FunASR")
        self.progress.setVisible(False)
        if success:
            self.status_label.setText("✅ FunASR installed. Speech recognition is ready.")
        else:
            self.status_label.setText(f"❌ Failed to install FunASR: {message}")
        self._refresh_status()

    def isComplete(self) -> bool:
        """Page is complete once the runtime check has finished."""
        return self._is_complete

    def validatePage(self) -> bool:
        """Save the device choice when leaving the page."""
        self._save_device_to_config()
        return True

    def _save_device_to_config(self) -> bool:
        """Save the selected compute device to the config file."""
        try:
            from jarvis.config import _load_json, _save_json

            config_path = default_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config = _load_json(config_path) or {}
            config["sensevoice_device"] = self._device_combo.currentData() or "auto"

            # _save_json keeps the file at 0o600 (it can hold llm_api_key).
            return _save_json(config_path, config)
        except Exception:
            return False

    def nextId(self) -> int:
        """Go to Provider Choice so the user can confirm or change
        their LLM runtime before continuing."""
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.provider_choice_page_id
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

        self.open_ip_btn = QPushButton("🔍 Detect My IP")
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
            try:
                cfg = load_settings()
                location_context = get_location_context(
                    config_ip=cfg.location_ip_address,
                    auto_detect=cfg.location_auto_detect,
                    resolve_cgnat_public_ip=cfg.location_cgnat_resolve_public_ip,
                )
            except Exception:
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
        """Resolve public IP via OpenDNS and populate the input field."""
        from jarvis.utils.location import _resolve_public_ip_via_opendns
        resolved = _resolve_public_ip_via_opendns()
        if resolved:
            self.ip_input.setText(resolved)
            self.test_result_label.setText(f"✅ Detected public IP: {resolved}")
            self.test_result_label.setStyleSheet("color: #4ade80;")
        else:
            self.test_result_label.setText("⚠️ Could not detect public IP via DNS")
            self.test_result_label.setStyleSheet("color: #fbbf24;")

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

        if _is_private_ip(ip):
            self.test_result_label.setText("⚠️ This appears to be a private IP. Use your public IP instead.")
            self.test_result_label.setStyleSheet("color: #fbbf24;")
            self.save_btn.setEnabled(False)
            self._validated_ip = None
            return

        if _is_cgnat_ip(ip):
            self.test_result_label.setText("⚠️ This is a CGNAT IP (100.64.0.0/10). Use your true public IP instead.")
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
            from jarvis.config import _load_json, _save_json

            config_path = default_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config = _load_json(config_path) or {}
            config["location_ip_address"] = self._validated_ip

            # _save_json keeps the file at 0o600 (it can hold llm_api_key).
            _save_json(config_path, config)

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


class DictationPage(QWizardPage):
    """Page for configuring dictation (hold-to-dictate) settings."""

    @staticmethod
    def _hotkey_options():
        from jarvis.dictation.dictation_engine import format_hotkey_display
        from jarvis.config import _default_dictation_hotkey
        default = _default_dictation_hotkey()
        options = [
            ("ctrl+alt", format_hotkey_display("ctrl+alt")),
            ("ctrl+cmd", format_hotkey_display("ctrl+cmd")),
            ("ctrl+shift+d", format_hotkey_display("ctrl+shift+d")),
            ("ctrl+shift", format_hotkey_display("ctrl+shift")),
        ]
        # Tag the platform default
        return [
            (val, f"{label} (default)" if val == default else label)
            for val, label in options
        ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("🎙️ Dictation Mode")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Hold a hotkey to record speech, release to paste the transcription "
            "into any app. A free, offline alternative to WisprFlow."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # Enabled checkbox
        self._enabled_check = QCheckBox("  Enable dictation mode")
        self._enabled_check.setChecked(True)
        self._enabled_check.setStyleSheet("font-size: 14px; color: #fafafa;")
        layout.addWidget(self._enabled_check)

        layout.addSpacing(4)

        # Filler removal checkbox
        self._filler_check = QCheckBox("  Remove filler words (um, uh, like) using local LLM")
        self._filler_check.setChecked(self._load_current_filler_removal())
        self._filler_check.setStyleSheet("font-size: 14px; color: #fafafa;")
        layout.addWidget(self._filler_check)

        filler_note = QLabel(
            "Uses your chat model to clean up dictation output. "
            "Adds a small delay (~1–3 s) after each dictation."
        )
        filler_note.setWordWrap(True)
        filler_note.setStyleSheet("color: #71717a; font-size: 12px; margin-left: 28px;")
        layout.addWidget(filler_note)

        layout.addSpacing(8)

        # Hotkey selection
        hotkey_card = QFrame()
        hotkey_card.setObjectName("card")
        hotkey_layout = QVBoxLayout(hotkey_card)
        hotkey_layout.setContentsMargins(24, 24, 24, 24)
        hotkey_layout.setSpacing(12)

        hotkey_title = QLabel("⌨️ Dictation Hotkey")
        hotkey_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        hotkey_layout.addWidget(hotkey_title)

        hotkey_desc = QLabel(
            "Choose the key combination you hold down while speaking. "
            "Double-tap the same hotkey for hands-free mode (continuous recording)."
        )
        hotkey_desc.setWordWrap(True)
        hotkey_desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        hotkey_layout.addWidget(hotkey_desc)

        self._hotkey_combo = QComboBox()
        for value, label in self._hotkey_options():
            self._hotkey_combo.addItem(label, value)
        self._hotkey_combo.setStyleSheet(
            "QComboBox { padding: 8px; font-size: 14px; background: #27272a; "
            "color: #fafafa; border: 1px solid #3f3f46; border-radius: 6px; }"
        )

        # Pre-select the current/default hotkey
        current_hotkey = self._load_current_hotkey()
        idx = self._hotkey_combo.findData(current_hotkey)
        if idx >= 0:
            self._hotkey_combo.setCurrentIndex(idx)

        hotkey_layout.addWidget(self._hotkey_combo)
        layout.addWidget(hotkey_card)

        # Tips
        tips_card = QFrame()
        tips_card.setObjectName("card")
        tips_layout = QVBoxLayout(tips_card)
        tips_layout.setContentsMargins(24, 24, 24, 24)
        tips_layout.setSpacing(8)

        tips_title = QLabel("💡 How it Works")
        tips_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        tips_layout.addWidget(tips_title)

        tips = QLabel(
            "• <b>Hold</b> the hotkey to record, <b>release</b> to transcribe and paste\n"
            "• <b>Double-tap</b> the hotkey for hands-free mode (tap again or press Esc to stop)\n"
            "• Uses the same SenseVoice model as voice input — no extra memory\n"
            "• View past dictations from the system tray → 🎙️ Dictation History\n"
            "• Fine-tune in Settings: filler word removal, custom dictionary, and more"
        )
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #d4d4d8; font-size: 13px; line-height: 1.6;")
        tips_layout.addWidget(tips)

        layout.addWidget(tips_card)
        layout.addStretch()
        self.setLayout(layout)

    def _load_current_filler_removal(self) -> bool:
        """Load the current filler removal setting from config, defaulting to False."""
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path())
            if config and "dictation_filler_removal" in config:
                return bool(config["dictation_filler_removal"])
            return False
        except Exception:
            return False

    def _load_current_hotkey(self) -> str:
        """Load the current hotkey from config, or platform default."""
        try:
            from jarvis.config import default_config_path, _load_json, _default_dictation_hotkey
            config = _load_json(default_config_path())
            if config and "dictation_hotkey" in config:
                return config["dictation_hotkey"]
            return _default_dictation_hotkey()
        except Exception:
            if sys.platform == "win32":
                return "ctrl+cmd"
            return "ctrl+alt"

    def validatePage(self) -> bool:
        """Save dictation settings to config before leaving page."""
        try:
            from jarvis.config import default_config_path, _load_json, _save_json
            config_path = default_config_path()
            config = _load_json(config_path) or {}

            enabled = self._enabled_check.isChecked()
            hotkey = self._hotkey_combo.currentData()
            filler_removal = self._filler_check.isChecked()

            config["dictation_enabled"] = enabled
            if hotkey:
                config["dictation_hotkey"] = hotkey
            config["dictation_filler_removal"] = filler_removal

            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def isComplete(self) -> bool:
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.mcp_page_id
        return super().nextId()


class MCPPage(QWizardPage):
    """Page for selecting popular MCP servers to enable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("🔌 MCP Servers")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "MCP (Model Context Protocol) servers give Jarvis extra abilities. "
            "Select any you'd like to enable — you can always change these later in Settings."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Node.js availability warning
        self._node_warning = QLabel(
            "⚠️  <b>Node.js not found.</b> The MCP servers below require Node.js to run. "
            "<a href='https://nodejs.org/' style='color: #f59e0b;'>Download Node.js</a> "
            "and restart Jarvis, or skip this page for now."
        )
        self._node_warning.setOpenExternalLinks(True)
        self._node_warning.setWordWrap(True)
        self._node_warning.setStyleSheet(
            "background: rgba(239, 68, 68, 0.12);"
            "border: 1px solid rgba(239, 68, 68, 0.35);"
            "border-radius: 8px; padding: 12px 16px; color: #fca5a5; font-size: 13px;"
        )
        self._node_warning.setVisible(not self._is_node_available())
        layout.addWidget(self._node_warning)

        # Scrollable cards for wizard-featured entries
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(10)

        self._checkboxes: Dict[str, QCheckBox] = {}
        for entry in get_wizard_entries():
            card = QFrame()
            card.setObjectName("card")
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(14)

            cb = QCheckBox()
            cb.setChecked(self._is_already_configured(entry.name))
            self._checkboxes[entry.name] = cb
            card_layout.addWidget(cb)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)

            name_label = QLabel(entry.display_name)
            name_label.setStyleSheet("font-size: 15px; font-weight: bold;")
            text_layout.addWidget(name_label)

            desc_label = QLabel(entry.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #a1a1aa; font-size: 13px;")
            text_layout.addWidget(desc_label)

            card_layout.addLayout(text_layout, 1)
            inner_layout.addWidget(card)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        # Tip about more MCPs in settings
        tip = QLabel(
            "💡  Many more MCP servers are available in <b>Settings → 🔌 MCP Servers</b>, "
            "including GitHub, Slack, Spotify, and custom servers."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 rgba(245, 158, 11, 0.12), stop:1 rgba(139, 92, 246, 0.08));"
            "border: 1px solid rgba(245, 158, 11, 0.25);"
            "border-radius: 8px; padding: 12px 16px; color: #fbbf24; font-size: 13px;"
        )
        layout.addWidget(tip)

        self.setLayout(layout)

    @staticmethod
    def _is_node_available() -> bool:
        """Check if Node.js (npx) is available on the system."""
        try:
            from jarvis.tools.external.mcp_client import _resolve_command
            _resolve_command("npx")
            return True
        except (FileNotFoundError, Exception):
            return False

    @staticmethod
    def _is_already_configured(name: str) -> bool:
        """Check if an MCP server is already in the user's config."""
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path())
            return name in (config.get("mcps") or {})
        except Exception:
            return False

    def validatePage(self) -> bool:
        """Save selected MCPs to config before leaving page."""
        try:
            from jarvis.config import default_config_path, _load_json, _save_json
            config_path = default_config_path()
            config = _load_json(config_path) or {}

            mcps = config.get("mcps", {})
            if not isinstance(mcps, dict):
                mcps = {}

            for entry in get_wizard_entries():
                cb = self._checkboxes.get(entry.name)
                if cb and cb.isChecked() and entry.name not in mcps:
                    mcps[entry.name] = entry.to_config()
                elif cb and not cb.isChecked() and entry.name in mcps:
                    del mcps[entry.name]

            if mcps:
                config["mcps"] = mcps
            else:
                config.pop("mcps", None)

            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def isComplete(self) -> bool:
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            return wizard.search_providers_page_id
        return super().nextId()


class SearchProvidersPage(QWizardPage):
    """Explain and configure web-search fallback providers.

    Ordering mirrors the runtime fallback chain: DDG → Brave → Wikipedia →
    honest "blocked" envelope. The page is always shown (even when nothing
    needs configuring) because the explainer itself is the point — users
    should understand what Jarvis will and won't reach over the network
    before they start using it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("🔎 Search Providers")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Jarvis uses DuckDuckGo for web search. When DuckDuckGo blocks a "
            "request or has nothing useful, these optional fallbacks keep "
            "answers flowing — all off by default except Wikipedia."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # --- Brave Search card ---
        brave_card = QFrame()
        brave_card.setObjectName("card")
        brave_layout = QVBoxLayout(brave_card)
        brave_layout.setContentsMargins(16, 14, 16, 14)
        brave_layout.setSpacing(8)

        brave_title = QLabel("🦁 Brave Search (optional)")
        brave_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        brave_layout.addWidget(brave_title)

        brave_desc = QLabel(
            "When set, Brave becomes the first fallback the moment "
            "DuckDuckGo is rate-limited. Free tier: 2,000 queries/month. "
            "Get a key at "
            "<a href='https://api.search.brave.com/app/keys' "
            "style='color: #f59e0b;'>api.search.brave.com</a>."
        )
        brave_desc.setOpenExternalLinks(True)
        brave_desc.setWordWrap(True)
        brave_desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        brave_layout.addWidget(brave_desc)

        self._brave_input = QLineEdit()
        self._brave_input.setPlaceholderText("BSA... (leave empty to skip)")
        self._brave_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._brave_input.setText(self._load_current_brave_key())
        brave_layout.addWidget(self._brave_input)

        layout.addWidget(brave_card)

        # --- Wikipedia card ---
        wiki_card = QFrame()
        wiki_card.setObjectName("card")
        wiki_layout = QVBoxLayout(wiki_card)
        wiki_layout.setContentsMargins(16, 14, 16, 14)
        wiki_layout.setSpacing(8)

        wiki_title = QLabel("📚 Wikipedia (zero-config)")
        wiki_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        wiki_layout.addWidget(wiki_title)

        wiki_desc = QLabel(
            "Last-resort fallback. No key, no account, privacy-light. Uses "
            "the Wikipedia host matching the language SenseVoice detects in "
            "your utterance, so a Turkish question gets a Turkish answer."
        )
        wiki_desc.setWordWrap(True)
        wiki_desc.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        wiki_layout.addWidget(wiki_desc)

        self._wiki_check = QCheckBox("  Enable Wikipedia fallback")
        self._wiki_check.setChecked(self._load_current_wikipedia_enabled())
        wiki_layout.addWidget(self._wiki_check)

        layout.addWidget(wiki_card)

        tip = QLabel(
            "💡  When every provider fails, Jarvis tells you the search was "
            "blocked rather than making something up."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 rgba(245, 158, 11, 0.12), stop:1 rgba(139, 92, 246, 0.08));"
            "border: 1px solid rgba(245, 158, 11, 0.25);"
            "border-radius: 8px; padding: 12px 16px; color: #fbbf24; font-size: 13px;"
        )
        layout.addWidget(tip)

        layout.addStretch()

        self.setLayout(layout)

    @staticmethod
    def _load_current_brave_key() -> str:
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path())
            return str(config.get("brave_search_api_key", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _load_current_wikipedia_enabled() -> bool:
        try:
            from jarvis.config import default_config_path, _load_json
            config = _load_json(default_config_path())
            # Default True to match config.py's default.
            val = config.get("wikipedia_fallback_enabled", True)
            return bool(val)
        except Exception:
            return True

    def validatePage(self) -> bool:
        """Persist Brave key + Wikipedia toggle. Only writes non-default
        values to keep config.json minimal (consistent with the settings
        window's "only non-default values written" invariant)."""
        try:
            from jarvis.config import default_config_path, _load_json, _save_json
            config_path = default_config_path()
            config = _load_json(config_path) or {}

            brave_key = (self._brave_input.text() or "").strip()
            if brave_key:
                config["brave_search_api_key"] = brave_key
            else:
                config.pop("brave_search_api_key", None)

            wiki_on = bool(self._wiki_check.isChecked())
            # Default is True; only persist when the user diverges from it.
            if not wiki_on:
                config["wikipedia_fallback_enabled"] = False
            else:
                config.pop("wikipedia_fallback_enabled", None)

            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def isComplete(self) -> bool:
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        if isinstance(wizard, SetupWizard):
            if not wizard.is_location_working():
                return wizard.location_page_id
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
            "• Say your wake word (e.g. 'Jarvis') anywhere in your sentence to activate the assistant\n"
            "• After Jarvis replies, speak your follow-up — no need to repeat the wake word\n"
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

