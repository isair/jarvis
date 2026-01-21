"""
Jarvis Desktop App - System Tray Application

A cross-platform system tray app for controlling the Jarvis voice assistant.
Supports Windows, Ubuntu (Linux), and macOS.
"""

from __future__ import annotations
import sys
import os
import time

# Fix OpenBLAS threading crash in bundled apps
# Must be set before numpy is imported (via faster-whisper, etc.)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import subprocess
import signal
import psutil
import threading
import traceback
import atexit
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel, QDialog
from PyQt6.QtGui import QIcon, QAction, QFont, QTextCursor
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QUrl

# Global lock file handle (must remain open for the lock to persist)
_lock_file_handle = None

# Try to import WebEngine (optional dependency for embedded memory viewer)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False
    QWebEngineView = None

from jarvis.debug import debug_log
from jarvis.diary_dialog import DiaryUpdateDialog
from jarvis.config import _default_config_path, _default_db_path
from jarvis.themes import JARVIS_THEME_STYLESHEET
from jarvis.face_widget import FaceWindow


def setup_crash_logging():
    """Set up crash logging for the bundled app to capture startup errors."""
    if getattr(sys, 'frozen', False):
        # Running as bundled app
        log_dir = Path.home() / "Library" / "Logs" if sys.platform == "darwin" else Path.home()
        log_file = log_dir / "jarvis_desktop_crash.log"

        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Redirect stdout and stderr to log file with line buffering for immediate writes
            # buffering=1 means line-buffered mode (flush on newline)
            log_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
            sys.stdout = log_handle
            sys.stderr = log_handle

            print(f"=== Jarvis Desktop App Crash Log ===", flush=True)
            print(f"Timestamp: {__import__('datetime').datetime.now()}", flush=True)
            print(f"Platform: {sys.platform}", flush=True)
            print(f"Python: {sys.version}", flush=True)
            print(f"Executable: {sys.executable}", flush=True)
            print(f"Frozen: {getattr(sys, 'frozen', False)}", flush=True)
            print(f"Bundle dir: {getattr(sys, '_MEIPASS', 'N/A')}", flush=True)
            print("=" * 50, flush=True)
            print(flush=True)

            return log_file
        except Exception as e:
            # If we can't set up logging, at least try to show a dialog
            return None
    return None


def acquire_single_instance_lock() -> bool:
    """
    Acquire a lock to ensure only one instance of the desktop app runs.

    Returns True if lock acquired (we're the only instance), False otherwise.
    The lock file handle is kept open globally to maintain the lock.
    """
    global _lock_file_handle

    # Use a predictable lock file location
    if sys.platform == "darwin":
        lock_dir = Path.home() / "Library" / "Application Support" / "Jarvis"
    elif sys.platform == "win32":
        lock_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jarvis"
    else:
        lock_dir = Path.home() / ".jarvis"

    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "jarvis_desktop.lock"

    try:
        # Open lock file (create if doesn't exist)
        _lock_file_handle = open(lock_file, 'w')

        if sys.platform == "win32":
            # Windows: use msvcrt for file locking
            import msvcrt
            try:
                msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                # Lock failed - another instance is running
                _lock_file_handle.close()
                _lock_file_handle = None
                return False
        else:
            # Unix (macOS, Linux): use fcntl for file locking
            import fcntl
            try:
                fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                # Lock failed - another instance is running
                _lock_file_handle.close()
                _lock_file_handle = None
                return False

        # Write our PID to the lock file for debugging
        _lock_file_handle.write(str(os.getpid()))
        _lock_file_handle.flush()

        # Register cleanup to release lock on exit
        def release_lock():
            global _lock_file_handle
            if _lock_file_handle:
                try:
                    _lock_file_handle.close()
                except Exception:
                    pass
                _lock_file_handle = None

        atexit.register(release_lock)

        return True

    except Exception as e:
        print(f"Warning: Could not acquire single-instance lock: {e}")
        # On any error, allow the app to run (fail open)
        return True


class LogSignals(QObject):
    """Signals for thread-safe log updates."""
    new_log = pyqtSignal(str)


class LogViewerWindow(QMainWindow):
    """Window for viewing Jarvis logs in real-time."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📝 Jarvis Logs")
        self.setGeometry(100, 100, 900, 650)

        # Apply theme
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 8)
        header_layout.setSpacing(4)

        title = QLabel("📝 Jarvis Logs")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #fbbf24;")
        header_layout.addWidget(title)

        subtitle = QLabel("Real-time activity and debug output")
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(subtitle)

        layout.addWidget(header)

        # Create text display for logs with monospace font
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        mono_font = QFont("JetBrains Mono", 11) if sys.platform == "darwin" else QFont("Consolas", 10)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_display.setFont(mono_font)
        layout.addWidget(self.log_display)

        # Initial message
        self.append_log("🚀 Jarvis Log Viewer Ready\n" + "─"*50 + "\n\n")

    def append_log(self, text: str) -> None:
        """Append text to the log display."""
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)
        self.log_display.insertPlainText(text)
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.log_display.clear()
        self.append_log("📝 Jarvis Logs\n" + "="*60 + "\n")


class MemoryViewerWindow(QMainWindow):
    """Window for viewing Jarvis memory using embedded web view."""

    MEMORY_VIEWER_PORT = 5050

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Jarvis Memory")
        self.setGeometry(150, 150, 1200, 800)

        # Apply theme
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        self.server_process: Optional[subprocess.Popen] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_server_running = False

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        web_view_created = False
        if HAS_WEBENGINE:
            # Use embedded web view - URL will be set in showEvent when window is shown
            try:
                self.web_view = QWebEngineView()
                layout.addWidget(self.web_view)
                web_view_created = True
            except Exception as e:
                debug_log(f"failed to create QWebEngineView: {e}", "desktop")
                self.web_view = None

        if not web_view_created:
            # Fallback: show message with button to open in browser
            self.web_view = None

            fallback_container = QWidget()
            fallback_layout = QVBoxLayout(fallback_container)
            fallback_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            icon_label = QLabel("🧠")
            icon_label.setStyleSheet("font-size: 64px; background: transparent;")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_layout.addWidget(icon_label)

            title_label = QLabel("Memory Viewer")
            title_label.setStyleSheet("""
                font-size: 24px;
                font-weight: 600;
                color: #fbbf24;
                background: transparent;
                margin-top: 16px;
            """)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_layout.addWidget(title_label)

            message_label = QLabel(
                "PyQt6-WebEngine not installed.\n"
                "Opening in your default browser..."
            )
            message_label.setStyleSheet("""
                font-size: 14px;
                color: #71717a;
                background: transparent;
                margin-top: 8px;
            """)
            message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_layout.addWidget(message_label)

            layout.addWidget(fallback_container)

    def start_server(self) -> bool:
        """Start the memory viewer Flask server."""
        if self.is_server_running:
            return True

        try:
            # Check if server is already running on the port
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.MEMORY_VIEWER_PORT))
            sock.close()

            if result == 0:
                # Port is already in use, assume server is running
                self.is_server_running = True
                debug_log(f"memory viewer server already running on port {self.MEMORY_VIEWER_PORT}", "desktop")
                return True

            # Check if we're running as a frozen/bundled app
            is_frozen = getattr(sys, 'frozen', False)

            if is_frozen:
                # Bundled app: run Flask server in a thread
                try:
                    from jarvis.memory_viewer import app as flask_app
                except Exception as import_err:
                    debug_log(f"failed to import memory_viewer: {import_err}", "desktop")
                    return False

                def run_flask_server():
                    try:
                        # Disable Flask's reloader and debug mode
                        flask_app.run(
                            host="127.0.0.1",
                            port=self.MEMORY_VIEWER_PORT,
                            debug=False,
                            use_reloader=False,
                            threaded=True
                        )
                    except Exception as server_err:
                        debug_log(f"memory viewer server error: {server_err}", "desktop")

                self.server_thread = threading.Thread(target=run_flask_server, daemon=True)
                self.server_thread.start()
                debug_log("memory viewer server started in thread (bundled mode)", "desktop")
            else:
                # Development: start server in subprocess
                python_exe = sys.executable

                # Set up environment with PYTHONPATH for source runs
                env = os.environ.copy()
                src_path = Path(__file__).parent.parent  # Go up to src/
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = str(src_path)

                self.server_process = subprocess.Popen(
                    [python_exe, "-m", "jarvis.memory_viewer"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                debug_log("memory viewer server started in subprocess (development mode)", "desktop")

            # Wait a moment for server to start
            import time
            time.sleep(1)

            self.is_server_running = True
            return True

        except Exception as e:
            debug_log(f"failed to start memory viewer server: {e}", "desktop")
            return False

    def stop_server(self) -> None:
        """Stop the memory viewer Flask server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                debug_log(f"error stopping memory viewer server: {e}", "desktop")
            finally:
                self.server_process = None
                self.is_server_running = False

        # Thread-based server (bundled mode) will stop when app exits (daemon thread)
        if self.server_thread:
            self.server_thread = None
            self.is_server_running = False

    def showEvent(self, event) -> None:
        """Called when window is shown."""
        super().showEvent(event)

        try:
            # Start server when window opens
            if self.start_server():
                if self.web_view:
                    # Set URL and load (URL is set here, not in __init__, to avoid WebEngine crash)
                    self.web_view.setUrl(QUrl(f"http://localhost:{self.MEMORY_VIEWER_PORT}"))
                else:
                    # Open in system browser as fallback
                    import webbrowser
                    webbrowser.open(f"http://localhost:{self.MEMORY_VIEWER_PORT}")
            else:
                # Server failed to start, open in browser as fallback
                debug_log("memory viewer server failed to start, opening in browser", "desktop")
                import webbrowser
                webbrowser.open(f"http://localhost:{self.MEMORY_VIEWER_PORT}")
        except Exception as e:
            debug_log(f"error in memory viewer showEvent: {e}", "desktop")
            # Fallback to browser
            import webbrowser
            webbrowser.open(f"http://localhost:{self.MEMORY_VIEWER_PORT}")

    def closeEvent(self, event) -> None:
        """Called when window is closed."""
        # Don't stop the server on close - just hide the window
        # Server will be stopped on app quit
        event.accept()


class JarvisSystemTray:
    """System tray application for Jarvis voice assistant."""

    def __init__(self):
        # Use existing QApplication if available, otherwise create one
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        # Initialize state
        self.daemon_process: Optional[subprocess.Popen] = None
        self.daemon_thread: Optional[QThread] = None
        self.is_listening = False
        self.is_bundled = getattr(sys, 'frozen', False)

        # Kill any orphaned Jarvis processes from previous sessions
        self.cleanup_orphaned_processes()

        # Create log viewer window (hidden by default)
        self.log_viewer = LogViewerWindow()
        self.log_signals = LogSignals()
        self.log_signals.new_log.connect(self.log_viewer.append_log)

        # Create memory viewer window (hidden by default)
        self.memory_viewer = MemoryViewerWindow()

        # Create face window (hidden by default)
        # Note: Creating the face window also initializes the SpeakingState singleton
        # in the main thread, which is important for cross-thread signal delivery
        self.face_window = FaceWindow()

        # Log reader threads
        self.log_reader_threads = []

        # Create system tray icon
        self.tray_icon = QSystemTrayIcon()
        self.update_icon()

        # Create context menu
        self.create_menu()

        # Set up status checking timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_daemon_status)
        self.status_timer.start(2000)  # Check every 2 seconds

        # Show tray icon
        self.tray_icon.show()

        # Register cleanup on app exit
        self.app.aboutToQuit.connect(self.cleanup_on_exit)

        # Check for updates on startup (delayed by 5 seconds to not block app startup)
        QTimer.singleShot(5000, self.check_for_updates)

        debug_log("desktop app initialized", "desktop")

    def cleanup_orphaned_processes(self) -> None:
        """Kill any orphaned Jarvis daemon processes from previous sessions."""
        try:
            current_pid = os.getpid()
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'jarvis.main' in ' '.join(cmdline):
                        # This is a Jarvis daemon process
                        if proc.pid != current_pid:
                            debug_log(f"killing orphaned jarvis process: {proc.pid}", "desktop")
                            proc.terminate()
                            try:
                                proc.wait(timeout=2)
                            except psutil.TimeoutExpired:
                                proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            debug_log(f"error cleaning up orphaned processes: {e}", "desktop")

    def cleanup_on_exit(self) -> None:
        """Cleanup when app is exiting."""
        debug_log("cleaning up on exit", "desktop")
        if self.is_listening:
            self.stop_daemon()
        # Stop memory viewer server
        if hasattr(self, 'memory_viewer'):
            self.memory_viewer.stop_server()
        # Safety net: if daemon process exists but is_listening was False, still clean up
        # (This shouldn't happen in normal operation, but handles edge cases)
        if self.daemon_process:
            try:
                self.daemon_process.terminate()
                try:
                    # Use longer timeout to allow diary update to complete
                    self.daemon_process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.daemon_process.kill()
                    self.daemon_process.wait()
            except Exception as e:
                debug_log(f"error during exit cleanup: {e}", "desktop")

    def create_menu(self) -> None:
        """Create the system tray context menu."""
        self.menu = QMenu()

        # Toggle listening action
        self.toggle_action = QAction("▶️ Start Listening")
        self.toggle_action.triggered.connect(self.toggle_listening)
        self.menu.addAction(self.toggle_action)

        self.menu.addSeparator()

        # View logs action
        self.logs_action = QAction("📝 View Logs")
        self.logs_action.triggered.connect(self.toggle_log_viewer)
        self.menu.addAction(self.logs_action)

        # Memory viewer action
        self.memory_action = QAction("🧠 Memory Viewer")
        self.memory_action.triggered.connect(self.toggle_memory_viewer)
        self.menu.addAction(self.memory_action)

        # Face window action
        self.face_action = QAction("👤 Show Face")
        self.face_action.triggered.connect(self.toggle_face_window)
        self.menu.addAction(self.face_action)

        # Setup wizard action
        self.setup_wizard_action = QAction("🔧 Setup Wizard")
        self.setup_wizard_action.triggered.connect(self.show_setup_wizard)
        self.menu.addAction(self.setup_wizard_action)

        # Check for updates action
        self.check_updates_action = QAction("🔄 Check for Updates")
        self.check_updates_action.triggered.connect(lambda: self.check_for_updates(show_no_update_dialog=True))
        self.menu.addAction(self.check_updates_action)

        self.menu.addSeparator()

        # Open directories actions
        self.open_config_action = QAction("📁 Open Config Directory")
        self.open_config_action.triggered.connect(self.open_config_directory)
        self.menu.addAction(self.open_config_action)

        self.open_data_action = QAction("💾 Open Data Directory")
        self.open_data_action.triggered.connect(self.open_data_directory)
        self.menu.addAction(self.open_data_action)

        self.menu.addSeparator()

        # Status action (non-clickable)
        self.status_action = QAction("⚪ Status: Stopped")
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)

        self.menu.addSeparator()

        # Quit action
        self.quit_action = QAction("🚪 Quit")
        self.quit_action.triggered.connect(self.quit_app)
        self.menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(self.menu)

    def show_setup_wizard(self) -> None:
        """Show the setup wizard window."""
        from jarvis.setup_wizard import SetupWizard
        from PyQt6.QtWidgets import QWizard

        # Remember if daemon was running before wizard
        was_listening = self.is_listening

        # Stop daemon while setup wizard is open (to allow changes to take effect)
        if was_listening:
            self.stop_daemon()

        wizard = SetupWizard()
        result = wizard.exec()

        # Restart daemon after wizard completes (finished or cancelled)
        # This ensures any config changes (model selection, etc.) are applied
        # For first-time users: daemon wasn't running, so we start it
        # For existing users: restart to apply changes
        if result == QWizard.DialogCode.Accepted or was_listening:
            self.start_daemon()

    def check_for_updates(self, show_no_update_dialog: bool = False) -> None:
        """Check for available updates.

        Args:
            show_no_update_dialog: If True, shows a dialog even when no update is available.
        """
        from jarvis.updater import check_for_updates, is_frozen
        from jarvis.update_dialog import (
            UpdateAvailableDialog,
            UpdateProgressDialog,
            show_no_update_dialog as show_no_update,
            show_update_error_dialog,
        )

        # Only check for updates if running as bundled app
        if not is_frozen():
            if show_no_update_dialog:
                from PyQt6.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Updates")
                msg.setText("Auto-update is only available in the bundled desktop app.")
                msg.setInformativeText("You're running from source. Use git pull to update.")
                msg.setStyleSheet(JARVIS_THEME_STYLESHEET)
                msg.exec()
            return

        try:
            status = check_for_updates()

            if status.error:
                debug_log(f"Update check failed: {status.error}", "desktop")
                if show_no_update_dialog:
                    show_update_error_dialog(status.error)
                return

            if status.update_available and status.latest_release:
                # Show update available dialog
                dialog = UpdateAvailableDialog(status)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    # User chose to update
                    progress_dialog = UpdateProgressDialog(status.latest_release)
                    progress_dialog.show()
                    progress_dialog.start_download()

                    result = progress_dialog.exec()
                    if result == QDialog.DialogCode.Accepted:
                        # Update successful, exit app
                        self.quit_app()
            elif show_no_update_dialog:
                show_no_update(status.current_version)

        except Exception as e:
            debug_log(f"Update check error: {e}", "desktop")
            if show_no_update_dialog:
                show_update_error_dialog(str(e))

    def toggle_log_viewer(self) -> None:
        """Toggle the log viewer window visibility."""
        if self.log_viewer.isVisible():
            self.log_viewer.hide()
        else:
            self.log_viewer.show()
            self.log_viewer.raise_()
            self.log_viewer.activateWindow()

    def toggle_memory_viewer(self) -> None:
        """Toggle the memory viewer window visibility."""
        if self.memory_viewer.isVisible():
            self.memory_viewer.hide()
        else:
            self.memory_viewer.show()
            self.memory_viewer.raise_()
            self.memory_viewer.activateWindow()

    def toggle_face_window(self) -> None:
        """Toggle the face window visibility."""
        if self.face_window.isVisible():
            self.face_window.hide()
        else:
            self.face_window.show()
            self.face_window.raise_()
            self.face_window.activateWindow()

    def open_directory(self, directory_path: Path, directory_name: str) -> None:
        """Open a directory in the system file manager."""
        try:
            # Ensure directory exists
            directory_path.mkdir(parents=True, exist_ok=True)

            # Open directory based on platform
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", str(directory_path)])
            elif sys.platform == "win32":  # Windows
                os.startfile(str(directory_path))
            else:  # Linux and other Unix-like systems
                subprocess.Popen(["xdg-open", str(directory_path)])

            debug_log(f"opened {directory_name} directory: {directory_path}", "desktop")
            self.log_signals.new_log.emit(f"📂 Opened {directory_name} directory\n")
        except Exception as e:
            debug_log(f"failed to open {directory_name} directory: {e}", "desktop")
            self.log_signals.new_log.emit(f"❌ Failed to open {directory_name} directory: {str(e)}\n")
            self.tray_icon.showMessage(
                f"Error Opening {directory_name} Directory",
                f"Failed to open directory: {str(e)}",
                QSystemTrayIcon.MessageIcon.Warning,
                3000
            )

    def open_config_directory(self) -> None:
        """Open the configuration directory in the system file manager."""
        config_path = _default_config_path()
        config_dir = config_path.parent
        self.open_directory(config_dir, "Config")

    def open_data_directory(self) -> None:
        """Open the data directory (where database is stored) in the system file manager."""
        db_path = Path(_default_db_path())
        data_dir = db_path.parent
        self.open_directory(data_dir, "Data")

    def get_icon_path(self, icon_name: str) -> Path:
        """Get the path to an icon file."""
        # Try to find icons in the package directory
        package_dir = Path(__file__).parent
        icons_dir = package_dir / "desktop_assets"
        icon_path = icons_dir / icon_name

        if icon_path.exists():
            return icon_path

        # Fallback: return a simple colored icon
        return icon_path

    def update_icon(self) -> None:
        """Update the tray icon based on current state."""
        if self.is_listening:
            icon_name = "icon_listening.png"
        else:
            icon_name = "icon_idle.png"

        icon_path = self.get_icon_path(icon_name)

        # If icon file doesn't exist, use a default from system
        if icon_path.exists():
            icon = QIcon(str(icon_path))
        else:
            # Use a simple text-based icon as fallback
            from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)

            # Draw a circle
            color = QColor("#4CAF50" if self.is_listening else "#9E9E9E")
            painter.setBrush(color)
            painter.setPen(color)
            painter.drawEllipse(4, 4, 56, 56)

            # Draw letter J
            painter.setPen(Qt.GlobalColor.white)
            font = QFont("Arial", 32, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "J")

            painter.end()
            icon = QIcon(pixmap)

        self.tray_icon.setIcon(icon)

    def toggle_listening(self) -> None:
        """Toggle the Jarvis daemon on/off."""
        if self.is_listening:
            self.stop_daemon()
        else:
            self.start_daemon()

    def start_daemon(self) -> None:
        """Start the Jarvis daemon."""
        try:
            if self.is_bundled:
                # When bundled, run daemon in a QThread since Qt components may be used

                class DaemonThread(QThread):
                    """QThread to run the daemon."""
                    def __init__(self, log_signals):
                        super().__init__()
                        self.log_signals = log_signals

                    def run(self):
                        """Run the daemon in this QThread."""
                        import sys as sys_module
                        old_stdout = sys_module.stdout
                        old_stderr = sys_module.stderr

                        try:
                            # Redirect stdout/stderr to capture logs
                            class LogWriter:
                                def __init__(self, emit_func):
                                    self.emit_func = emit_func
                                    self.buffer = ""

                                def write(self, text):
                                    if text:
                                        # Handle both bytes and str (Flask can send bytes)
                                        if isinstance(text, bytes):
                                            text = text.decode('utf-8', errors='replace')
                                        self.buffer += text
                                        if '\n' in self.buffer:
                                            lines = self.buffer.split('\n')
                                            self.buffer = lines[-1]
                                            for line in lines[:-1]:
                                                if line.strip():
                                                    self.emit_func(line + '\n')

                                def flush(self):
                                    if self.buffer.strip():
                                        self.emit_func(self.buffer)
                                        self.buffer = ""

                            log_writer = LogWriter(self.log_signals.new_log.emit)
                            sys_module.stdout = log_writer
                            sys_module.stderr = log_writer

                            try:
                                # Import and run the daemon
                                from jarvis.daemon import main as daemon_main
                                self.log_signals.new_log.emit("🚀 Jarvis daemon started\n")
                                self.log_signals.new_log.emit("📋 Initializing daemon components...\n")

                                # Run daemon - this should run the main loop
                                daemon_main()

                                self.log_signals.new_log.emit("⏸️ Daemon main() returned (unexpected)\n")
                            except KeyboardInterrupt:
                                self.log_signals.new_log.emit("⏸️ Daemon interrupted\n")
                            except Exception as e:
                                error_msg = f"❌ Daemon runtime error: {str(e)}\n{traceback.format_exc()}\n"
                                self.log_signals.new_log.emit(error_msg)
                                # Also try to log via debug_log (though it might not work)
                                try:
                                    debug_log(f"daemon thread error: {e}", "desktop")
                                except:
                                    pass
                            finally:
                                sys_module.stdout = old_stdout
                                sys_module.stderr = old_stderr
                        except Exception as e:
                            # Outer exception handler for setup errors
                            error_msg = f"❌ Daemon setup error: {str(e)}\n{traceback.format_exc()}\n"
                            try:
                                self.log_signals.new_log.emit(error_msg)
                            except:
                                # If we can't emit, at least try stdout
                                print(error_msg, file=old_stderr)

                self.daemon_thread = DaemonThread(self.log_signals)
                # Connect finished signal to reset UI state
                self.daemon_thread.finished.connect(lambda: self._on_daemon_finished())
                self.daemon_thread.start()
            else:
                # When not bundled, use subprocess as before
                python_exe = sys.executable

                # Set up environment with PYTHONPATH for source runs
                env = os.environ.copy()
                src_path = Path(__file__).parent.parent.parent  # Go up to src/
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = str(src_path)

                self.daemon_process = subprocess.Popen(
                    [python_exe, "-m", "jarvis.main"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )

                # Start log reader thread
                log_thread = threading.Thread(
                    target=self._read_daemon_logs,
                    daemon=True
                )
                log_thread.start()
                self.log_reader_threads.append(log_thread)
                self.log_signals.new_log.emit("🚀 Jarvis daemon started\n")

            self.is_listening = True
            self.toggle_action.setText("⏸️ Stop Listening")
            self.status_action.setText("🟢 Status: Listening")
            self.update_icon()

            # Show log viewer when starting listening
            self.log_viewer.show()
            self.log_viewer.raise_()
            self.log_viewer.activateWindow()

            self.tray_icon.showMessage(
                "Jarvis Started",
                "Voice assistant is now listening",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )

            # Show face window when starting
            self.face_window.show()
            self.face_window.raise_()

            debug_log("daemon started from desktop app", "desktop")

        except Exception as e:
            debug_log(f"failed to start daemon: {e}", "desktop")
            self.log_signals.new_log.emit(f"❌ Failed to start: {str(e)}\n{traceback.format_exc()}\n")
            self.tray_icon.showMessage(
                "Error Starting Jarvis",
                f"Failed to start: {str(e)}",
                QSystemTrayIcon.MessageIcon.Critical,
                3000
            )

    def _on_daemon_finished(self) -> None:
        """Called when daemon thread finishes."""
        if self.is_listening:
            self.is_listening = False
            self.toggle_action.setText("▶️ Start Listening")
            self.status_action.setText("⚪ Status: Stopped")
            self.update_icon()
            self.daemon_thread = None

    def _read_daemon_logs(self) -> None:
        """Read logs from daemon subprocess in a background thread."""
        if not self.daemon_process or not self.daemon_process.stdout:
            return

        try:
            for line in self.daemon_process.stdout:
                if line:
                    self.log_signals.new_log.emit(line)
        except Exception as e:
            debug_log(f"log reader error: {e}", "desktop")

    def stop_daemon(self, show_diary_dialog: bool = True) -> None:
        """Stop the Jarvis daemon.

        Args:
            show_diary_dialog: If True (and bundled), shows a dialog with live diary update progress.
        """
        # Timeout must be longer than SHUTDOWN_DIARY_TIMEOUT_SEC (45s) in daemon.py
        # to allow the diary update LLM call to complete before force-killing
        shutdown_wait_timeout_sec = 60
        diary_dialog = None

        debug_log(f"stop_daemon called: is_bundled={self.is_bundled}, daemon_thread={self.daemon_thread}, show_diary_dialog={show_diary_dialog}", "desktop")

        try:
            if self.is_bundled and self.daemon_thread:
                # When running in a QThread, use the stop flag for graceful shutdown
                # This ensures the daemon's finally block runs (for diary update)
                self.log_signals.new_log.emit("⏸️ Stopping Jarvis daemon...\n")

                # Show diary update dialog for bundled app
                if show_diary_dialog:
                    diary_dialog = DiaryUpdateDialog()

                    # Set up thread-safe callbacks that emit Qt signals
                    # These callbacks run in the daemon thread, so we use signals
                    def on_token(token: str):
                        diary_dialog.signals.token_received.emit(token)

                    def on_status(status: str):
                        diary_dialog.signals.status_changed.emit(status)

                    def on_chunks(chunks: list):
                        # Schedule on main thread
                        QTimer.singleShot(0, lambda: diary_dialog.set_conversations(chunks))

                    def on_complete(success: bool):
                        diary_dialog.signals.completed.emit(success)

                    # Set callbacks in daemon before requesting stop
                    from jarvis.daemon import set_diary_update_callbacks, request_stop
                    set_diary_update_callbacks(
                        on_token=on_token,
                        on_status=on_status,
                        on_chunks=on_chunks,
                        on_complete=on_complete,
                    )

                    # Hide other windows while showing diary dialog
                    if hasattr(self, 'face_window') and self.face_window and self.face_window.isVisible():
                        self.face_window.hide()
                    if hasattr(self, 'log_viewer') and self.log_viewer.isVisible():
                        self.log_viewer.hide()

                    # Show dialog (non-modal so we can process events)
                    diary_dialog.show()
                    diary_dialog.raise_()
                    diary_dialog.activateWindow()
                    self.app.processEvents()

                    # Request graceful stop
                    request_stop()

                    # Process events while waiting for thread to finish
                    start_time = time.time()
                    while not self.daemon_thread.isFinished():
                        self.app.processEvents()
                        elapsed = time.time() - start_time
                        if elapsed > shutdown_wait_timeout_sec:
                            self.log_signals.new_log.emit("⚠️ Daemon taking too long, forcing termination...\n")
                            self.daemon_thread.terminate()
                            self.daemon_thread.wait(1000)
                            break
                        time.sleep(0.05)

                    # Brief delay to show completion state
                    self.app.processEvents()
                    time.sleep(0.5)

                    # Close dialog
                    diary_dialog.close()

                    # Clear callbacks
                    set_diary_update_callbacks()
                else:
                    # No dialog - simple wait
                    from jarvis.daemon import request_stop
                    request_stop()

                    if not self.daemon_thread.wait(shutdown_wait_timeout_sec * 1000):
                        self.log_signals.new_log.emit("⚠️ Daemon taking too long, forcing termination...\n")
                        self.daemon_thread.terminate()
                        self.daemon_thread.wait(1000)

                self.daemon_thread = None
            elif self.daemon_process:
                # For subprocess mode, show diary dialog too
                if show_diary_dialog:
                    diary_dialog = DiaryUpdateDialog()
                    diary_dialog.set_status("Saving diary (this may take a moment)...")
                    diary_dialog.show()
                    diary_dialog.raise_()
                    diary_dialog.activateWindow()
                    self.app.processEvents()

                    # Hide other windows
                    if hasattr(self, 'face_window') and self.face_window and self.face_window.isVisible():
                        self.face_window.hide()
                    if hasattr(self, 'log_viewer') and self.log_viewer.isVisible():
                        self.log_viewer.hide()

                # Send SIGINT for graceful shutdown
                if sys.platform == "win32":
                    self.daemon_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.daemon_process.send_signal(signal.SIGINT)

                # Wait for process to terminate while keeping UI responsive
                start_time = time.time()
                while self.daemon_process.poll() is None:
                    self.app.processEvents()
                    elapsed = time.time() - start_time
                    if elapsed > shutdown_wait_timeout_sec:
                        self.daemon_process.kill()
                        self.daemon_process.wait()
                        break
                    time.sleep(0.05)

                # Close diary dialog
                if diary_dialog:
                    diary_dialog.mark_completed(True)
                    self.app.processEvents()
                    time.sleep(0.5)
                    diary_dialog.close()

                self.daemon_process = None

            self.is_listening = False
            self.toggle_action.setText("▶️ Start Listening")
            self.status_action.setText("⚪ Status: Stopped")
            self.update_icon()

            self.tray_icon.showMessage(
                "Jarvis Stopped",
                "Voice assistant is no longer listening",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )

            self.log_signals.new_log.emit("⏸️ Jarvis daemon stopped\n")
            debug_log("daemon stopped from desktop app", "desktop")

        except Exception as e:
            debug_log(f"failed to stop daemon: {e}", "desktop")
            self.log_signals.new_log.emit(f"❌ Failed to stop: {str(e)}\n")
        finally:
            # Ensure dialog is closed
            if diary_dialog:
                diary_dialog.close()

    def check_daemon_status(self) -> None:
        """Check if the daemon process/thread is still running."""
        if self.is_bundled and self.daemon_thread:
            # Check if QThread is still running
            if self.daemon_thread.isFinished() and self.is_listening:
                # Thread has terminated
                self._on_daemon_finished()
                self.tray_icon.showMessage(
                    "Jarvis Stopped",
                    "Voice assistant process ended unexpectedly",
                    QSystemTrayIcon.MessageIcon.Warning,
                    3000
                )
                debug_log("daemon thread ended unexpectedly", "desktop")
        elif self.daemon_process:
            # Check if process is still alive
            poll = self.daemon_process.poll()
            if poll is not None:
                # Process has terminated
                self.daemon_process = None
                if self.is_listening:
                    self.is_listening = False
                    self.toggle_action.setText("▶️ Start Listening")
                    self.status_action.setText("⚪ Status: Stopped")
                    self.update_icon()

                    self.tray_icon.showMessage(
                        "Jarvis Stopped",
                        "Voice assistant process ended unexpectedly",
                        QSystemTrayIcon.MessageIcon.Warning,
                        3000
                    )

                    debug_log("daemon process ended unexpectedly", "desktop")

    def quit_app(self) -> None:
        """Quit the desktop app."""
        # Stop daemon if running
        if self.is_listening:
            self.stop_daemon()

        debug_log("desktop app shutting down", "desktop")
        self.tray_icon.hide()
        self.app.quit()

    def run(self) -> int:
        """Run the application event loop."""
        return self.app.exec()


def main() -> int:
    """Main entry point for the desktop app."""
    # Required for PyInstaller: must be called before any multiprocessing
    # Without this, bundled apps can spawn infinite copies of themselves
    import multiprocessing
    multiprocessing.freeze_support()

    # Single-instance check - must be done BEFORE any GUI is created
    # This prevents multiple tray icons and log windows from spawning
    if not acquire_single_instance_lock():
        print("⚠️ Another instance of Jarvis Desktop is already running. Exiting.")
        return 0

    # Set up crash logging for bundled apps
    crash_log_file = setup_crash_logging()

    print("Starting Jarvis Desktop App...", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"__file__: {__file__}", flush=True)
    print(flush=True)

    # Set up signal handlers for clean shutdown
    import signal
    tray_instance = None

    def signal_handler(signum, frame):
        """Handle termination signals."""
        print(f"Received signal {signum}, shutting down...", flush=True)
        if tray_instance:
            tray_instance.cleanup_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("Creating QApplication...", flush=True)
        from PyQt6.QtWidgets import QApplication
        print("QApplication imported successfully", flush=True)

        # Create QApplication first (needed for wizard)
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        # Check if setup wizard is needed
        print("Checking Ollama setup status...", flush=True)
        print("  Loading setup wizard module...", flush=True)
        try:
            from jarvis.setup_wizard import should_show_setup_wizard, SetupWizard
            print("  Setup wizard module loaded successfully", flush=True)
        except Exception as e:
            print(f"  ❌ Failed to load setup wizard: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        if should_show_setup_wizard():
            print("🔧 Setup required - launching setup wizard...", flush=True)
            wizard = SetupWizard()
            # Ensure wizard is visible and has focus (prevents window manager issues)
            wizard.show()
            wizard.raise_()
            wizard.activateWindow()
            result = wizard.exec()

            if result != wizard.DialogCode.Accepted:
                print("Setup wizard cancelled - exiting", flush=True)
                return 0

            print("✅ Setup wizard completed successfully", flush=True)
        else:
            print("✅ Ollama setup looks good", flush=True)

        print("Initializing JarvisSystemTray...", flush=True)
        tray_instance = JarvisSystemTray()
        print("JarvisSystemTray initialized successfully", flush=True)

        # Always auto-start listening (logs will be shown via start_daemon)
        print("🚀 Auto-starting Jarvis listener...", flush=True)
        tray_instance.start_daemon()

        if crash_log_file:
            # Show notification with log file location
            from PyQt6.QtWidgets import QSystemTrayIcon
            tray_instance.tray_icon.showMessage(
                "Jarvis Started",
                f"Crash logs available at:\n{crash_log_file}",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )

        print("Starting event loop...", flush=True)
        return tray_instance.run()
    except Exception as e:
        error_msg = f"desktop app fatal error: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        debug_log(error_msg, "desktop")

        # Try to show an error dialog if possible
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            if not QApplication.instance():
                app = QApplication(sys.argv)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Jarvis Desktop App Error")
            msg.setText("Failed to start Jarvis Desktop App")
            msg.setDetailedText(str(e) + "\n\n" + traceback.format_exc())
            if crash_log_file:
                msg.setInformativeText(f"Check log file at:\n{crash_log_file}")
            msg.exec()
        except:
            # Can't show dialog, error is already logged
            pass

        return 1


if __name__ == "__main__":
    # Required for PyInstaller to handle multiprocessing correctly
    # Without this, bundled apps spawn infinite copies of themselves
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())

