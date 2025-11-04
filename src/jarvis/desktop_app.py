"""
Jarvis Desktop App - System Tray Application

A cross-platform system tray app for controlling the Jarvis voice assistant.
Supports Windows, Ubuntu (Linux), and macOS.
"""

from __future__ import annotations
import sys
import os
import subprocess
import signal
import psutil
import threading
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PyQt6.QtGui import QIcon, QAction, QFont, QTextCursor
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject

from .debug import debug_log
from .config import _default_config_path, _default_db_path


class LogSignals(QObject):
    """Signals for thread-safe log updates."""
    new_log = pyqtSignal(str)


class LogViewerWindow(QMainWindow):
    """Window for viewing Jarvis logs in real-time."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📝 Jarvis Logs")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create text display for logs
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Menlo", 11) if sys.platform == "darwin" else QFont("Consolas", 10))
        layout.addWidget(self.log_display)

        # Set dark theme for logs
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                padding: 10px;
            }
        """)

        # Initial message
        self.append_log("📝 Jarvis Logs\n" + "="*60 + "\n")

    def append_log(self, text: str) -> None:
        """Append text to the log display."""
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)
        self.log_display.insertPlainText(text)
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.log_display.clear()
        self.append_log("📝 Jarvis Logs\n" + "="*60 + "\n")


class JarvisSystemTray:
    """System tray application for Jarvis voice assistant."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        # Initialize state
        self.daemon_process: Optional[subprocess.Popen] = None
        self.is_listening = False

        # Kill any orphaned Jarvis processes from previous sessions
        self.cleanup_orphaned_processes()

        # Create log viewer window (hidden by default)
        self.log_viewer = LogViewerWindow()
        self.log_signals = LogSignals()
        self.log_signals.new_log.connect(self.log_viewer.append_log)

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
        if self.daemon_process:
            try:
                self.daemon_process.terminate()
                try:
                    self.daemon_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.daemon_process.kill()
                    self.daemon_process.wait()
            except Exception as e:
                debug_log(f"error during exit cleanup: {e}", "desktop")

    def create_menu(self) -> None:
        """Create the system tray context menu."""
        menu = QMenu()

        # Toggle listening action
        self.toggle_action = QAction("▶️ Start Listening")
        self.toggle_action.triggered.connect(self.toggle_listening)
        menu.addAction(self.toggle_action)

        menu.addSeparator()

        # View logs action
        self.logs_action = QAction("📝 View Logs")
        self.logs_action.triggered.connect(self.toggle_log_viewer)
        menu.addAction(self.logs_action)

        menu.addSeparator()

        # Open directories actions
        open_config_action = QAction("📁 Open Config Directory")
        open_config_action.triggered.connect(self.open_config_directory)
        menu.addAction(open_config_action)

        open_data_action = QAction("💾 Open Data Directory")
        open_data_action.triggered.connect(self.open_data_directory)
        menu.addAction(open_data_action)

        menu.addSeparator()

        # Status action (non-clickable)
        self.status_action = QAction("⚪ Status: Stopped")
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)

        menu.addSeparator()

        # Quit action
        quit_action = QAction("🚪 Quit")
        quit_action.triggered.connect(self.quit_app)
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)

    def toggle_log_viewer(self) -> None:
        """Toggle the log viewer window visibility."""
        if self.log_viewer.isVisible():
            self.log_viewer.hide()
        else:
            self.log_viewer.show()
            self.log_viewer.raise_()
            self.log_viewer.activateWindow()

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
            # Find the Python executable and jarvis module
            python_exe = sys.executable

            # Start daemon as a subprocess
            self.daemon_process = subprocess.Popen(
                [python_exe, "-m", "jarvis.main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start log reader thread
            log_thread = threading.Thread(
                target=self._read_daemon_logs,
                daemon=True
            )
            log_thread.start()
            self.log_reader_threads.append(log_thread)

            self.is_listening = True
            self.toggle_action.setText("⏸️ Stop Listening")
            self.status_action.setText("🟢 Status: Listening")
            self.update_icon()

            self.tray_icon.showMessage(
                "Jarvis Started",
                "Voice assistant is now listening",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )

            self.log_signals.new_log.emit("🚀 Jarvis daemon started\n")
            debug_log("daemon started from desktop app", "desktop")

        except Exception as e:
            debug_log(f"failed to start daemon: {e}", "desktop")
            self.log_signals.new_log.emit(f"❌ Failed to start: {str(e)}\n")
            self.tray_icon.showMessage(
                "Error Starting Jarvis",
                f"Failed to start: {str(e)}",
                QSystemTrayIcon.MessageIcon.Critical,
                3000
            )

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

    def stop_daemon(self) -> None:
        """Stop the Jarvis daemon."""
        try:
            if self.daemon_process:
                # Send SIGINT for graceful shutdown
                if sys.platform == "win32":
                    self.daemon_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.daemon_process.send_signal(signal.SIGINT)

                # Wait for process to terminate (with timeout)
                try:
                    self.daemon_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    self.daemon_process.kill()
                    self.daemon_process.wait()

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

    def check_daemon_status(self) -> None:
        """Check if the daemon process is still running."""
        if self.daemon_process:
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
    # Set up signal handlers for clean shutdown
    import signal
    tray_instance = None

    def signal_handler(signum, frame):
        """Handle termination signals."""
        if tray_instance:
            tray_instance.cleanup_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        tray_instance = JarvisSystemTray()
        return tray_instance.run()
    except Exception as e:
        debug_log(f"desktop app fatal error: {e}", "desktop")
        return 1


if __name__ == "__main__":
    sys.exit(main())

