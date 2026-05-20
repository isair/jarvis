"""In-app WhatsApp pairing: QR display + automatic MCP config."""

from __future__ import annotations

from PyQt6.QtCore import QEventLoop, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from jarvis.integrations.whatsapp.bridge import (
    WhatsAppBridgeController,
    render_qr_png,
    save_whatsapp_mcp_to_config,
)


def _top_level_parent(parent) -> object | None:
    """Use the QWizard window as parent — not QWizardPage (avoids wizard closing on exec)."""
    if parent is None:
        return None
    if isinstance(parent, QWizardPage):
        wiz = parent.wizard()
        return wiz if wiz is not None else parent
    if isinstance(parent, QWizard):
        return parent
    return parent


class _BridgeWorker(QThread):
    qr_ready = pyqtSignal(str)
    authenticated = pyqtSignal()
    log_line = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, controller: WhatsAppBridgeController, parent=None):
        super().__init__(parent)
        self._controller = controller

    def run(self) -> None:
        self._controller.start(
            on_qr=lambda code: self.qr_ready.emit(code),
            on_authenticated=lambda: self.authenticated.emit(),
            on_log=lambda msg: self.log_line.emit(msg),
            on_error=lambda msg: self.failed.emit(msg),
        )


class WhatsAppSetupDialog(QDialog):
    """Show QR code and connect Jarvis to the user's WhatsApp account."""

    def __init__(self, parent=None):
        super().__init__(_top_level_parent(parent))
        self.setWindowTitle("Connect WhatsApp")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._controller = WhatsAppBridgeController()
        self._worker: _BridgeWorker | None = None
        self._connected = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("📱 WhatsApp setup")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fbbf24;")
        layout.addWidget(title)

        self._status = QLabel(
            "Press <b>Start setup</b> to show a QR code. "
            "Scan it in WhatsApp → Settings → Linked devices → Link a device."
        )
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._status)

        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setMinimumSize(280, 280)
        self._qr_label.setStyleSheet(
            "background: #fff; border-radius: 12px; padding: 12px;"
        )
        self._qr_label.hide()
        layout.addWidget(self._qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setStyleSheet(
            "background: #18181b; color: #a1a1aa; font-size: 12px; border-radius: 8px;"
        )
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start setup")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._start_btn.setEnabled(False)
        self._status.setText("Preparing WhatsApp bridge…")
        self._log.clear()
        self._worker = _BridgeWorker(self._controller, self)
        self._worker.qr_ready.connect(self._on_qr)
        self._worker.authenticated.connect(self._on_authenticated)
        self._worker.log_line.connect(self._append_log)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(lambda: self._start_btn.setEnabled(True))
        self._worker.start()

    def _on_qr(self, code: str) -> None:
        try:
            png = render_qr_png(code)
            pix = QPixmap()
            pix.loadFromData(png)
            self._qr_label.setPixmap(
                pix.scaled(
                    260,
                    260,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._qr_label.show()
            self._status.setText(
                "Scan this QR with your phone (WhatsApp → Linked devices → Link a device)."
            )
        except Exception as exc:
            self._on_failed(f"Could not render QR: {exc}")

    def _on_authenticated(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._status.setText("Connected. Saving Jarvis configuration…")
        try:
            save_whatsapp_mcp_to_config()
            self._status.setText(
                "✅ WhatsApp is linked. Use <b>Next</b> in the wizard or restart listening "
                "so MCP tools refresh. Keep Jarvis running — the bridge stays in the background."
            )
            self._append_log("Saved mcps.whatsapp to config.json")
        except Exception as exc:
            self._on_failed(f"Connected but config save failed: {exc}")

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"❌ {msg}")
        self._append_log(msg)
        self._start_btn.setEnabled(True)

    def closeEvent(self, event) -> None:
        # Keep bridge running after dialog closes if pairing succeeded
        super().closeEvent(event)


def show_whatsapp_setup(parent=None) -> int:
    """Show pairing UI without closing an open QWizard (nested exec() bug on Windows)."""
    top = _top_level_parent(parent)
    dlg = WhatsAppSetupDialog(top)
    app = QApplication.instance()

    # Nested exec() inside QWizard can close the wizard; use a local event loop instead.
    if top is not None and isinstance(top, QWizard):
        loop = QEventLoop()
        result_holder: list[int] = [QDialog.DialogCode.Rejected]

        def _done(code: int) -> None:
            result_holder[0] = code
            loop.quit()

        dlg.finished.connect(_done)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        loop.exec()
        return result_holder[0]

    if app is not None:
        return dlg.exec()
    dlg.show()
    return QDialog.DialogCode.Accepted


def show_whatsapp_setup_safe(parent=None) -> None:
    """Open dialog; show QMessageBox instead of crashing the app on failure."""
    try:
        show_whatsapp_setup(parent)
    except Exception as exc:
        QMessageBox.critical(
            _top_level_parent(parent) or None,
            "WhatsApp setup failed",
            str(exc),
        )
