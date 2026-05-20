"""Setup wizard page: WhatsApp, Gmail, and personal page bookmarks."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from desktop_app.integration_setup import (
    GMAIL_SERVER,
    WHATSAPP_SERVER,
    apply_integrations_to_config,
    integration_prereq_hints,
    load_business_socials,
    load_personal_pages,
    pages_to_text,
    parse_pages_text,
    parse_socials_text,
    read_mcp_server_state,
    socials_placeholder_text,
    socials_to_text,
)


class IntegrationsPage(QWizardPage):
    """Guided comms + bookmark setup (replaces ad-hoc MCP hunting for daily tools)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")

        layout = QVBoxLayout()
        layout.setSpacing(14)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("🔗 Your integrations")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Connect messaging and the sites you use every day. "
            "Everything runs locally on your machine — Jarvis only talks to MCP servers you enable."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(12)

        hints = integration_prereq_hints()

        # --- WhatsApp ---
        wa_card = self._card(inner_layout)
        self._wa_check = QCheckBox("Enable WhatsApp (MCP)")
        self._wa_check.setChecked(self._mcp_enabled(WHATSAPP_SERVER))
        wa_card.layout().addWidget(self._wa_check)
        wa_hint = QLabel(hints["whatsapp"])
        wa_hint.setWordWrap(True)
        wa_hint.setTextFormat(Qt.TextFormat.RichText)
        wa_hint.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        wa_card.layout().addWidget(wa_hint)
        self._wa_setup_btn = QPushButton("Connect WhatsApp (show QR)")
        self._wa_setup_btn.clicked.connect(self._open_whatsapp_setup)
        wa_card.layout().addWidget(self._wa_setup_btn)
        wa_steps = QLabel(
            "Requires <a href='https://go.dev/dl/'>Go</a> on PATH. "
            "Jarvis downloads the bridge, shows the QR here, and saves MCP settings when you scan."
        )
        wa_steps.setOpenExternalLinks(True)
        wa_steps.setWordWrap(True)
        wa_steps.setTextFormat(Qt.TextFormat.RichText)
        wa_steps.setStyleSheet("font-size: 12px; color: #71717a;")
        wa_card.layout().addWidget(wa_steps)
        self._wa_status = QLabel(read_mcp_server_state(WHATSAPP_SERVER))
        self._wa_status.setWordWrap(True)
        self._wa_status.setStyleSheet("font-size: 12px; color: #94a3b8;")
        wa_card.layout().addWidget(self._wa_status)

        # --- Gmail ---
        gm_card = self._card(inner_layout)
        self._gmail_check = QCheckBox("Enable Gmail / Google Workspace (MCP)")
        self._gmail_check.setChecked(self._mcp_enabled(GMAIL_SERVER))
        gm_card.layout().addWidget(self._gmail_check)
        gm_hint = QLabel(hints["gmail"])
        gm_hint.setWordWrap(True)
        gm_hint.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        gm_card.layout().addWidget(gm_hint)
        self._gmail_id = QLineEdit()
        self._gmail_id.setPlaceholderText("Google OAuth Client ID")
        gm_card.layout().addWidget(self._gmail_id)
        self._gmail_secret = QLineEdit()
        self._gmail_secret.setPlaceholderText("Google OAuth Client Secret")
        self._gmail_secret.setEchoMode(QLineEdit.EchoMode.Password)
        gm_card.layout().addWidget(self._gmail_secret)
        self._load_gmail_fields()
        oauth_link = QLabel(
            '<a href="https://console.cloud.google.com/apis/credentials">'
            "Create credentials in Google Cloud Console</a> "
            "(OAuth 2.0 Client ID, Desktop app)."
        )
        oauth_link.setOpenExternalLinks(True)
        oauth_link.setWordWrap(True)
        oauth_link.setStyleSheet("font-size: 12px;")
        gm_card.layout().addWidget(oauth_link)
        self._gmail_status = QLabel(read_mcp_server_state(GMAIL_SERVER))
        self._gmail_status.setWordWrap(True)
        self._gmail_status.setStyleSheet("font-size: 12px; color: #94a3b8;")
        gm_card.layout().addWidget(self._gmail_status)

        # --- Business socials ---
        social_card = self._card(inner_layout)
        social_title = QLabel("Business social media")
        social_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        social_card.layout().addWidget(social_title)
        self._business_name = QLineEdit()
        self._business_name.setPlaceholderText("Business name (e.g. your café brand)")
        cfg = self._load_config()
        self._business_name.setText(str(cfg.get("business_name") or ""))
        social_card.layout().addWidget(self._business_name)
        social_help = QLabel(
            "One profile per line. Format: <b>Instagram | https://instagram.com/you</b> "
            "or paste the full URL. Shown on the Pulse dashboard as quick links."
        )
        social_help.setWordWrap(True)
        social_help.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        social_card.layout().addWidget(social_help)
        self._socials_edit = QTextEdit()
        self._socials_edit.setPlaceholderText(socials_placeholder_text())
        self._socials_edit.setMaximumHeight(100)
        self._load_socials_field()
        social_card.layout().addWidget(self._socials_edit)

        # --- Personal pages ---
        pages_card = self._card(inner_layout)
        pages_title = QLabel("Personal pages (bookmarks for Chrome / context)")
        pages_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        pages_card.layout().addWidget(pages_title)
        pages_help = QLabel(
            "One site per line. Format: <b>Label | https://example.com</b> or just the URL. "
            "Jarvis uses these in briefings and can open them via Chrome MCP."
        )
        pages_help.setWordWrap(True)
        pages_help.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        pages_card.layout().addWidget(pages_help)
        self._pages_edit = QTextEdit()
        self._pages_edit.setPlaceholderText(
            "Bank | https://www.mybank.lv\n"
            "Calendar | https://calendar.google.com\n"
            "https://mail.google.com"
        )
        self._pages_edit.setMaximumHeight(120)
        self._load_pages_field()
        pages_card.layout().addWidget(self._pages_edit)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        tip = QLabel(
            "💡  Chrome automation, Maps, and more stay on the next page (MCP Servers) "
            "or in Settings → MCP Servers."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);"
            "border-radius: 8px; padding: 12px 16px; color: #93c5fd; font-size: 13px;"
        )
        layout.addWidget(tip)

        self.setLayout(layout)

    @staticmethod
    def _card(parent_layout: QVBoxLayout) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)
        parent_layout.addWidget(card)
        return card

    @staticmethod
    def _load_config() -> dict:
        try:
            from jarvis.config import _load_json, default_config_path

            return _load_json(default_config_path()) or {}
        except Exception:
            return {}

    def _mcp_enabled(self, name: str) -> bool:
        mcps = self._load_config().get("mcps") or {}
        return isinstance(mcps, dict) and name in mcps

    def _load_gmail_fields(self) -> None:
        mcps = self._load_config().get("mcps") or {}
        entry = mcps.get(GMAIL_SERVER) if isinstance(mcps, dict) else None
        if isinstance(entry, dict):
            env = entry.get("env") or {}
            if isinstance(env, dict):
                self._gmail_id.setText(str(env.get("GOOGLE_CLIENT_ID") or ""))
                self._gmail_secret.setText(str(env.get("GOOGLE_CLIENT_SECRET") or ""))

    def _load_pages_field(self) -> None:
        pages = load_personal_pages(self._load_config())
        self._pages_edit.setPlainText(pages_to_text(pages))

    def _load_socials_field(self) -> None:
        socials = load_business_socials(self._load_config())
        self._socials_edit.setPlainText(socials_to_text(socials))

    def _open_whatsapp_setup(self) -> None:
        from desktop_app.whatsapp_setup_dialog import show_whatsapp_setup_safe

        if self._wa_check.isChecked() is False:
            self._wa_check.setChecked(True)
        show_whatsapp_setup_safe(self)

    def validatePage(self) -> bool:
        try:
            from jarvis.config import _load_json, _save_json, default_config_path

            config_path = default_config_path()
            config = _load_json(config_path) or {}
            pages = parse_pages_text(self._pages_edit.toPlainText())
            socials = parse_socials_text(self._socials_edit.toPlainText())
            apply_integrations_to_config(
                config,
                whatsapp_enabled=self._wa_check.isChecked(),
                gmail_enabled=self._gmail_check.isChecked(),
                gmail_client_id=self._gmail_id.text(),
                gmail_client_secret=self._gmail_secret.text(),
                personal_pages=pages,
                business_socials=socials,
                business_name=self._business_name.text(),
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(config_path, config)
        except Exception:
            pass
        return True

    def isComplete(self) -> bool:
        return True

    def nextId(self) -> int:
        wizard = self.wizard()
        from desktop_app.setup_wizard import SetupWizard

        if isinstance(wizard, SetupWizard):
            return wizard.mcp_page_id
        return super().nextId()
