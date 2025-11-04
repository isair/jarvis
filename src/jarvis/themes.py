"""
🎨 Jarvis UI Themes

Shared stylesheets for Qt interfaces, matching the Memory Viewer's
deep space theme with amber accents.
"""

# Color palette
COLORS = {
    "bg_primary": "#0a0b0f",
    "bg_secondary": "#12141a",
    "bg_tertiary": "#1a1d26",
    "bg_card": "#161920",
    "bg_hover": "#1e222c",
    
    "accent_primary": "#f59e0b",
    "accent_secondary": "#fbbf24",
    "accent_glow": "rgba(245, 158, 11, 0.15)",
    "accent_muted": "#92400e",
    
    "text_primary": "#f4f4f5",
    "text_secondary": "#a1a1aa",
    "text_muted": "#71717a",
    
    "border": "#27272a",
    "border_glow": "rgba(245, 158, 11, 0.3)",
    
    "success": "#22c55e",
    "success_light": "#4ade80",
    "warning": "#f59e0b",
    "warning_light": "#fbbf24",
    "error": "#ef4444",
    "error_light": "#f87171",
}


# Comprehensive Qt stylesheet matching the Memory Viewer's design
JARVIS_THEME_STYLESHEET = """
    QMainWindow, QDialog, QWizard, QWizardPage {
        background-color: #0a0b0f;
    }
    
    QWidget {
        background-color: #0a0b0f;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    QLabel {
        color: #f4f4f5;
        background: transparent;
    }
    
    QLabel#title {
        font-size: 18px;
        font-weight: 600;
        color: #f4f4f5;
    }
    
    QLabel#subtitle {
        font-size: 12px;
        color: #71717a;
    }
    
    QLabel#section_title {
        font-size: 16px;
        font-weight: bold;
        color: #fbbf24;
    }
    
    QTextEdit, QPlainTextEdit {
        background-color: #12141a;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 12px;
        selection-background-color: rgba(245, 158, 11, 0.3);
        selection-color: #fbbf24;
    }
    
    QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #f59e0b;
    }
    
    QLineEdit {
        background-color: #12141a;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 8px 12px;
        selection-background-color: rgba(245, 158, 11, 0.3);
    }
    
    QLineEdit:focus {
        border-color: #f59e0b;
    }
    
    QLineEdit::placeholder {
        color: #71717a;
    }
    
    QPushButton {
        background-color: #1a1d26;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    QPushButton:hover {
        background-color: #1e222c;
        border-color: #f59e0b;
        color: #fbbf24;
    }
    
    QPushButton:pressed {
        background-color: rgba(245, 158, 11, 0.15);
    }
    
    QPushButton:disabled {
        background-color: #12141a;
        color: #71717a;
        border-color: #1a1d26;
    }
    
    QPushButton#primary {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #f59e0b, stop:1 #d97706);
        color: #0a0b0f;
        border: none;
        font-weight: 600;
    }
    
    QPushButton#primary:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #fbbf24, stop:1 #f59e0b);
    }
    
    QPushButton#primary:disabled {
        background: #27272a;
        color: #71717a;
    }
    
    QPushButton#danger {
        background-color: #1a1d26;
        border-color: #ef4444;
        color: #ef4444;
    }
    
    QPushButton#danger:hover {
        background-color: rgba(239, 68, 68, 0.15);
        border-color: #f87171;
        color: #f87171;
    }
    
    QComboBox {
        background-color: #12141a;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 8px 12px;
        min-width: 120px;
    }
    
    QComboBox:hover {
        border-color: #f59e0b;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #71717a;
        margin-right: 8px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #161920;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        selection-background-color: rgba(245, 158, 11, 0.15);
        selection-color: #fbbf24;
    }
    
    QCheckBox {
        color: #f4f4f5;
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #27272a;
        border-radius: 4px;
        background-color: #12141a;
    }
    
    QCheckBox::indicator:hover {
        border-color: #f59e0b;
    }
    
    QCheckBox::indicator:checked {
        background-color: #f59e0b;
        border-color: #f59e0b;
    }
    
    QRadioButton {
        color: #f4f4f5;
        spacing: 8px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #27272a;
        border-radius: 9px;
        background-color: #12141a;
    }
    
    QRadioButton::indicator:hover {
        border-color: #f59e0b;
    }
    
    QRadioButton::indicator:checked {
        background-color: #f59e0b;
        border-color: #f59e0b;
    }
    
    QProgressBar {
        background-color: #12141a;
        border: 1px solid #27272a;
        border-radius: 6px;
        height: 8px;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #f59e0b, stop:1 #fbbf24);
        border-radius: 5px;
    }
    
    QScrollArea {
        background: transparent;
        border: none;
    }
    
    QScrollBar:vertical {
        background-color: #12141a;
        width: 10px;
        border-radius: 5px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #27272a;
        border-radius: 5px;
        min-height: 30px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #f59e0b;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    
    QScrollBar:horizontal {
        background-color: #12141a;
        height: 10px;
        border-radius: 5px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #27272a;
        border-radius: 5px;
        min-width: 30px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #f59e0b;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    
    QGroupBox {
        background-color: #161920;
        border: 1px solid #27272a;
        border-radius: 12px;
        margin-top: 12px;
        padding: 16px;
        padding-top: 24px;
        font-weight: 500;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
        color: #a1a1aa;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    QTabWidget::pane {
        background-color: #161920;
        border: 1px solid #27272a;
        border-radius: 12px;
        top: -1px;
    }
    
    QTabBar::tab {
        background-color: #12141a;
        color: #a1a1aa;
        border: 1px solid #27272a;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 10px 20px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: #161920;
        color: #fbbf24;
        border-color: #27272a;
        border-bottom-color: #161920;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #1a1d26;
        color: #f4f4f5;
    }
    
    QSpinBox, QDoubleSpinBox {
        background-color: #12141a;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 8px 12px;
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #f59e0b;
    }
    
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        background-color: #1a1d26;
        border: none;
        width: 20px;
    }
    
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #f59e0b;
    }
    
    QListWidget {
        background-color: #12141a;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 8px;
    }
    
    QListWidget::item {
        padding: 8px 12px;
        border-radius: 6px;
    }
    
    QListWidget::item:selected {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
    }
    
    QListWidget::item:hover:!selected {
        background-color: #1e222c;
    }
    
    QMessageBox {
        background-color: #0a0b0f;
    }
    
    QMessageBox QLabel {
        color: #f4f4f5;
    }
    
    QToolTip {
        background-color: #161920;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 6px 10px;
    }
    
    QMenu {
        background-color: #161920;
        color: #f4f4f5;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 4px;
    }
    
    QMenu::item {
        padding: 8px 24px;
        border-radius: 4px;
    }
    
    QMenu::item:selected {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #27272a;
        margin: 4px 8px;
    }
    
    /* Wizard-specific styles */
    QWizard QPushButton {
        min-width: 100px;
    }
    
    QWizard QLabel#qt_watermark_label {
        background: transparent;
    }
    
    /* Card-style container */
    QFrame#card {
        background-color: #161920;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 16px;
    }
"""


def apply_theme(widget) -> None:
    """Apply the Jarvis theme to a Qt widget."""
    widget.setStyleSheet(JARVIS_THEME_STYLESHEET)

