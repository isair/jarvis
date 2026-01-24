"""
🚀 Jarvis Splash Screen

A stylish startup splash screen with animated loading indicator
that shows progress during application initialization.
"""

import math
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QFont
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal

from desktop_app.themes import COLORS


class AnimatedOrb(QWidget):
    """Animated pulsing orb with rotating arcs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(120, 120)

        # Animation state
        self._rotation = 0.0
        self._pulse_phase = 0.0
        self._glow_intensity = 0.5

        # Animation timer (60 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(16)

    def _animate(self):
        """Update animation state."""
        self._rotation += 2.0  # Degrees per frame
        if self._rotation >= 360:
            self._rotation -= 360

        self._pulse_phase += 0.08
        self._glow_intensity = 0.4 + 0.3 * math.sin(self._pulse_phase)

        self.update()

    def paintEvent(self, event):
        """Draw the animated orb."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x = self.width() / 2
        center_y = self.height() / 2

        # Colors from theme
        accent = QColor(COLORS["accent_primary"])
        accent_secondary = QColor(COLORS["accent_secondary"])
        bg = QColor(COLORS["bg_primary"])

        # Draw outer glow
        glow_radius = 50 + 5 * math.sin(self._pulse_phase)
        glow = QRadialGradient(center_x, center_y, glow_radius)
        glow_color = QColor(accent)
        glow_color.setAlphaF(self._glow_intensity * 0.3)
        glow.setColorAt(0, glow_color)
        glow_color.setAlphaF(0)
        glow.setColorAt(1, glow_color)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - glow_radius, center_y - glow_radius,
                                   glow_radius * 2, glow_radius * 2))

        # Draw core orb
        core_radius = 25 + 3 * math.sin(self._pulse_phase)
        core_gradient = QRadialGradient(center_x - 5, center_y - 5, core_radius * 1.5)
        core_gradient.setColorAt(0, accent_secondary)
        core_gradient.setColorAt(0.7, accent)
        darker = QColor(COLORS["accent_muted"])
        core_gradient.setColorAt(1, darker)
        painter.setBrush(QBrush(core_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - core_radius, center_y - core_radius,
                                   core_radius * 2, core_radius * 2))

        # Draw rotating arcs
        painter.setBrush(Qt.BrushStyle.NoBrush)
        arc_pen = QPen(accent_secondary)
        arc_pen.setWidth(3)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)

        arc_rect = QRectF(center_x - 40, center_y - 40, 80, 80)

        # Three arcs at different rotations
        for i, offset in enumerate([0, 120, 240]):
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(self._rotation + offset)
            painter.translate(-center_x, -center_y)

            # Vary alpha for each arc
            arc_color = QColor(accent_secondary)
            arc_color.setAlphaF(0.6 + 0.2 * math.sin(self._pulse_phase + i))
            arc_pen.setColor(arc_color)
            painter.setPen(arc_pen)

            painter.drawArc(arc_rect, 0 * 16, 60 * 16)  # 60 degree arc
            painter.restore()

    def stop(self):
        """Stop the animation."""
        self._timer.stop()


class SplashScreen(QWidget):
    """Splash screen shown during application startup."""

    # Signal emitted when splash should close
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Frameless, always on top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setFixedSize(300, 280)
        self._setup_ui()
        self._center_on_screen()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title
        title = QLabel("JARVIS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setWeight(QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {COLORS['accent_secondary']}; background: transparent;")
        layout.addWidget(title)

        # Animated orb
        self._orb = AnimatedOrb()
        orb_container = QWidget()
        orb_layout = QVBoxLayout(orb_container)
        orb_layout.setContentsMargins(0, 0, 0, 0)
        orb_layout.addWidget(self._orb, alignment=Qt.AlignmentFlag.AlignCenter)
        orb_container.setStyleSheet("background: transparent;")
        layout.addWidget(orb_container)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(11)
        self._status_label.setFont(status_font)
        self._status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")
        layout.addWidget(self._status_label)

    def _center_on_screen(self):
        """Center the splash screen on the primary display."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2 + screen_geometry.x()
            y = (screen_geometry.height() - self.height()) // 2 + screen_geometry.y()
            self.move(x, y)

    def paintEvent(self, event):
        """Draw the splash background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark background with rounded corners
        bg_color = QColor(COLORS["bg_primary"])
        bg_color.setAlphaF(0.95)
        painter.setBrush(QBrush(bg_color))

        border_color = QColor(COLORS["border"])
        painter.setPen(QPen(border_color, 1))

        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

    def set_status(self, status: str):
        """Update the status message."""
        self._status_label.setText(status)
        # Process events to ensure the UI updates
        QApplication.processEvents()

    def close_splash(self):
        """Close the splash screen gracefully."""
        self._orb.stop()
        self.finished.emit()
        self.close()
