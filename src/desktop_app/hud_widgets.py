"""Reusable HUD-style widgets for the Jarvis desktop interface."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop_app.themes import COLORS


class HudPanel(QFrame):
    """A titled glass-panel container used across Jarvis desktop windows."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("hudPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("hudPanelTitle")
        header.addWidget(self.title_label)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("hudPanelSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        header.addWidget(self.subtitle_label)

        layout.addLayout(header)

        self.body = QWidget()
        self.body.setObjectName("hudPanelBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(10)
        layout.addWidget(self.body)

    def addWidget(self, widget: QWidget) -> None:
        """Add a widget to the panel body."""
        self.body_layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        """Add a layout to the panel body."""
        self.body_layout.addLayout(layout)


class StatusPill(QLabel):
    """Compact semantic status chip for readiness and runtime telemetry."""

    _VALID_STATUSES = {"neutral", "success", "warning", "error", "active"}

    def __init__(
        self,
        text: str,
        status: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status(text, status)

    def set_status(self, text: str, status: str = "neutral") -> None:
        """Update chip text and semantic style in one call."""
        if status not in self._VALID_STATUSES:
            status = "neutral"
        self.setText(text)
        self.setObjectName(f"statusPill-{status}")
        self.style().unpolish(self)
        self.style().polish(self)


class MetricTile(QFrame):
    """Dashboard tile that presents one key value with a short label."""

    def __init__(
        self,
        label: str,
        value: str,
        detail: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricTile")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.label_label = QLabel(label)
        self.label_label.setObjectName("metricTileLabel")
        layout.addWidget(self.label_label)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricTileValue")
        layout.addWidget(self.value_label)

        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("metricTileDetail")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(bool(detail))
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str = "") -> None:
        """Update the tile value and optional supporting detail."""
        self.value_label.setText(value)
        self.detail_label.setText(detail)
        self.detail_label.setVisible(bool(detail))


class TelemetryRow(QFrame):
    """One dense HUD row containing a label, value, and semantic status."""

    def __init__(
        self,
        label: str,
        value: str,
        status: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("telemetryRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(12)

        self.label_label = QLabel(label)
        self.label_label.setObjectName("telemetryLabel")
        layout.addWidget(self.label_label)

        layout.addStretch()

        self.value_label = QLabel(value)
        self.value_label.setObjectName("telemetryValue")
        layout.addWidget(self.value_label)

        self.status_pill = StatusPill(status.upper(), status)
        layout.addWidget(self.status_pill)

    def set_value(self, value: str, status: str = "neutral") -> None:
        """Update the displayed value and status pill."""
        self.value_label.setText(value)
        self.status_pill.set_status(status.upper(), status)


class HudHeader(QWidget):
    """Branded page header with title, subtitle, and optional status pill."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        *,
        status_text: str = "",
        status: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("hudHeader")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("hudHeaderTitle")
        top.addWidget(self.title_label, stretch=1)

        self.status_pill = StatusPill(status_text, status) if status_text else None
        if self.status_pill is not None:
            top.addWidget(self.status_pill)

        layout.addLayout(top)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("hudHeaderSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)

    def set_status(self, text: str, status: str = "neutral") -> None:
        if self.status_pill is None:
            return
        self.status_pill.set_status(text, status)


class SectionDivider(QFrame):
    """Horizontal rule with optional centred caption between settings sections."""

    def __init__(self, caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionDivider")
        self.setFixedHeight(28 if caption else 12)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)

        if caption:
            line = QFrame()
            line.setObjectName("sectionDividerLine")
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFixedHeight(1)
            layout.addWidget(line, stretch=1)

            cap = QLabel(caption.upper())
            cap.setObjectName("sectionDividerCaption")
            layout.addWidget(cap)

            line2 = QFrame()
            line2.setObjectName("sectionDividerLine")
            line2.setFrameShape(QFrame.Shape.HLine)
            line2.setFixedHeight(1)
            layout.addWidget(line2, stretch=1)
        else:
            line = QFrame()
            line.setObjectName("sectionDividerLine")
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFixedHeight(1)
            layout.addWidget(line)


class HudGridBackdrop(QWidget):
    """Subtle animated grid + scanline backdrop for splash and presence surfaces."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._phase = 0.0

    def advance_phase(self, delta: float = 0.04) -> None:
        self._phase += delta
        if self._phase > 1000.0:
            self._phase = 0.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w, h = self.width(), self.height()
        grid = QColor(COLORS["cyan_primary"])
        grid.setAlphaF(0.06)
        painter.setPen(QPen(grid, 1))

        step = 24
        for x in range(0, w, step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            painter.drawLine(0, y, w, y)

        scan_y = int((math.sin(self._phase) * 0.5 + 0.5) * h)
        scan = QColor(COLORS["cyan_primary"])
        scan.setAlphaF(0.08)
        painter.fillRect(QRectF(0, scan_y, w, 2), scan)
