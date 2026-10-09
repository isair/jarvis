"""A quiet, translucent amber face and shared desktop animation state."""

from __future__ import annotations
import math
import random
import threading
import time as _time
from typing import Optional
from enum import Enum
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication, QMenu
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QPainterPathStroker, QLinearGradient, QRadialGradient, QBrush, QRegion
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QObject

from jarvis.debug import debug_log
from desktop_app.themes import COLORS


class Expression(Enum):
    """Available face expressions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    THINKING = "thinking"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    EXCITED = "excited"
    CONCERNED = "concerned"


class JarvisState(Enum):
    """Overall Jarvis state for face animation."""
    ASLEEP = "asleep"          # Daemon not started yet
    IDLE = "idle"              # Awake and ready, waiting for wake word
    LISTENING = "listening"    # Actively listening (collecting or hot window)
    THINKING = "thinking"      # Processing query
    SPEAKING = "speaking"      # Speaking response
    DICTATING = "dictating"    # Hold-to-dictate recording active
    DICTATION_PROCESSING = "dictation_processing"  # Transcribing & pasting captured dictation


# Global Jarvis state - allows daemon to signal overall state to face widget
# Uses a file-based approach to work across processes (dev mode runs daemon as subprocess)
import tempfile
import os

def _get_jarvis_state_file() -> str:
    """Get the path to the Jarvis state file."""
    return os.path.join(tempfile.gettempdir(), "jarvis_state")


class JarvisStateManager(QObject):
    """Global singleton for Jarvis state management.

    Uses a file-based approach to communicate across processes:
    - In dev mode, daemon runs as subprocess (different process)
    - In bundled mode, daemon runs as QThread (same process)
    - File-based state works in both cases

    Note: Singleton pattern uses module-level instance instead of __new__
    because PyQt6 QObject doesn't support __new__ override properly.
    """
    state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._state = JarvisState.ASLEEP  # Start asleep
        self._state_lock = threading.Lock()
        self._state_file = _get_jarvis_state_file()
        # Always start fresh in ASLEEP state on app launch
        # (state file is for cross-process communication during a session,
        # not for persisting state across app restarts)
        self._write_state(JarvisState.ASLEEP)

    @property
    def state(self) -> JarvisState:
        """Read current state (checks file for cross-process communication)."""
        # First check file (for cross-process), then fall back to memory
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    content = f.read().strip()
                    return JarvisState(content)
        except (ValueError, OSError):
            # Invalid content or read error - fall back to in-memory state
            pass

        with self._state_lock:
            return self._state

    def _write_state(self, state: JarvisState) -> None:
        """Write state to file for cross-process communication."""
        try:
            with open(self._state_file, 'w') as f:
                f.write(state.value)
        except OSError:
            # File write failed - state won't be shared across processes
            pass

    def set_state(self, state: JarvisState) -> None:
        """Set the Jarvis state (thread-safe, cross-process)."""
        with self._state_lock:
            self._state = state

        # Write to file for cross-process communication
        self._write_state(state)

        # Emit signal for same-process listeners
        try:
            self.state_changed.emit(state.value)
        except RuntimeError:
            # If Qt event loop isn't running, just update the flag
            pass


# Module-level singleton instance
_jarvis_state_instance: Optional[JarvisStateManager] = None
_jarvis_state_lock = threading.Lock()


def get_jarvis_state() -> JarvisStateManager:
    """Get the global Jarvis state singleton."""
    global _jarvis_state_instance
    with _jarvis_state_lock:
        if _jarvis_state_instance is None:
            _jarvis_state_instance = JarvisStateManager()
        return _jarvis_state_instance


class FaceWidget(QWidget):
    """An angular light-beam mask with quiet, time-based expressions.

    Only the connected beams, junctions, eyes and mouth paint over the desktop.
    Motion stays within that silhouette, apart from a close-fitting interaction echo.
    """

    state_observed = pyqtSignal(str)

    DESIGN_WIDTH = 220
    DESIGN_HEIGHT = 280
    FACE_WIDTH = 142.0
    FACE_HEIGHT = 177.0
    PRIMARY_COLOUR = QColor(COLORS["accent_secondary"])
    SECONDARY_COLOUR = QColor(COLORS["accent_primary"])
    INK_COLOUR = QColor(COLORS["bg_primary"])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 208)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._state_manager = get_jarvis_state()
        self._jarvis_state = JarvisState.ASLEEP
        self._expression = Expression.NEUTRAL
        self._activation_level = 0.0
        self._listening = 0.0
        self._thinking = 0.0
        self._speaking = 0.0
        self._dictating = 0.0
        self._started_at = _time.monotonic()
        self._last_frame_at = self._started_at
        self._state_entered_at = self._started_at
        self._elapsed = 0.0
        self._state_elapsed = 0.0
        self._listening_started_at: Optional[float] = None
        self._breathing_scale = 1.0
        self._blink_factor = 0.0
        self._blink_started_at: Optional[float] = None
        self._next_blink_at = self._started_at + random.uniform(4.5, 7.5)
        self._glance_started_at: Optional[float] = None
        self._next_glance_at = self._started_at + random.uniform(9.0, 15.0)
        self._glance_target = (0.0, 0.0)
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(33)
        self._animation_timer.timeout.connect(self._animate)

    def showEvent(self, event):
        super().showEvent(event)
        # Observe the latest state immediately, without replaying hidden time.
        now = _time.monotonic()
        self._started_at += now - self._last_frame_at
        self._last_frame_at = now
        self._blink_started_at = None
        self._next_blink_at = now + random.uniform(4.5, 7.5)
        self._glance_started_at = None
        self._next_glance_at = now + random.uniform(9.0, 15.0)
        self._animate()
        self._animation_timer.start()

    def hideEvent(self, event):
        self._animation_timer.stop()
        super().hideEvent(event)

    def set_expression(self, expression: Expression):
        """Set the face expression without changing the assistant's state."""
        if expression != self._expression:
            self._expression = expression
            self.update()

    @staticmethod
    def _approach(value: float, target: float, elapsed: float, duration: float) -> float:
        """Ease a visual property at the same rate on every frame cadence."""
        result = target + (value - target) * math.exp(-elapsed / duration)
        return target if abs(result - target) < 0.0001 else result

    def _animate(self):
        """Observe state once, then advance a bounded amount of visual work."""
        now = _time.monotonic()
        elapsed = max(0.0, now - self._last_frame_at)
        self._last_frame_at = now
        self._elapsed = max(0.0, now - self._started_at)
        previous = self._jarvis_state
        self._jarvis_state = self._state_manager.state
        if self._jarvis_state != previous:
            self._state_entered_at = now
            if self._jarvis_state == JarvisState.LISTENING:
                self._listening_started_at = now
            self.state_observed.emit(self._jarvis_state.value)
            debug_log(f"👤 Face state: {self._jarvis_state.value}", "face")
        self._state_elapsed = max(0.0, now - self._state_entered_at)
        state = self._jarvis_state
        awake = state != JarvisState.ASLEEP
        self._activation_level = self._approach(self._activation_level, float(awake), elapsed, 0.32)
        for name, target in (
            ("_listening", state == JarvisState.LISTENING),
            ("_thinking", state in (JarvisState.THINKING, JarvisState.DICTATION_PROCESSING)),
            ("_speaking", state == JarvisState.SPEAKING),
            ("_dictating", state in (JarvisState.DICTATING, JarvisState.DICTATION_PROCESSING)),
        ):
            setattr(self, name, self._approach(getattr(self, name), float(target), elapsed, 0.18))
        self._breathing_scale = 1.0 + math.sin(self._elapsed * math.tau / 7.5) * 0.006 * self._activation_level
        self._update_blink(now, awake)
        self._update_glance(now, state == JarvisState.IDLE)
        # Sleeping faces still observe daemon state promptly; hidden ones do no work.
        self._animation_timer.setInterval(
            250 if not awake and self._activation_level == 0.0
            else 50 if state == JarvisState.IDLE else 33
        )
        self.update()

    def _update_blink(self, now: float, awake: bool):
        self._blink_factor = 0.0
        if not awake:
            self._blink_started_at = None
            self._next_blink_at = now + random.uniform(4.5, 7.5)
            return
        if self._blink_started_at is None and now >= self._next_blink_at:
            self._blink_started_at = now
        if self._blink_started_at is not None:
            progress = (now - self._blink_started_at) / 0.28
            if progress < 1.0:
                self._blink_factor = math.sin(progress * math.pi) ** 2
            else:
                self._blink_started_at = None
                self._next_blink_at = now + random.uniform(4.5, 7.5)

    def _update_glance(self, now: float, idle: bool):
        self._gaze_x = self._gaze_y = 0.0
        if not idle:
            self._glance_started_at = None
            self._next_glance_at = now + random.uniform(9.0, 15.0)
            return
        if self._glance_started_at is None and now >= self._next_glance_at:
            self._glance_started_at = now
            self._glance_target = (random.uniform(-2.5, 2.5), random.uniform(-1.0, 1.0))
        if self._glance_started_at is not None:
            progress = (now - self._glance_started_at) / 3.2
            if progress < 1.0:
                envelope = math.sin(progress * math.pi) ** 2
                self._gaze_x = self._glance_target[0] * envelope
                self._gaze_y = self._glance_target[1] * envelope
            else:
                self._glance_started_at = None
                self._next_glance_at = now + random.uniform(9.0, 15.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min(self.width() / self.DESIGN_WIDTH, self.height() / self.DESIGN_HEIGHT)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(scale * self._breathing_scale, scale * self._breathing_scale)
        cx, cy, width, height = 0.0, 0.0, self.FACE_WIDTH, self.FACE_HEIGHT
        self._draw_state_contour(painter, cx, cy, width, height)
        contour = self._face_contour(cx, cy, width, height)
        self._stroke(painter, contour, 1.65, 0.46 + 0.4 * self._activation_level)
        self._draw_mask_junctions(painter, contour)
        self._draw_accent_lines(painter, cx, cy, width, height)
        self._draw_beam_flow(painter, contour)
        eye_y = cy - height * 0.12
        for is_left in (True, False):
            eye_x = cx + (-1 if is_left else 1) * width * 0.24
            self._draw_eye(painter, eye_x, eye_y, width * 0.14, self._blink_factor, is_left)
        self._draw_mouth(painter, cx, cy, width, height)
        painter.end()

    @classmethod
    def input_region(cls, width: int, height: int) -> QRegion:
        """Leave empty margins clickable while retaining the face's drag area."""
        scale = min(width / cls.DESIGN_WIDTH, height / cls.DESIGN_HEIGHT)
        contour = cls._face_contour(width / 2, height / 2,
                                    cls.FACE_WIDTH * scale, cls.FACE_HEIGHT * scale)
        padding = QPainterPathStroker()
        padding.setWidth(18 * scale)
        boundary = contour.united(padding.createStroke(contour))
        return QRegion(boundary.toFillPolygon().toPolygon())

    @staticmethod
    def _face_contour(cx: float, cy: float, width: float, height: float) -> QPainterPath:
        """The angular crown, temples and tapered jaw of the light-beam mask."""
        hw, hh = width / 2, height / 2
        vertices = [QPointF(cx + x * hw, cy + y * hh) for x, y in (
            (0, -1), (0.5, -0.85), (0.8, -0.5), (1, -0.1),
            (0.9, 0.3), (0.6, 0.7), (0.3, 0.9), (0, 1),
            (-0.3, 0.9), (-0.6, 0.7), (-0.9, 0.3), (-1, -0.1),
            (-0.8, -0.5), (-0.5, -0.85),
        )]
        path = QPainterPath(vertices[0])
        for vertex in vertices[1:]:
            path.lineTo(vertex)
        path.closeSubpath()
        return path

    def _draw_mask_junctions(self, painter: QPainter, contour: QPainterPath):
        """Small fixed lights join the beams without wandering beyond the mask."""
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(0.46 + 0.4 * self._activation_level)
        for index in range(contour.elementCount() - 1):
            vertex = contour.elementAt(index)
            glow = QRadialGradient(vertex.x, vertex.y, 4)
            light = QColor(self.PRIMARY_COLOUR)
            light.setAlpha(90)
            glow.setColorAt(0, light)
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(vertex.x, vertex.y), 4, 4)
            painter.setBrush(self.PRIMARY_COLOUR.lighter(150))
            painter.drawEllipse(QPointF(vertex.x, vertex.y), 1.8, 1.8)
        painter.restore()

    def _draw_accent_lines(self, painter: QPainter, cx: float, cy: float,
                           width: float, height: float):
        """Short cheek beams give the transparent mask its geometric structure."""
        for side in (-1, 1):
            path = QPainterPath(QPointF(cx + side * width * 0.35, cy + height * 0.05))
            path.lineTo(cx + side * width * 0.20, cy + height * 0.05 + width * 0.045)
            self._stroke(painter, path, 1, 0.25 + 0.35 * self._activation_level)

    def _stroke(self, painter: QPainter, path: QPainterPath, width: float,
                opacity: float, colour: Optional[QColor] = None):
        """Draw fine luminous ink, with contrast confined to the same stroke."""
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setOpacity(opacity)
        ink = QColor(self.INK_COLOUR)
        ink.setAlpha(175)
        pen = QPen(ink, width + 1.7, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)
        glow = QColor(colour or self.PRIMARY_COLOUR)
        glow.setAlpha(36)
        pen.setColor(glow)
        pen.setWidthF(width + 3.0)
        painter.setPen(pen)
        painter.drawPath(path)
        if colour is None:
            bounds = path.boundingRect()
            light = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
            light.setColorAt(0.0, self.PRIMARY_COLOUR.lighter(165))
            light.setColorAt(0.28, self.PRIMARY_COLOUR.lighter(120))
            light.setColorAt(0.66, self.SECONDARY_COLOUR)
            light.setColorAt(1.0, self.PRIMARY_COLOUR.lighter(125))
            pen.setBrush(QBrush(light))
        else:
            pen.setColor(colour)
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.drawPath(path)
        core = QColor(colour or self.PRIMARY_COLOUR.lighter(185))
        core.setAlpha(170)
        pen.setColor(core)
        pen.setWidthF(max(0.45, width * 0.35))
        painter.setPen(pen)
        painter.drawPath(path)
        painter.restore()

    def _draw_eye(self, painter: QPainter, ex: float, ey: float,
                  size: float, blink_factor: float, is_left: bool):
        """Diamond eye beams retain the mask identity through expressions and blinks."""
        openness = self._activation_level * (1.0 - blink_factor)
        height = size * (0.86 + self._listening * 0.10)
        if self._expression == Expression.HAPPY:
            height *= 0.7
        elif self._expression == Expression.SURPRISED:
            height *= 1.16
        elif self._expression in (Expression.SAD, Expression.CONCERNED):
            height *= 0.85
        if self._expression == Expression.CURIOUS and is_left:
            ey -= size * 0.12
        if self._expression == Expression.THINKING:
            ey -= size * 0.08
        height *= openness
        path = QPainterPath(QPointF(ex - size, ey))
        if openness < 0.05:
            path.lineTo(ex + size, ey)
            self._stroke(painter, path, 1.5, 0.52 + self._activation_level * 0.48)
            return
        path.lineTo(ex, ey - height)
        path.lineTo(ex + size, ey)
        path.lineTo(ex, ey + height * 0.5)
        path.closeSubpath()
        opacity = 0.52 + self._activation_level * 0.48
        self._stroke(painter, path, 1.8, opacity)
        if openness < 0.15:
            return
        pupil_size = size * 0.33 * openness
        pupil_x = ex + (self._gaze_x + self._thinking * size * 0.12) * openness
        pupil_y = ey + (self._gaze_y - self._thinking * size * 0.14) * openness
        painter.save()
        painter.setOpacity(opacity)
        pupil = QRadialGradient(pupil_x - pupil_size * 0.3,
                                pupil_y - pupil_size * 0.4, pupil_size * 1.4)
        pupil.setColorAt(0.0, self.PRIMARY_COLOUR.lighter(165))
        pupil.setColorAt(0.5, self.PRIMARY_COLOUR)
        pupil.setColorAt(1.0, self.SECONDARY_COLOUR)
        painter.setBrush(pupil)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(pupil_x, pupil_y), pupil_size, pupil_size)
        painter.restore()

    def _draw_beam_flow(self, painter: QPainter, contour: QPainterPath):
        """Coherent energy flows through fixed beams, brighter during processing."""
        if self._activation_level < 0.001:
            return
        # Continuous phase avoids a jump when the assistant changes state.
        phase = self._elapsed / 12 % 1.0
        intensity = self._activation_level * (0.22 + self._thinking * 0.48)
        for offset in (0.0, 0.5):
            for index in range(16):
                start = (phase + offset - index * 0.004) % 1.0
                end = (start + 0.004) % 1.0
                if end < start:
                    continue
                glint = QPainterPath(contour.pointAtPercent(start))
                glint.lineTo(contour.pointAtPercent(end))
                self._stroke(painter, glint, 2.2, intensity * (1 - index / 16),
                             self.PRIMARY_COLOUR.lighter(190))

    def _draw_mouth(self, painter: QPainter, cx: float, cy: float,
                    face_width: float, face_height: float):
        """A straight mask seam opens into a quiet, tapered speaking waveform."""
        half_width = face_width * (0.16 + self._speaking * 0.075)
        mouth_y = cy + face_height * 0.20
        amplitude = face_height * 0.035 * self._speaking
        path = QPainterPath(QPointF(cx - half_width, mouth_y))
        for step in range(1, 49):
            t = step / 48
            envelope = math.sin(t * math.pi)
            wave = (math.sin(t * math.tau * 2 - self._elapsed * 9)
                    + 0.25 * math.sin(t * math.tau * 3 + self._elapsed * 6)) / 1.25
            y = mouth_y + amplitude * wave * envelope ** 2
            path.lineTo(cx - half_width + 2 * half_width * t, y)
        self._stroke(painter, path, 1.6, 0.48 + self._activation_level * 0.48)

    def _draw_state_contour(self, painter: QPainter, cx: float, cy: float,
                            width: float, height: float):
        """Keep active cues close to the face rather than sending out large waves."""
        if self._listening > 0.001:
            started = self._listening_started_at or self._last_frame_at
            phase = (self._last_frame_at - started) % 2.0 / 2.0
            scale = 1.025 + phase * 0.035
            contour = self._face_contour(cx, cy, width * scale, height * scale)
            self._stroke(painter, contour, 0.7, (1.0 - phase) * 0.33 * self._listening)
        if self._dictating > 0.001:
            pulse = (1.0 - math.cos(self._state_elapsed * math.tau / 2.4)) / 2
            scale = 1.045 + pulse * 0.008
            contour = self._face_contour(cx, cy, width * scale, height * scale)
            self._stroke(painter, contour, 1.0, (0.4 + pulse * 0.2) * self._dictating,
                         QColor(COLORS['error_light']))


class FaceWindow(QWidget):
    """A standalone window containing the Jarvis face."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Jarvis")
        self.setMinimumSize(160, 208)
        self.resize(FaceWidget.DESIGN_WIDTH, FaceWidget.DESIGN_HEIGHT)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setStyleSheet("background: transparent;")
        self.setAccessibleName("Jarvis face")
        self.setToolTip("Drag to move. Right-click to hide.")
        self._drag_offset = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.face = FaceWidget()
        layout.addWidget(self.face)
        self.face.state_observed.connect(self._update_presence)
        self._update_presence(self.face._state_manager.state.value)

        # Position on the right side of the screen
        self._position_on_right()

    def _position_on_right(self):
        """Position the window on the right side of the screen, vertically centered."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return

        screen_geometry = screen.availableGeometry()
        window_width = self.width()
        window_height = self.height()

        # Position on right side with margin, vertically centered
        margin = 20
        x = screen_geometry.right() - window_width - margin
        y = screen_geometry.top() + (screen_geometry.height() - window_height) // 2

        self.move(x, y)

    def set_expression(self, expression: Expression):
        """Set the face expression."""
        self.face.set_expression(expression)

    def _update_presence(self, state_value: str) -> None:
        """Keep resting states quiet and active interaction clearly visible."""
        state = JarvisState(state_value)
        opacity = (
            0.45 if state == JarvisState.ASLEEP
            else 0.72 if state == JarvisState.IDLE else 0.96
        )
        self.setWindowOpacity(opacity)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.setMask(FaceWidget.input_region(self.width(), self.height()))

    def mousePressEvent(self, event) -> None:
        self._drag_offset = None
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.windowHandle()
            if handle is None or not handle.startSystemMove():
                self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.addAction("Hide face", self.hide)
        menu.popup(event.globalPos())
        event.accept()
