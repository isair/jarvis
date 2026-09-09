"""
Vector toaster widget for the Talkie Toaster desktop app with state-driven
animations (complete replacement of the old low-poly orange head).

Drawn entirely with Qt's QPainter primitives (code-native vector, resolution
independent, transparent background). The state machine is shared with the
daemon through a small file-based channel so dev (subprocess) and bundled
(QThread) modes both drive the same widget.

States (JarvisState):
  * ASLEEP: dark, static.
  * IDLE: subtle breathing glow, toast rests inside the slots.
  * WAKE: lever clicks down + one short acknowledgement pulse.
  * LISTENING: toast rises slightly, input-level glow.
  * THINKING: heating elements fill progressively.
  * TOOL: running status dot scans a strip on the body.
  * SPEAKING: mouth arc + light pulse (follows last TTS level when known).
  * SUCCESS: toast pops up once, settles.
  * ERROR: heating glow switches to red briefly, no pop.
  * MUTED: lever up, mic indicator visibly disabled.
  * DICTATING / DICTATION_PROCESSING: pulsing ring (same as before).

Animation is timer-driven (~30 FPS), pauses while the widget is hidden,
becomes a plain static render under the Windows reduced-motion hint, and the
drawing is derived from wall-clock time so restarts stay phase-stable.
"""

from __future__ import annotations
import math
import time as _time
from enum import Enum
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QObject


class Expression(Enum):
    """Available face expressions (kept for API compatibility)."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    THINKING = "thinking"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    EXCITED = "excited"
    CONCERNED = "concerned"


class JarvisState(Enum):
    """Overall assistant state for the toaster animation."""
    ASLEEP = "asleep"                 # Daemon not started yet
    IDLE = "idle"                     # Awake and ready, waiting for wake word
    LISTENING = "listening"           # Actively listening (collecting or hot window)
    THINKING = "thinking"             # Processing query
    SPEAKING = "speaking"             # Speaking response
    DICTATING = "dictating"           # Hold-to-dictate recording active
    DICTATION_PROCESSING = "dictation_processing"  # Transcribing & pasting captured dictation
    WAKE = "wake"                     # Wake phrase recognised this moment
    TOOL = "tool"                     # A tool execution is running
    SUCCESS = "success"               # Tool/reply finished successfully
    ERROR = "error"                   # Tool/reply failed
    MUTED = "muted"                   # Microphone disabled / not listening


# Global assistant state - allows daemon to signal overall state to the widget.
# Uses a file-based approach to work across processes (dev mode runs daemon as
# subprocess). Format: "<state_value>" | "<state_value>|<level_float>" |
# "<state_value>|<level>|<reason label>".
import tempfile
import os


def _get_jarvis_state_file() -> str:
    """Get the path to the Jarvis state file."""
    return os.path.join(tempfile.gettempdir(), "jarvis_state")


class JarvisStateManager(QObject):
    """Global singleton for assistant state management.

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
        self._state_lock = threading_lock()
        self._state_file = _get_jarvis_state_file()
        self._level: float = 0.0
        self._label: str = ""
        # Always start fresh in ASLEEP state on app launch
        # (state file is for cross-process communication during a session,
        # not for persisting state across app restarts)
        self._write_state(JarvisState.ASLEEP)

    @property
    def state(self) -> JarvisState:
        """Read current state (checks file for cross-process communication)."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    content = f.read().strip()
                if content:
                    parts = content.split("|")
                    head = parts[0]
                    try:
                        self._level = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                    except ValueError:
                        pass
                    self._label = parts[2] if len(parts) > 2 else ""
                    return JarvisState(head)
        except (ValueError, OSError):
            # Invalid content or read error - fall back to in-memory state
            pass

        with self._state_lock:
            return self._state

    @property
    def level(self) -> float:
        """0..1 amplitude for glow/mouth following (0 when unknown)."""
        # Refresh from file (cheap 1-read) so cross-process levels arrive.
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    content = f.read().strip()
                parts = content.split("|")
                if len(parts) > 1 and parts[1]:
                    return float(parts[1])
        except (ValueError, OSError):
            pass
        return self._level

    @property
    def label(self) -> str:
        """Short reason label (e.g. 'CPU temperature') shown under the body."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    content = f.read().strip()
                parts = content.split("|")
                if len(parts) > 2:
                    return parts[2]
        except OSError:
            pass
        return self._label

    def _write_state(self, state: JarvisState, level: float = 0.0, label: str = "") -> None:
        """Write state (and optional level/label) to file for cross-process use."""
        try:
            parts = [state.value]
            if label and not level:
                parts.append("")  # keep slot ordering for the label field
            if level:
                parts.append(f"{level:.3f}")
            if label:
                parts.append(label)
            with open(self._state_file, 'w') as f:
                f.write("|".join(parts))
        except OSError:
            # File write failed - state won't be shared across processes
            pass

    def set_state(self, state: JarvisState, level: float = 0.0, label: Optional[str] = None) -> None:
        """Set the assistant state (thread-safe, cross-process)."""
        with self._state_lock:
            self._state = state
            if level:
                self._level = level
            if label is not None:
                self._label = label

        # Write to file for cross-process communication
        self._write_state(state, level or self._level, label or self._label)

        # Emit signal for same-process listeners
        try:
            self.state_changed.emit(state.value)
        except RuntimeError:
            # If Qt event loop isn't running, just update the flag
            pass


def threading_lock():
    import threading
    return threading.Lock()


# Module-level singleton instance
_jarvis_state_instance: Optional[JarvisStateManager] = None
import threading
_jarvis_state_lock = threading.Lock()


def get_jarvis_state() -> JarvisStateManager:
    """Get the global Jarvis state singleton."""
    global _jarvis_state_instance
    with _jarvis_state_lock:
        if _jarvis_state_instance is None:
            _jarvis_state_instance = JarvisStateManager()
        return _jarvis_state_instance


class LowPolyFaceWidget(QWidget):
    """
    Vector toaster widget with expressions and speaking animation.

    The old low-poly head is fully replaced: the widget now draws a compact
    polished-metal toaster (two bread slots, two toast slices, lever, warm
    heating glow) whose face is integrated into the body. No raster assets.
    """

    # Colors
    PRIMARY_COLOR = QColor("#fbbf24")     # Amber/gold accents
    SECONDARY_COLOR = QColor("#f59e0b")
    GLOW_COLOR = QColor("#fcd34d")
    BG_COLOR = QColor(10, 11, 15, 242)    # Near-black rounded panel
    GRID_COLOR = QColor("#1f1f1f")
    BODY_LIGHT = QColor("#d7dbe0")
    BODY_DARK = QColor("#8f959c")
    ERROR_COLOR = QColor("#ef4444")
    TOAST_COLOR = QColor("#e8b96b")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 400)

        # Current assistant state
        self._state_manager = get_jarvis_state()
        self._state_manager.state_changed.connect(self._on_state_changed)
        self._jarvis_state = self._state_manager.state

        self._expression = Expression.NEUTRAL

        # Animation clock (wall time keeps animations phase-stable across
        # pauses; frame counters only exist for the blink scheduler).
        self._t0 = _time.monotonic()
        self._last_tick = self._t0

        # Reduced-motion hint (Windows: Settings > Accessibility).
        self._reduced_motion = self._detect_reduced_motion()

        # Per-transition marks (pop / lever / pulse bookkeeping).
        self._wake_at: Optional[float] = None
        self._success_at: Optional[float] = None
        self._error_until: float = 0.0
        self._prev_state: JarvisState = JarvisState.ASLEEP

        # Blink timers (only meaningful without reduced motion).
        self._is_blinking = False
        self._blink_started_at: Optional[float] = None
        self._schedule_next_blink()

        # Animation timer (≈30 FPS). Paused while the widget is hidden.
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        if self._reduced_motion:
            # Reduced motion: still animate, but the render path uses
            # fewer, larger steps (see _animate). Timer stays at 33 ms.
            pass
        self._animation_timer.start(33)

    # ------------------------------------------------------------------ #
    # State plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_reduced_motion() -> bool:
        """Honour the Windows reduced-motion accessibility hint when Qt
        exposes it (Qt >= 6.3: styleHints().timeLineCurveStyle())."""
        app = QApplication.instance()
        if app is None:
            return False
        try:
            return app.styleHints().timeLineCurveStyle() == Qt.TimeLineCurveStyle.CurveStyleLinear
        except Exception:
            return False

    def _on_state_changed(self, state_value: str):
        try:
            new_state = JarvisState(state_value)
        except ValueError:
            return
        self._apply_state(new_state)

    def _apply_state(self, new_state: JarvisState) -> None:
        now = _time.monotonic() - self._t0
        if new_state != self._prev_state:
            if new_state == JarvisState.WAKE:
                self._wake_at = now
            elif new_state == JarvisState.SUCCESS:
                self._success_at = now
            elif new_state == JarvisState.ERROR:
                self._error_until = now + 1.2
        self._prev_state = new_state
        self._jarvis_state = new_state

    def showEvent(self, event):
        self._animation_timer.start(33)
        super().showEvent(event)

    def hideEvent(self, event):
        self._animation_timer.stop()  # No repaints while hidden.
        super().hideEvent(event)

    def set_expression(self, expression: Expression):
        if expression != self._expression:
            self._expression = expression

    # ------------------------------------------------------------------ #
    # Animation tick
    # ------------------------------------------------------------------ #
    def _schedule_next_blink(self):
        if self._reduced_motion:
            return  # Static eyes under reduced motion.
        interval = random_interval_ms()
        QTimer.singleShot(interval, self._start_blink)

    def _start_blink(self):
        if not self._is_blinking:
            self._is_blinking = True
            self._blink_started_at = _time.monotonic() - self._t0
        self._schedule_next_blink()

    def _animate(self):
        """Timer tick: refresh the state from the shared file and repaint."""
        try:
            polled = self._state_manager.state
        except Exception:
            polled = self._jarvis_state
        if polled != self._jarvis_state:
            self._apply_state(polled)

        # Blink progression (0.24 s close+open cycle).
        if self._is_blinking and self._blink_started_at is not None:
            elapsed = (_time.monotonic() - self._t0) - self._blink_started_at
            if elapsed > 0.24:
                self._is_blinking = False
                self._blink_started_at = None

        self.update()

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _elapsed(self) -> float:
        return _time.monotonic() - self._t0

    def _blink_factor(self) -> float:
        if self._activation() < 0.5:
            return 1.0
        if self._is_blinking and self._blink_started_at is not None:
            p = min(1.0, max(0.0, (_time.monotonic() - self._t0 - self._blink_started_at) / 0.24))
            return p * 2 if p < 0.5 else 2 - p * 2
        return 0.0

    def _activation(self) -> float:
        return 0.0 if self._jarvis_state == JarvisState.ASLEEP else 1.0

    def _level(self) -> float:
        try:
            return max(0.0, min(1.0, self._state_manager.level))
        except Exception:
            return 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Panel (rounded, translucent-dark) — no orange head behind it.
        painter.setPen(QPen(QColor("#27272a"), 1))
        painter.setBrush(QBrush(self.BG_COLOR))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)

        activation = self._activation()
        t = self._elapsed()

        # Geometry: centred 3:4-ish toaster body.
        body_w = min(w, h) * 0.62
        body_h = body_w * 0.78
        cx, cy = w / 2, h / 2 + body_h * 0.06
        left, top = cx - body_w / 2, cy - body_h / 2
        right, bottom = cx + body_w / 2, cy + body_h / 2

        op = 0.35 + 0.65 * activation  # activation-driven opacity

        # Breathing scale (IDLE/LISTENING): tiny, slow.
        breathe = 1.0
        if self._jarvis_state in (JarvisState.IDLE, JarvisState.LISTENING) and not self._reduced_motion:
            breathe = 1.0 + 0.012 * math.sin(t * 1.6) * activation

        painter.save()
        painter.translate(cx, cy)
        painter.scale(*_pair(breathe))
        painter.translate(-cx, -cy)

        # ---- Glow (warm heating; red briefly on ERROR) ----
        glow_alpha = op
        glow_color = QColor(self.ERROR_COLOR) if self._jarvis_state == JarvisState.ERROR else QColor(self.GLOW_COLOR)
        pulse = 1.0
        if not self._reduced_motion and self._jarvis_state in (JarvisState.IDLE, JarvisState.LISTENING, JarvisState.SPEAKING):
            pulse = 0.55 + 0.45 * math.sin(t * 2.4)
        glow = QRadialGradient(cx, cy, body_w * 0.85)
        c = QColor(glow_color)
        c.setAlphaF(0.35 * glow_alpha * pulse)
        glow.setColorAt(0, c)
        c.setAlphaF(0)
        glow.setColorAt(1, c)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), body_w * 0.85, body_w * 0.85)

        # ---- Toast slices behind the slots (rise per state) ----
        slice_w = body_w * 0.30
        slice_h = body_h * 0.26
        gap = body_w * 0.10
        slice_top = top - body_h * 0.06
        rise = 0.0
        if self._jarvis_state == JarvisState.LISTENING:
            rise = slice_h * 0.30
        elif self._jarvis_state == JarvisState.SUCCESS:
            # One-time pop: overshoot then settle.
            since = t - (self._success_at if self._success_at is not None else t)
            if since < 0.6:
                rise = slice_h * 0.9 * math.sin(min(1.0, since / 0.6) * math.pi * 1.15) + slice_h * 0.35
            else:
                rise = slice_h * 0.35
        for sgn in (-1, 1):
            sx = cx + sgn * (slice_w / 2 + gap / 2)
            rect_top = slice_top - rise
            toast_pen = QPen(QColor("#c98f3d"), 2)
            painter.setOpacity(op)
            painter.setPen(toast_pen)
            painter.setBrush(QBrush(QColor("#e8b96b")))
            painter.drawRoundedRect(
                QRectF_(sx - slice_w / 2, rect_top, slice_w, slice_h),
                slice_w * 0.18, slice_h * 0.18,
            )
            # Crust inner line
            inner = QPen(QColor("#b0782f"), 1)
            painter.setPen(inner)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QRectF_(sx - slice_w / 2 + 3, rect_top + 3, slice_w - 6, slice_h - 6),
                slice_w * 0.14, slice_h * 0.14,
            )

        # ---- Toaster body (polished metal gradient) ----
        grad = QLinearGradient(left, top, right, bottom)
        grad.setColorAt(0.0, self.BODY_LIGHT)
        grad.setColorAt(0.55, self.BODY_DARK)
        grad.setColorAt(1.0, self.BODY_LIGHT)
        painter.setOpacity(op)
        painter.setPen(QPen(QColor("#5f666d"), 2))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(QRectF_(left, top, body_w, body_h), 14, 12)

        # Two bread slots on the top edge.
        slot_h = body_h * 0.10
        for sgn in (-1, 1):
            sx = cx + sgn * (slice_w / 2 + gap / 2)
            painter.setPen(QPen(QColor("#3a4046"), 1))
            painter.setBrush(QBrush(QColor(20, 23, 30, int(230 * op))))
            painter.drawRoundedRect(
                QRectF_(sx - slice_w / 2 + 4, top + 2, slice_w - 8, slot_h),
                slot_h * 0.5, slot_h * 0.5,
            )

        # Lever (front-right side). Pressed down in WAKE, up otherwise.
        lever_x = right - body_w * 0.08
        track_top = top + body_h * 0.22
        track_bottom = top + body_h * 0.62
        painter.setPen(QPen(QColor("#5f666d"), 2))
        painter.drawLine(QPointF(lever_x, track_top), QPointF(lever_x, track_bottom))
        pressed = 1.0
        if self._jarvis_state == JarvisState.WAKE:
            since = t - (self._wake_at if self._wake_at is not None else t)
            # Quick click: down (0.12 s), hold, rebound (0.35 s total).
            if since < 0.35:
                pressed = 1.0 if since > 0.25 else 0.0 + (since / 0.28) * 0.15
        knob_y = track_bottom - (track_bottom - track_top) * (0.35 + 0.65 * pressed)
        painter.setBrush(QBrush(self.PRIMARY_COLOR))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(lever_x, knob_y), body_w * 0.035, body_w * 0.035)

        # ---- Heating elements (inside body, below slots) ----
        elem_top = top + body_h * 0.16
        elem_h = body_h * 0.055
        elem_gap = body_h * 0.045
        element_color = QColor(self.ERROR_COLOR) if self._jarvis_state == JarvisState.ERROR else self.SECONDARY_COLOR
        fill = 0.0
        if self._jarvis_state == JarvisState.THINKING:
            # Progressive: three lines fill in sequence over ~1.8 s.
            if self._wake_at is None:
                self._wake_at = t  # anchor for the loop cycle
            cycle = (t % 1.8) / 1.8
            fill = cycle
            painter.setPen(QPen(element_color, 2))
            for i in range(3):
                frac = max(0.0, min(1.0, fill * 3 - i))
                yy = elem_top + (elem_h + elem_gap) * i
                x0 = cx - body_w * 0.30
                x1 = x0 + body_w * 0.60 * frac
                if x1 > x0:
                    painter.drawLine(QPointF(x0, yy), QPointF(x1, yy))
        else:
            painter.setOpacity(op * 0.9)
            painter.setPen(QPen(element_color, 2))
            for i in range(3):
                yy = elem_top + (elem_h + elem_gap) * i
                painter.drawLine(
                    QPointF(cx - body_w * 0.30, yy),
                    QPointF(cx + body_w * 0.30, yy),
                )

        # ---- Tool execution: running dot across a strip ----
        if self._jarvis_state == JarvisState.TOOL:
            strip_y = bottom - body_h * 0.10
            painter.setPen(QPen(QColor("#3a4046"), 1))
            painter.drawLine(QPointF(cx - body_w * 0.32, strip_y),
                             QPointF(cx + body_w * 0.32, strip_y))
            prog = (t % 1.2) / 1.2
            dot_x = cx - body_w * 0.32 + (body_w * 0.64) * prog
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.PRIMARY_COLOR))
            painter.drawEllipse(QPointF(dot_x, strip_y), 3.0, 3.0)

        # ---- Face: eyes + mouth integrated into the body ----
        eye_y = elem_top + 3 * (elem_h + elem_gap) + body_h * 0.04
        eye_r = body_w * 0.045
        blink = self._blink_factor() if not self._reduced_motion else 1.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.PRIMARY_COLOR))
        for sgn in (-1, 1):
            ex = cx + sgn * body_w * 0.16
            er = eye_r * (1.0 - blink * 0.85)
            if er > 0.4:
                painter.drawEllipse(QPointF(ex, eye_y), er, max(er, 0.8))

        # Mouth: waveform follows level when known, else gentle sine.
        level = self._level()
        mouth_y = eye_y + body_h * 0.10
        mouth_w = body_w * 0.22
        amp = max(0.10, level) * body_h * 0.045
        if self._jarvis_state == JarvisState.SPEAKING and not self._reduced_motion:
            amp = level or (0.35 + 0.35 * math.sin(t * 5.2))
            amp *= body_h * 0.045 + 1.2
        painter.setOpacity(op)
        path = QPainterPath()
        n = 36
        x0 = cx - mouth_w
        path.moveTo(x0, mouth_y)
        for i in range(n + 1):
            tt = i / n
            x = x0 + mouth_w * 2 * tt
            edge = 1.0 - abs(tt - 0.5) * 1.2
            if not self._reduced_motion and self._jarvis_state == JarvisState.SPEAKING:
                yy = mouth_y + amp * edge * math.sin((tt * 6.0 + t * 4.0) * math.pi) 
            else:
                yy = mouth_y + 2.5 * edge
            path.lineTo(x, yy)
        mpen = QPen(self.PRIMARY_COLOR, 2.0)
        mpen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mpen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # ---- Muted state: gray mic indicator, lever stays up ----
        if self._jarvis_state == JarvisState.MUTED:
            painter.setOpacity(op)
            mic_r = body_w * 0.05
            painter.setPen(QPen(QColor("#a1a1aa"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, bottom - body_h * 0.19), mic_r, mic_r)
            painter.drawLine(QPointF(cx - mic_r * 0.6, bottom - body_h * 0.19 - mic_r * 0.6),
                             QPointF(cx + mic_r * 0.6, bottom - body_h * 0.19 + mic_r * 0.6))

        # ---- Dictation ring (same as legacy, around the body) ----
        if self._jarvis_state in (JarvisState.DICTATING, JarvisState.DICTATION_PROCESSING):
            pulse = (math.sin(t * 2.6) + 1.0) / 2.0
            scale = 1.08 + pulse * 0.07
            ring_color = QColor(239, 68, 68) if self._jarvis_state == JarvisState.DICTATING else QColor(self.GLOW_COLOR)
            painter.setOpacity(0.35 + pulse * 0.25)
            painter.setPen(QPen(ring_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QRectF_(cx - body_w * scale / 2, cy - body_h * scale / 1.55,
                        body_w * scale, body_h * scale * 0.95),
                14, 12,
            )

        painter.restore()
        painter.setOpacity(1.0)

        # ---- Reason label for proactive speech (why did it just speak?) ----
        try:
            reason_label = (self._state_manager.label or "").strip()
        except Exception:
            reason_label = ""
        if reason_label and activation > 0:
            painter.setPen(QPen(self.PRIMARY_COLOR))
            font = painter.font()
            font.setPixelSize(max(10, int(body_h * 0.075)))
            painter.setFont(font)
            painter.drawText(
                QRectF_(left, bottom + body_h * 0.04, body_w, body_h * 0.10),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                reason_label,
            )

        painter.end()

    def _draw_background(self, painter: QPainter, w: int, h: int):
        # Kept for API parity with the legacy widget name set.
        pass


def QRectF_(x, y, w, h):
    from PyQt6.QtCore import QRectF
    return QRectF(x, y, w, h)


def _pair(v):
    return (v, v)


def random_interval_ms() -> int:
    import random
    return random.randint(2000, 5000)


class FaceWindow(QWidget):
    """A standalone window containing the Toustovač toaster."""

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            from jarvis.config import BRANDING
            self.setWindowTitle(f"🍞 {BRANDING['display_name']}")
        except Exception:
            self.setWindowTitle("Toustovač")
        self.setMinimumSize(320, 420)
        self.resize(350, 450)

        # Set window flags for floating window (always-on-top; recording mode
        # keeps the overlay persistent but unobtrusive).
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowStaysOnTopHint
        )

        # Transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Toaster widget
        self.face = LowPolyFaceWidget()
        layout.addWidget(self.face)

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

        # Animate overlay scale from config (recording mode).
        try:
            from jarvis.config import load_config
            scale = float(load_config().get("overlay_scale", 1.0) or 1.0)
        except Exception:
            scale = 1.0
        if 0.4 <= scale <= 2.0 and abs(scale - 1.0) > 1e-6:
            window_width = max(240, int(window_width * scale))
            window_height = max(320, int(window_height * scale))
            self.resize(window_width, window_height)

        # Position on right side with margin, vertically centered
        margin = 20
        x = screen_geometry.right() - window_width - margin
        y = screen_geometry.top() + (screen_geometry.height() - window_height) // 2

        self.move(x, y)

    def set_expression(self, expression: Expression):
        """Set the face expression."""
        self.face.set_expression(expression)
