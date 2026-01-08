"""
Low-poly grid face widget for Jarvis with intelligent state management and organic idle behavior.

Features:
- Low-poly wireframe aesthetic with glowing effects
- Smooth continuous waveform mouth visualization:
  * Uses multiple layered sine waves for natural audio-like appearance
  * Amplitude and frequency vary to simulate speech patterns
  * Edge tapering for organic look
  * 60-point smooth curve with glow effect
- Comprehensive state system (ASLEEP, IDLE, LISTENING, THINKING, SPEAKING)
- Smooth wake/sleep transitions with opacity-based activation
- Animated spinner pupils when thinking
- Intelligent idle activity system (only active in IDLE state) that alternates between behaviors:
  * looking_around (35%) - Frequent eye movement scanning the environment
  * hovering (25%) - Gentle vertical floating motion
  * head_tilt (20%) - Subtle head rotation
  * deep_gaze (10%) - Focused staring at one point
  * stretch (7%) - Bigger movement with enhanced breathing
  * yawn (3%) - Rare tired behavior with eye closing
- Base breathing animation always active when awake
- All activities smoothly transition and respect current state
- Multiple expressions for future use (neutral, happy, sad, thinking, etc.)
"""

from __future__ import annotations
import math
import random
import threading
from typing import Optional, List, Tuple
from enum import Enum
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QObject


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
    """
    state_changed = pyqtSignal(str)

    _instance: Optional["JarvisStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._state = JarvisState.ASLEEP  # Start asleep
        self._state_lock = threading.Lock()
        self._state_file = _get_jarvis_state_file()
        self._initialized = True
        # Clean up any stale state file on init
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
        except Exception:
            pass

        with self._state_lock:
            return self._state

    def _write_state(self, state: JarvisState) -> None:
        """Write state to file for cross-process communication."""
        try:
            with open(self._state_file, 'w') as f:
                f.write(state.value)
        except Exception:
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


def get_jarvis_state() -> JarvisStateManager:
    """Get the global Jarvis state singleton."""
    return JarvisStateManager()


class LowPolyFaceWidget(QWidget):
    """
    A low-poly wireframe face widget with expressions and speaking animation.
    
    The face is rendered as a geometric mesh with glowing vertices and edges,
    creating a futuristic AI assistant aesthetic.
    """
    
    # Colors
    PRIMARY_COLOR = QColor("#fbbf24")  # Amber/gold - matches Jarvis theme
    SECONDARY_COLOR = QColor("#f59e0b")  # Darker amber
    GLOW_COLOR = QColor("#fcd34d")  # Light amber for glow
    BG_COLOR = QColor("#0a0a0a")  # Near black background
    GRID_COLOR = QColor("#1f1f1f")  # Dark gray for background grid
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 400)

        # Current Jarvis state
        self._jarvis_state = JarvisState.ASLEEP  # Start asleep until daemon ready
        self._mouth_openness = 0.0  # 0.0 = closed, 1.0 = fully open
        self._target_mouth_openness = 0.0
        self._blink_timer = 0
        self._is_blinking = False
        self._blink_progress = 0.0

        # Soundwave visualization (for mouth) - continuous line waveform
        self._waveform_time = 0.0  # Time parameter for waveform animation
        self._waveform_amplitude = 0.0  # Overall amplitude (smoothly changes)
        self._waveform_frequency_base = 0.15  # Base frequency for wave oscillation
        self._waveform_detail_offset = 0.0  # Offset for detail variations

        # Expression state
        self._expression = Expression.NEUTRAL
        self._expression_transition = 1.0  # 1.0 = fully transitioned

        # Vertex jitter for organic feel
        self._jitter_offset = 0.0
        self._vertex_jitters: List[Tuple[float, float]] = []

        # Activation state (for sleep/wake animation)
        self._activation_level = 0.0  # 0.0 = asleep, 1.0 = fully awake
        self._target_activation = 0.0

        # Idle animations - base layer (always active when awake)
        self._breathing_scale = 1.0  # Breathing scale factor
        self._breathing_time = 0.0

        # Idle activity system - activities alternate with different probabilities
        self._current_activity = None  # Current idle activity
        self._activity_timer = 0  # Frames in current activity
        self._activity_duration = 0  # Duration of current activity
        self._activity_cooldown = 0  # Frames until next activity selection

        # Activity-specific animation state
        self._hover_offset = 0.0
        self._hover_time = 0.0
        self._head_tilt = 0.0
        self._head_tilt_time = 0.0
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._target_gaze_x = 0.0
        self._target_gaze_y = 0.0
        self._stretch_intensity = 0.0  # For stretching activity
        self._yawn_progress = 0.0  # For yawning activity

        # Thinking spinner animation
        self._spinner_angle = 0.0  # Rotation angle for thinking spinner

        # Connect to global Jarvis state
        self._state_manager = get_jarvis_state()
        self._state_manager.state_changed.connect(self._on_state_changed)

        # Animation timer
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start(33)  # ~30 FPS

        # Blink timer (random intervals)
        self._schedule_next_blink()
        
    def _schedule_next_blink(self):
        """Schedule the next blink at a random interval."""
        interval = random.randint(2000, 5000)  # 2-5 seconds
        QTimer.singleShot(interval, self._start_blink)
    
    def _start_blink(self):
        """Start a blink animation."""
        if not self._is_blinking:
            self._is_blinking = True
            self._blink_progress = 0.0
        self._schedule_next_blink()

    def _on_state_changed(self, state_value: str):
        """Handle Jarvis state change from global state."""
        try:
            self._jarvis_state = JarvisState(state_value)
        except ValueError:
            pass

    def set_expression(self, expression: Expression):
        """Set the face expression."""
        if expression != self._expression:
            self._expression = expression
            self._expression_transition = 0.0

    def _select_idle_activity(self) -> str:
        """Select a random idle activity based on weighted probabilities."""
        activities = [
            ("looking_around", 35),  # Most common - natural eye movement
            ("hovering", 25),        # Common - gentle floating
            ("head_tilt", 20),       # Common - subtle head rotation
            ("deep_gaze", 10),       # Occasional - stare at one spot
            ("stretch", 7),          # Occasional - bigger movement
            ("yawn", 3),            # Rare - eyes close briefly
        ]

        # Weighted random selection
        total_weight = sum(weight for _, weight in activities)
        rand = random.random() * total_weight
        cumulative = 0

        for activity, weight in activities:
            cumulative += weight
            if rand <= cumulative:
                return activity

        return "looking_around"  # Fallback

    def _get_activity_duration(self, activity: str) -> int:
        """Get duration in frames for an activity."""
        durations = {
            "looking_around": random.randint(90, 240),   # 3-8 seconds
            "hovering": random.randint(120, 300),        # 4-10 seconds
            "head_tilt": random.randint(90, 210),        # 3-7 seconds
            "deep_gaze": random.randint(150, 360),       # 5-12 seconds (longer stare)
            "stretch": random.randint(60, 120),          # 2-4 seconds (quick stretch)
            "yawn": random.randint(90, 150),             # 3-5 seconds
        }
        return durations.get(activity, 120)

    def _update_activity_animation(self):
        """Update animation for the current activity."""
        if self._current_activity == "looking_around":
            # Frequently change gaze direction
            if self._activity_timer % 60 == 0:  # Change every 2 seconds
                self._target_gaze_x = (random.random() - 0.5) * 25  # ±12.5 pixels
                self._target_gaze_y = (random.random() - 0.5) * 15  # ±7.5 pixels
            self._gaze_x += (self._target_gaze_x - self._gaze_x) * 0.08
            self._gaze_y += (self._target_gaze_y - self._gaze_y) * 0.08
            # Minimal other movements
            self._hover_offset *= 0.95
            self._head_tilt *= 0.95
            self._stretch_intensity *= 0.9

        elif self._current_activity == "hovering":
            # Gentle floating motion
            self._hover_time += 0.02
            self._hover_offset = math.sin(self._hover_time) * 8.0
            # Minimal other movements
            self._gaze_x *= 0.98
            self._gaze_y *= 0.98
            self._head_tilt *= 0.95
            self._stretch_intensity *= 0.9

        elif self._current_activity == "head_tilt":
            # Subtle head rotation
            self._head_tilt_time += 0.015
            self._head_tilt = math.sin(self._head_tilt_time * 0.7) * 2.5
            # Minimal other movements
            self._gaze_x *= 0.98
            self._gaze_y *= 0.98
            self._hover_offset *= 0.95
            self._stretch_intensity *= 0.9

        elif self._current_activity == "deep_gaze":
            # Stare at one spot intently
            if self._activity_timer == 0:  # Pick spot at start
                self._target_gaze_x = (random.random() - 0.5) * 30  # ±15 pixels (wider range)
                self._target_gaze_y = (random.random() - 0.5) * 20  # ±10 pixels
            self._gaze_x += (self._target_gaze_x - self._gaze_x) * 0.04  # Slower, more focused
            self._gaze_y += (self._target_gaze_y - self._gaze_y) * 0.04
            # Very minimal other movements
            self._hover_offset *= 0.98
            self._head_tilt *= 0.98
            self._stretch_intensity *= 0.9

        elif self._current_activity == "stretch":
            # Bigger movement - scale up briefly
            progress = self._activity_timer / self._activity_duration
            if progress < 0.3:  # Stretch out
                self._stretch_intensity += (1.0 - self._stretch_intensity) * 0.15
            elif progress > 0.7:  # Return to normal
                self._stretch_intensity *= 0.85
            else:  # Hold stretch
                self._stretch_intensity += (1.0 - self._stretch_intensity) * 0.05

            # Apply stretch to breathing scale (enhance it)
            stretch_boost = self._stretch_intensity * 0.03
            self._breathing_scale += stretch_boost

            # Add movement during stretch
            self._hover_time += 0.03
            self._hover_offset = math.sin(self._hover_time) * 12.0 * self._stretch_intensity
            self._head_tilt = math.sin(self._activity_timer * 0.1) * 4.0 * self._stretch_intensity

        elif self._current_activity == "yawn":
            # Eyes close and open, subtle mouth movement
            progress = self._activity_timer / self._activity_duration
            if progress < 0.3:  # Close eyes
                self._yawn_progress += (1.0 - self._yawn_progress) * 0.15
            elif progress > 0.7:  # Open eyes
                self._yawn_progress *= 0.85
            else:  # Hold
                self._yawn_progress += (1.0 - self._yawn_progress) * 0.05

            # Minimal other movements
            self._gaze_x *= 0.95
            self._gaze_y *= 0.95
            self._hover_offset *= 0.95
            self._head_tilt *= 0.95
            self._stretch_intensity *= 0.9

    def _decay_activity_animations(self):
        """Smoothly decay all activity animations when not idle."""
        self._gaze_x *= 0.92
        self._gaze_y *= 0.92
        self._hover_offset *= 0.92
        self._head_tilt *= 0.92
        self._stretch_intensity *= 0.85
        self._yawn_progress *= 0.85
        self._target_gaze_x *= 0.92
        self._target_gaze_y *= 0.92
    
    def _animate(self):
        """Animation tick - update all animated properties."""
        # Poll Jarvis state directly (more reliable than cross-thread signals)
        try:
            self._jarvis_state = self._state_manager.state
        except Exception:
            pass

        # Update activation level based on state
        if self._jarvis_state == JarvisState.ASLEEP:
            self._target_activation = 0.0
        else:
            # IDLE, LISTENING, THINKING, or SPEAKING - all should be awake
            self._target_activation = 1.0

        # Smooth activation transition
        activation_diff = self._target_activation - self._activation_level
        self._activation_level += activation_diff * 0.05  # Slow wake/sleep

        # Check if idle (when awake but not actively doing anything)
        # ONLY IDLE state gets idle activities - not listening, thinking, or speaking
        is_idle = self._jarvis_state == JarvisState.IDLE and self._activation_level > 0.5

        # Base layer: Breathing animation (always active when awake)
        self._breathing_time += 0.025
        breathing_factor = math.sin(self._breathing_time) * 0.015 * self._activation_level
        self._breathing_scale = 1.0 + breathing_factor

        # Idle activity system
        if is_idle:
            # Activity selection and management
            if self._activity_cooldown > 0:
                self._activity_cooldown -= 1
            elif self._current_activity is None or self._activity_timer >= self._activity_duration:
                # Select new activity
                self._current_activity = self._select_idle_activity()
                self._activity_duration = self._get_activity_duration(self._current_activity)
                self._activity_timer = 0
                # Set cooldown before next activity (1-3 seconds of neutral state)
                if self._current_activity != self._current_activity:  # Reset on new activity
                    self._activity_cooldown = 0
            else:
                self._activity_timer += 1

            # Update current activity
            self._update_activity_animation()
        else:
            # Not idle - smoothly decay all activity animations
            self._current_activity = None
            self._activity_timer = 0
            self._activity_cooldown = 0
            self._decay_activity_animations()

        # Reduce gaze when speaking
        if self._jarvis_state == JarvisState.SPEAKING:
            self._gaze_x *= 0.95
            self._gaze_y *= 0.95

        # Spinner animation (when thinking)
        if self._jarvis_state == JarvisState.THINKING:
            self._spinner_angle += 8.0  # Rotate 8 degrees per frame (~240 deg/sec)
            if self._spinner_angle >= 360:
                self._spinner_angle -= 360

        # Soundwave animation (when speaking)
        if self._jarvis_state == JarvisState.SPEAKING:
            # Animate waveform parameters for natural audio-like movement
            self._waveform_time += 0.12  # Speed of wave movement
            self._waveform_detail_offset += 0.08  # Speed of detail variations

            # Vary amplitude smoothly (simulates volume changes in speech)
            target_amplitude = 0.6 + random.random() * 0.4  # 0.6 to 1.0
            self._waveform_amplitude += (target_amplitude - self._waveform_amplitude) * 0.15

            # Occasionally change base frequency (simulates pitch changes in speech)
            if random.random() < 0.02:  # 2% chance per frame
                self._waveform_frequency_base = 0.1 + random.random() * 0.15  # 0.1 to 0.25
        else:
            # Decay waveform to flat line when not speaking
            self._waveform_amplitude *= 0.85
            self._waveform_time += 0.03  # Slower drift when not speaking

        # Blink animation (only when awake)
        if self._activation_level > 0.5:
            if self._is_blinking:
                self._blink_progress += 0.15
                if self._blink_progress >= 1.0:
                    self._is_blinking = False
                    self._blink_progress = 0.0
        else:
            # When asleep, keep eyes closed (will be forced in draw logic)
            self._is_blinking = False
            self._blink_progress = 0.0

        # Expression transition
        if self._expression_transition < 1.0:
            self._expression_transition += 0.1
            self._expression_transition = min(1.0, self._expression_transition)

        # Vertex jitter (reduce when asleep)
        jitter_speed = 0.1 * self._activation_level
        self._jitter_offset += jitter_speed

        self.update()
    
    def _get_jitter(self, index: int, scale: float = 1.0) -> Tuple[float, float]:
        """Get a subtle jitter offset for a vertex."""
        t = self._jitter_offset + index * 0.5
        jx = math.sin(t * 1.3) * scale
        jy = math.cos(t * 1.7) * scale
        return (jx, jy)
    
    def paintEvent(self, event):
        """Render the low-poly face."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        # Apply hover offset to center position
        cy += self._hover_offset

        # Draw background
        self._draw_background(painter, w, h)

        # Save painter state and apply transformations
        painter.save()
        painter.translate(cx, cy)  # Move origin to face center
        painter.scale(self._breathing_scale, self._breathing_scale)  # Apply breathing scale
        painter.rotate(self._head_tilt)  # Apply subtle rotation
        painter.translate(-cx, -cy)  # Move origin back

        # Calculate face dimensions
        face_width = min(w, h) * 0.7
        face_height = face_width * 1.3

        # Draw the face mesh
        self._draw_face_mesh(painter, cx, cy, face_width, face_height)

        # Draw eyes
        self._draw_eyes(painter, cx, cy, face_width, face_height)

        # Draw mouth
        self._draw_mouth(painter, cx, cy, face_width, face_height)

        # Draw accent lines
        self._draw_accent_lines(painter, cx, cy, face_width, face_height)

        # Restore painter state
        painter.restore()

        painter.end()
    
    def _draw_background(self, painter: QPainter, w: int, h: int):
        """Draw the dark background with subtle grid."""
        # Solid background
        painter.fillRect(0, 0, w, h, self.BG_COLOR)
        
        # Subtle background grid
        grid_pen = QPen(self.GRID_COLOR, 1)
        painter.setPen(grid_pen)
        
        grid_size = 30
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
    
    def _draw_face_mesh(self, painter: QPainter, cx: float, cy: float,
                        face_width: float, face_height: float):
        """Draw the low-poly face outline mesh."""
        # Face outline vertices (low-poly style)
        vertices = self._get_face_vertices(cx, cy, face_width, face_height)

        # Apply activation level to opacity
        base_glow_opacity = 0.3 * self._activation_level
        base_opacity = 0.3 + (0.7 * self._activation_level)  # 0.3 to 1.0

        # Draw mesh edges with glow effect
        glow_pen = QPen(self.GLOW_COLOR, 4)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        painter.setOpacity(base_glow_opacity)

        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            painter.drawLine(QPointF(*p1), QPointF(*p2))

        # Draw main edges
        painter.setOpacity(base_opacity)
        main_pen = QPen(self.PRIMARY_COLOR, 2)
        main_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(main_pen)

        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            painter.drawLine(QPointF(*p1), QPointF(*p2))

        # Draw vertices as glowing points
        for i, (vx, vy) in enumerate(vertices):
            jx, jy = self._get_jitter(i, 1.5)
            self._draw_vertex_glow(painter, vx + jx, vy + jy, self._activation_level)
    
    def _get_face_vertices(self, cx: float, cy: float, 
                           face_width: float, face_height: float) -> List[Tuple[float, float]]:
        """Generate vertices for the face outline polygon."""
        hw = face_width / 2
        hh = face_height / 2
        
        # Low-poly face shape (10 vertices)
        vertices = [
            (cx, cy - hh),  # Top
            (cx + hw * 0.5, cy - hh * 0.85),  # Top right
            (cx + hw * 0.8, cy - hh * 0.5),  # Upper right
            (cx + hw, cy - hh * 0.1),  # Mid right upper
            (cx + hw * 0.9, cy + hh * 0.3),  # Mid right
            (cx + hw * 0.6, cy + hh * 0.7),  # Lower right
            (cx + hw * 0.3, cy + hh * 0.9),  # Chin right
            (cx, cy + hh),  # Chin
            (cx - hw * 0.3, cy + hh * 0.9),  # Chin left
            (cx - hw * 0.6, cy + hh * 0.7),  # Lower left
            (cx - hw * 0.9, cy + hh * 0.3),  # Mid left
            (cx - hw, cy - hh * 0.1),  # Mid left upper
            (cx - hw * 0.8, cy - hh * 0.5),  # Upper left
            (cx - hw * 0.5, cy - hh * 0.85),  # Top left
        ]
        
        return vertices
    
    def _draw_vertex_glow(self, painter: QPainter, x: float, y: float, activation: float = 1.0):
        """Draw a glowing vertex point."""
        # Outer glow (scaled by activation)
        alpha = int(200 * activation)
        gradient = QRadialGradient(x, y, 8)
        gradient.setColorAt(0, QColor(251, 191, 36, alpha))
        gradient.setColorAt(1, QColor(251, 191, 36, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(activation)
        painter.drawEllipse(QPointF(x, y), 8, 8)

        # Core
        painter.setBrush(self.PRIMARY_COLOR)
        painter.drawEllipse(QPointF(x, y), 3, 3)
    
    def _draw_eyes(self, painter: QPainter, cx: float, cy: float,
                   face_width: float, face_height: float):
        """Draw the geometric eyes with expression-based shapes."""
        eye_y = cy - face_height * 0.15
        eye_spacing = face_width * 0.25
        eye_size = face_width * 0.12

        # Calculate blink factor (0 = open, 1 = closed)
        blink_factor = 0.0

        # If asleep (low activation), force eyes closed
        if self._activation_level < 0.5:
            blink_factor = 1.0  # Fully closed
        elif self._is_blinking:
            # Smooth blink curve (close then open)
            if self._blink_progress < 0.5:
                blink_factor = self._blink_progress * 2
            else:
                blink_factor = 1.0 - (self._blink_progress - 0.5) * 2

        # Add yawn factor (eyes close during yawn) - only when awake
        if self._activation_level >= 0.5:
            yawn_factor = self._yawn_progress * 0.7  # Partial close, not full
            blink_factor = max(blink_factor, yawn_factor)

        # Draw left eye
        self._draw_eye(painter, cx - eye_spacing, eye_y, eye_size, blink_factor, is_left=True)

        # Draw right eye
        self._draw_eye(painter, cx + eye_spacing, eye_y, eye_size, blink_factor, is_left=False)
    
    def _draw_eye(self, painter: QPainter, ex: float, ey: float,
                  size: float, blink_factor: float, is_left: bool):
        """Draw a single geometric eye."""
        # Expression-based eye shape modifications
        height_mult = 1.0
        y_offset = 0.0

        if self._expression == Expression.HAPPY:
            height_mult = 0.6  # Squinted happy eyes
            y_offset = -size * 0.1
        elif self._expression == Expression.SAD:
            height_mult = 0.8
            y_offset = size * 0.1
        elif self._expression == Expression.SURPRISED:
            height_mult = 1.3  # Wide eyes
        elif self._expression == Expression.CURIOUS:
            # One eyebrow raised
            if is_left:
                y_offset = -size * 0.15
        elif self._expression == Expression.THINKING:
            # Looking up/to the side
            y_offset = -size * 0.1

        # Apply blink
        height_mult *= (1.0 - blink_factor * 0.9)

        ey += y_offset

        # Eye shape - hexagonal for geometric look
        eye_height = size * height_mult

        # Apply activation level to glow
        glow_alpha = int(100 * self._activation_level)
        glow_gradient = QRadialGradient(ex, ey, size * 1.5)
        glow_gradient.setColorAt(0, QColor(251, 191, 36, glow_alpha))
        glow_gradient.setColorAt(1, QColor(251, 191, 36, 0))
        painter.setBrush(glow_gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(self._activation_level)
        painter.drawEllipse(QPointF(ex, ey), size * 1.5, size * 1.5)

        # Draw eye outline (diamond/hexagon shape)
        eye_path = QPainterPath()

        if self._expression == Expression.HAPPY:
            # Curved happy eye (arc shape)
            eye_path.moveTo(ex - size, ey)
            eye_path.quadTo(ex, ey - eye_height, ex + size, ey)
        else:
            # Diamond eye
            eye_path.moveTo(ex - size, ey)
            eye_path.lineTo(ex, ey - eye_height)
            eye_path.lineTo(ex + size, ey)
            eye_path.lineTo(ex, ey + eye_height * 0.5)
            eye_path.closeSubpath()

        # Draw outline with activation-adjusted opacity
        eye_opacity = 0.3 + (0.7 * self._activation_level)
        painter.setOpacity(eye_opacity)
        painter.setPen(QPen(self.PRIMARY_COLOR, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(eye_path)

        # Draw pupil or spinner (if not blinking and awake)
        if blink_factor < 0.7 and self._activation_level > 0.5:
            pupil_size = size * 0.3 * (1.0 - blink_factor)

            # Check if we should draw a spinner (thinking state)
            if self._jarvis_state == JarvisState.THINKING:
                # Draw spinning loader instead of pupil
                painter.setPen(QPen(self.PRIMARY_COLOR, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Draw 3 arc segments that rotate
                for i in range(3):
                    start_angle = (self._spinner_angle + i * 120) % 360
                    # Convert to Qt's angle format (1/16th of a degree)
                    qt_start = int(start_angle * 16)
                    qt_span = int(80 * 16)  # 80 degree arc

                    # Draw arc
                    painter.drawArc(
                        int(ex - pupil_size), int(ey - pupil_size),
                        int(pupil_size * 2), int(pupil_size * 2),
                        qt_start, qt_span
                    )
            else:
                # Draw normal pupil
                # Apply gaze offset to pupil position
                pupil_x = ex + self._gaze_x * 0.25  # Scale down gaze for subtle movement
                pupil_y = ey + self._gaze_y * 0.25

                # Clamp pupil within eye bounds
                max_offset = size * 0.5
                pupil_x = max(ex - max_offset, min(ex + max_offset, pupil_x))
                pupil_y = max(ey - max_offset * 0.6, min(ey + max_offset * 0.6, pupil_y))

                painter.setBrush(self.PRIMARY_COLOR)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(pupil_x, pupil_y), pupil_size, pupil_size)
    
    def _draw_mouth(self, painter: QPainter, cx: float, cy: float,
                    face_width: float, face_height: float):
        """Draw smooth continuous waveform mouth with speaking animation."""
        mouth_y = cy + face_height * 0.25
        mouth_width = face_width * 0.35
        max_wave_height = face_height * 0.08  # Maximum amplitude of waveform

        # Apply activation level to mouth opacity
        mouth_opacity = 0.3 + (0.7 * self._activation_level)

        # Generate waveform path using multiple sine waves for natural audio appearance
        wave_path = QPainterPath()
        num_points = 60  # Number of points for smooth curve

        # Start at left edge
        start_x = cx - mouth_width
        wave_path.moveTo(start_x, mouth_y)

        # Generate points along the waveform
        for i in range(num_points + 1):
            # Position along the mouth
            t = i / num_points
            x = start_x + (mouth_width * 2 * t)

            # Calculate waveform height using multiple sine waves for complexity
            # Main wave (low frequency, large amplitude)
            wave1 = math.sin((t * 3.0 + self._waveform_time) * self._waveform_frequency_base * 20)

            # Detail wave 1 (medium frequency)
            wave2 = math.sin((t * 8.0 + self._waveform_detail_offset) * 0.5) * 0.4

            # Detail wave 2 (high frequency, small amplitude for texture)
            wave3 = math.sin((t * 15.0 + self._waveform_time * 2) * 0.3) * 0.2

            # Combine waves with weighted sum
            combined_wave = (wave1 + wave2 + wave3) / 1.6

            # Apply amplitude envelope (less amplitude at edges)
            edge_factor = 1.0 - abs(t - 0.5) * 0.5  # Tapers at edges
            y = mouth_y + (combined_wave * max_wave_height * self._waveform_amplitude * edge_factor)

            wave_path.lineTo(x, y)

        # Draw glow effect
        painter.setOpacity(mouth_opacity * 0.25)
        glow_pen = QPen(self.GLOW_COLOR, 4)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(glow_pen)
        painter.drawPath(wave_path)

        # Draw main waveform line
        painter.setOpacity(mouth_opacity)
        main_pen = QPen(self.PRIMARY_COLOR, 2.5)
        main_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        main_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(main_pen)
        painter.drawPath(wave_path)

        # Draw endpoint vertices
        painter.setOpacity(1.0)
        jx1, jy1 = self._get_jitter(100, 1.5)
        jx2, jy2 = self._get_jitter(101, 1.5)
        self._draw_vertex_glow(painter, cx - mouth_width + jx1, mouth_y + jy1, self._activation_level)
        self._draw_vertex_glow(painter, cx + mouth_width + jx2, mouth_y + jy2, self._activation_level)
    
    def _draw_accent_lines(self, painter: QPainter, cx: float, cy: float,
                           face_width: float, face_height: float):
        """Draw decorative accent lines for the futuristic look."""
        # Apply activation level to accent lines
        accent_opacity = 0.5 * self._activation_level

        # Cheekbone lines
        painter.setPen(QPen(self.SECONDARY_COLOR, 1))
        painter.setOpacity(accent_opacity)

        cheek_y = cy + face_height * 0.05
        cheek_length = face_width * 0.15

        # Left cheekbone
        painter.drawLine(
            QPointF(cx - face_width * 0.35, cheek_y),
            QPointF(cx - face_width * 0.35 + cheek_length, cheek_y + cheek_length * 0.3)
        )

        # Right cheekbone
        painter.drawLine(
            QPointF(cx + face_width * 0.35, cheek_y),
            QPointF(cx + face_width * 0.35 - cheek_length, cheek_y + cheek_length * 0.3)
        )

        # Forehead lines (expression-dependent)
        if self._expression in [Expression.SURPRISED, Expression.CONCERNED]:
            forehead_y = cy - face_height * 0.35
            line_width = face_width * 0.2

            painter.drawLine(
                QPointF(cx - line_width, forehead_y),
                QPointF(cx + line_width, forehead_y)
            )

        painter.setOpacity(1.0)


class FaceWindow(QWidget):
    """A standalone window containing the Jarvis face."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Jarvis")
        self.setMinimumSize(320, 420)
        self.resize(350, 450)
        
        # Set window flags for floating window
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # Dark background
        self.setStyleSheet("background-color: #0a0a0a;")
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Face widget
        self.face = LowPolyFaceWidget()
        layout.addWidget(self.face)
    
    def set_expression(self, expression: Expression):
        """Set the face expression."""
        self.face.set_expression(expression)

