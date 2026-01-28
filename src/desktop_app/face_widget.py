"""
Low-poly grid face widget for Jarvis with intelligent state management and organic idle behavior.

Features:
- Low-poly wireframe aesthetic with glowing effects
- State-specific visual indicators:
  * LISTENING: Expanding ring echoes of face outline (bell chime effect)
  * THINKING: Animated spinner pupils (3 rotating arcs)
  * SPEAKING: Animated hexagonal mouth
- Hexagonal mouth visualization:
  * Displays as a flat horizontal line when idle
  * Morphs into a 6-vertex hexagon when speaking starts
  * Each vertex animates independently based on simulated audio patterns
  * Smoothly transitions back to line when speaking ends
  * Semi-transparent fill with glowing edges for futuristic look
- Comprehensive state system (ASLEEP, IDLE, LISTENING, THINKING, SPEAKING)
- Smooth wake/sleep transitions with opacity-based activation
- Intelligent idle activity system (only active in IDLE state) that alternates between behaviors:
  * looking_around (33%) - Frequent eye movement scanning the environment
  * hovering (24%) - Gentle vertical floating motion
  * head_tilt (19%) - Subtle head rotation
  * deep_gaze (10%) - Focused staring at one point
  * stretch (7%) - Bigger movement with enhanced breathing
  * wink (4%) - Playful one-eye wink with slight head tilt
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
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient, QPolygonF
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

        # Hexagonal mouth animation
        self._mouth_hex_openness = 0.0  # 0.0 = flat line, 1.0 = fully open hexagon
        self._mouth_time = 0.0  # Time parameter for smooth animation
        self._mouth_intensity = 0.7  # Target intensity (changes occasionally)
        self._vertex_phases = [0.0] * 6  # Individual phase offsets for each vertex
        # Slightly different frequencies for each vertex for organic wobble
        self._vertex_frequencies = [0.08, 0.11, 0.09, 0.10, 0.12, 0.07]

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
        self._wink_progress = 0.0  # For winking activity
        self._wink_eye = "left"  # Which eye is winking

        # Thinking spinner animation
        self._spinner_angle = 0.0  # Rotation angle for thinking spinner

        # Listening animation - bell ring echoes
        self._listening_pulse_time = 0.0  # Time for spawning rings
        self._listening_rings: List[float] = []  # Active ring expansions (0.0 to 1.0)

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
            ("looking_around", 33),  # Most common - natural eye movement
            ("hovering", 24),        # Common - gentle floating
            ("head_tilt", 19),       # Common - subtle head rotation
            ("deep_gaze", 10),       # Occasional - stare at one spot
            ("stretch", 7),          # Occasional - bigger movement
            ("wink", 4),             # Rare - playful one-eye wink
            ("yawn", 3),             # Rare - eyes close briefly
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
            "wink": random.randint(30, 50),              # 1-1.7 seconds (quick wink)
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

        elif self._current_activity == "wink":
            # Playful one-eye wink
            if self._activity_timer == 0:  # Pick which eye at start
                self._wink_eye = random.choice(["left", "right"])

            progress = self._activity_timer / self._activity_duration
            if progress < 0.25:  # Close winking eye
                self._wink_progress += (1.0 - self._wink_progress) * 0.25
            elif progress > 0.6:  # Open winking eye
                self._wink_progress *= 0.8
            else:  # Hold the wink
                self._wink_progress += (1.0 - self._wink_progress) * 0.1

            # Slight head tilt toward winking eye for extra charm
            tilt_dir = -1 if self._wink_eye == "left" else 1
            self._head_tilt += (tilt_dir * 2.0 - self._head_tilt) * 0.08

            # Minimal other movements
            self._gaze_x *= 0.95
            self._gaze_y *= 0.95
            self._hover_offset *= 0.95
            self._stretch_intensity *= 0.9
            self._yawn_progress *= 0.9

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
            self._wink_progress *= 0.9

    def _decay_activity_animations(self):
        """Smoothly decay all activity animations when not idle."""
        self._gaze_x *= 0.92
        self._gaze_y *= 0.92
        self._hover_offset *= 0.92
        self._head_tilt *= 0.92
        self._stretch_intensity *= 0.85
        self._yawn_progress *= 0.85
        self._wink_progress *= 0.85
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

        # Listening animation - bell ring echoes
        if self._jarvis_state == JarvisState.LISTENING:
            self._listening_pulse_time += 1
            # Spawn a new ring every ~40 frames (~1.3 seconds)
            if self._listening_pulse_time >= 40:
                self._listening_pulse_time = 0
                self._listening_rings.append(0.0)  # Add new ring at expansion 0

            # Update existing rings (expand them)
            new_rings = []
            for ring in self._listening_rings:
                ring += 0.025  # Expansion speed
                if ring < 1.0:  # Keep if not fully expanded
                    new_rings.append(ring)
            self._listening_rings = new_rings
        else:
            # Fade out any remaining rings when not listening
            new_rings = []
            for ring in self._listening_rings:
                ring += 0.04  # Faster fadeout
                if ring < 1.0:
                    new_rings.append(ring)
            self._listening_rings = new_rings

        # Spinner animation (when thinking)
        if self._jarvis_state == JarvisState.THINKING:
            self._spinner_angle += 8.0  # Rotate 8 degrees per frame (~240 deg/sec)
            if self._spinner_angle >= 360:
                self._spinner_angle -= 360

        # Hexagonal mouth animation (when speaking)
        if self._jarvis_state == JarvisState.SPEAKING:
            # Advance time for smooth pulsing animation
            self._mouth_time += 0.15

            # Occasionally change target intensity to simulate speech patterns
            if random.random() < 0.03:  # ~3% chance per frame
                self._mouth_intensity = 0.5 + random.random() * 0.5  # 0.5 to 1.0

            # Calculate target openness using sine wave for smooth pulsing
            # Base openness + pulsing variation based on intensity
            pulse = math.sin(self._mouth_time * 2.5) * 0.15 + math.sin(self._mouth_time * 4.1) * 0.08
            target_openness = 0.4 + self._mouth_intensity * 0.4 + pulse * self._mouth_intensity

            # Smoothly interpolate toward target
            self._mouth_hex_openness += (target_openness - self._mouth_hex_openness) * 0.12

            # Animate vertex phases for organic wobble
            for i in range(6):
                self._vertex_phases[i] += self._vertex_frequencies[i]
        else:
            # Smoothly close back to flat line when not speaking
            self._mouth_hex_openness *= 0.88
            self._mouth_time *= 0.95  # Slow down time

            # Slow down vertex phases when not speaking
            for i in range(6):
                self._vertex_phases[i] += self._vertex_frequencies[i] * 0.2

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

        # Draw listening ring echoes (behind the face)
        self._draw_listening_rings(painter, cx, cy, face_width, face_height)

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

        # Calculate wink factors for each eye
        left_blink = blink_factor
        right_blink = blink_factor

        # Apply wink to just one eye
        if self._wink_progress > 0.05:
            if self._wink_eye == "left":
                left_blink = max(blink_factor, self._wink_progress)
            else:
                right_blink = max(blink_factor, self._wink_progress)

        # Draw left eye
        self._draw_eye(painter, cx - eye_spacing, eye_y, eye_size, left_blink, is_left=True)

        # Draw right eye
        self._draw_eye(painter, cx + eye_spacing, eye_y, eye_size, right_blink, is_left=False)
    
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
        """Draw hexagonal mouth that morphs from flat line to hexagon when speaking.

        Hexagon vertex layout when fully open:
                [1]      [2]
                 *--------*
                /          \\
           [0] *            * [3]
                \\          /
                 *--------*
                [4]      [5]

        When idle (openness = 0), all vertices collapse to a horizontal line.
        """
        mouth_y = cy + face_height * 0.25
        mouth_width = face_width * 0.35
        max_hex_height = face_height * 0.06  # Maximum vertical extent of hexagon

        # Apply activation level to mouth opacity
        mouth_opacity = 0.3 + (0.7 * self._activation_level)
        openness = self._mouth_hex_openness

        # Define hexagon vertices at full openness (relative to mouth center)
        # Format: (x_offset, y_offset_at_full_open)
        hex_full = [
            (-mouth_width, 0),                      # [0] Left corner
            (-mouth_width * 0.5, -max_hex_height),  # [1] Top left
            (mouth_width * 0.5, -max_hex_height),   # [2] Top right
            (mouth_width, 0),                       # [3] Right corner
            (mouth_width * 0.5, max_hex_height),    # [4] Bottom right
            (-mouth_width * 0.5, max_hex_height),   # [5] Bottom left
        ]

        # Line positions (all on center y-line, evenly spaced)
        hex_line = [
            (-mouth_width, 0),                      # [0] Left corner
            (-mouth_width * 0.5, 0),                # [1] Top left (on line)
            (mouth_width * 0.5, 0),                 # [2] Top right (on line)
            (mouth_width, 0),                       # [3] Right corner
            (mouth_width * 0.5, 0),                 # [4] Bottom right (on line)
            (-mouth_width * 0.5, 0),                # [5] Bottom left (on line)
        ]

        # Interpolate between line and hexagon based on openness, then add per-vertex wobble
        vertices = []
        for i in range(6):
            # Get line and full hex positions
            lx, ly = hex_line[i]
            fx, fy = hex_full[i]

            # Interpolate based on openness
            vx = lx + (fx - lx) * openness
            vy = ly + (fy - ly) * openness

            # Add per-vertex oscillation for organic movement (scaled by openness)
            wobble_x = math.sin(self._vertex_phases[i]) * 1.2 * openness
            wobble_y = math.cos(self._vertex_phases[i] * 1.3) * 1.0 * openness

            # Corners (0 and 3) wobble more horizontally
            if i in [0, 3]:
                wobble_x *= 1.3
                wobble_y *= 0.4

            # Add subtle jitter for organic feel
            jx, jy = self._get_jitter(110 + i, 0.5 * openness)

            vertices.append((cx + vx + wobble_x + jx, mouth_y + vy + wobble_y + jy))

        # Create polygon from vertices
        polygon = QPolygonF([QPointF(vx, vy) for vx, vy in vertices])

        # Draw filled hexagon with semi-transparent fill (only when open enough)
        if openness > 0.05:
            fill_color = QColor(self.PRIMARY_COLOR)
            fill_color.setAlpha(int(60 * openness * self._activation_level))
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setOpacity(0.7 * openness)
            painter.drawPolygon(polygon)

        # Draw glow outline
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setOpacity(mouth_opacity * 0.4)
        glow_pen = QPen(self.GLOW_COLOR, 4)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(glow_pen)
        painter.drawPolygon(polygon)

        # Draw main outline
        painter.setOpacity(mouth_opacity)
        main_pen = QPen(self.PRIMARY_COLOR, 2)
        main_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        main_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(main_pen)
        painter.drawPolygon(polygon)

        # Draw vertex glows at corners (vertices 0 and 3)
        painter.setOpacity(1.0)
        self._draw_vertex_glow(painter, vertices[0][0], vertices[0][1], self._activation_level)
        self._draw_vertex_glow(painter, vertices[3][0], vertices[3][1], self._activation_level)
    
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

    def _draw_listening_rings(self, painter: QPainter, cx: float, cy: float,
                                face_width: float, face_height: float):
        """Draw expanding ring echoes of the face outline (bell chime effect)."""
        if not self._listening_rings:
            return

        # Get base vertices
        base_vertices = self._get_face_vertices(cx, cy, face_width, face_height)

        for ring_progress in self._listening_rings:
            # Scale factor - rings expand outward from 1.0 to ~1.3
            scale = 1.0 + (ring_progress * 0.35)

            # Opacity fades as ring expands (starts at ~0.6, fades to 0)
            opacity = (1.0 - ring_progress) * 0.5 * self._activation_level

            if opacity < 0.02:
                continue

            # Scale vertices outward from center
            scaled_vertices = []
            for vx, vy in base_vertices:
                # Vector from center to vertex
                dx, dy = vx - cx, vy - cy
                # Scale outward
                new_x = cx + dx * scale
                new_y = cy + dy * scale
                scaled_vertices.append((new_x, new_y))

            # Draw the ring outline
            painter.setOpacity(opacity)
            ring_pen = QPen(self.PRIMARY_COLOR, 1.5)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw edges
            for i in range(len(scaled_vertices)):
                p1 = scaled_vertices[i]
                p2 = scaled_vertices[(i + 1) % len(scaled_vertices)]
                painter.drawLine(QPointF(*p1), QPointF(*p2))

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

