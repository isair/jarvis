"""State management for listening modes (wake word, collection, hot window)."""

import time
import threading
from typing import Optional
from enum import Enum
from datetime import datetime

from ..debug import debug_log


class ListeningState(Enum):
    """Possible listening states."""
    WAKE_WORD = "wake_word"      # Waiting for wake word
    COLLECTING = "collecting"    # Accumulating query text
    HOT_WINDOW = "hot_window"    # Listening without wake word after TTS


class StateManager:
    """Manages listening state transitions and timing."""

    def __init__(self, hot_window_seconds: float = 6.0, echo_tolerance: float = 0.3,
                 voice_collect_seconds: float = 2.0, max_collect_seconds: float = 60.0):
        """
        Initialize state manager.

        Args:
            hot_window_seconds: Duration of hot window listening
            echo_tolerance: Delay before activating hot window (for echo suppression)
            voice_collect_seconds: Silence timeout for query collection
            max_collect_seconds: Maximum time to collect a single query
        """
        self.hot_window_seconds = hot_window_seconds
        self.echo_tolerance = echo_tolerance
        self.voice_collect_seconds = voice_collect_seconds
        self.max_collect_seconds = max_collect_seconds

        # Current state
        self._state = ListeningState.WAKE_WORD
        self._state_lock = threading.Lock()

        # Collection state
        self._pending_query: str = ""
        self._last_voice_time: float = 0.0
        self._collect_start_time: float = 0.0

        # Hot window state
        self._hot_window_start_time: float = 0.0
        self._was_hot_window_active_at_voice_start: bool = False

        # Timer-based hot window management
        self._hot_window_activation_timer: Optional[threading.Timer] = None
        self._hot_window_expiry_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._voice_debug: bool = False  # Cache for use in timer callbacks

        # Stop flag for background threads
        self._should_stop = False

    def get_state(self) -> ListeningState:
        """Get current listening state."""
        with self._state_lock:
            return self._state

    def is_collecting(self) -> bool:
        """Check if currently in collection mode."""
        return self.get_state() == ListeningState.COLLECTING

    def is_hot_window_active(self) -> bool:
        """Check if hot window is currently active."""
        return self.get_state() == ListeningState.HOT_WINDOW

    def start_collection(self, initial_text: str = "") -> None:
        """
        Start query collection mode.

        Args:
            initial_text: Optional initial text to seed the collection
        """
        with self._state_lock:
            self._state = ListeningState.COLLECTING
            self._pending_query = initial_text.strip()
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time

        start_time_str = datetime.fromtimestamp(self._collect_start_time).strftime('%H:%M:%S.%f')[:-3]
        debug_log(f"collection started at {start_time_str}: '{initial_text}'", "state")

        # Set face state to LISTENING
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            face_state_manager = get_jarvis_state()
            face_state_manager.set_state(JarvisState.LISTENING)
            debug_log("face state set to LISTENING (collection started)", "state")
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to set face state to LISTENING: {e}", "state")

    def add_to_collection(self, text: str) -> None:
        """
        Add text to current collection.

        Args:
            text: Text to append to pending query
        """
        if not self.is_collecting():
            return

        with self._state_lock:
            self._pending_query = (self._pending_query + " " + text).strip()
            self._last_voice_time = time.time()

        debug_log(f"added to collection: '{text}' -> '{self._pending_query}'", "state")

    def get_pending_query(self) -> str:
        """Get the current pending query text."""
        with self._state_lock:
            return self._pending_query

    def clear_collection(self) -> str:
        """
        Clear and return the current pending query.

        Returns:
            The query that was being collected
        """
        with self._state_lock:
            query = self._pending_query
            collect_start_time = self._collect_start_time
            self._pending_query = ""
            if self._state == ListeningState.COLLECTING:
                self._state = ListeningState.WAKE_WORD

        if query and collect_start_time > 0:
            end_time = time.time()
            duration = end_time - collect_start_time
            start_time_str = datetime.fromtimestamp(collect_start_time).strftime('%H:%M:%S.%f')[:-3]
            end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"collection cleared: '{query}' (started: {start_time_str}, ended: {end_time_str}, duration: {duration:.2f}s)", "state")
        else:
            debug_log(f"collection cleared: '{query}'", "state")

        # Note: Don't set face state here - it will be set to THINKING or ASLEEP by caller

        return query

    def check_collection_timeout(self) -> bool:
        """
        Check if collection should timeout due to silence or max duration.

        Returns:
            True if collection should be finalized
        """
        if not self.is_collecting():
            return False

        current_time = time.time()
        silence_timeout = current_time - self._last_voice_time >= self.voice_collect_seconds
        max_timeout = current_time - self._collect_start_time >= self.max_collect_seconds

        if silence_timeout or max_timeout:
            timeout_type = "silence" if silence_timeout else "max"

            end_time = time.time()
            duration = end_time - self._collect_start_time
            start_time_str = datetime.fromtimestamp(self._collect_start_time).strftime('%H:%M:%S.%f')[:-3]
            end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]

            debug_log(f"collection timeout ({timeout_type}): '{self._pending_query}' (started: {start_time_str}, ended: {end_time_str}, duration: {duration:.2f}s)", "state")
            return True

        return False

    def capture_hot_window_state_at_voice_start(self) -> None:
        """Capture whether hot window was active when voice input started.

        Note: We only check if the state is HOT_WINDOW, not whether the time
        has elapsed. This prevents a race condition where the user starts
        speaking near the end of the hot window - the state might not have
        been updated to WAKE_WORD yet even if the time limit just passed.
        The important thing is that the system was in hot window mode when
        the user started speaking.
        """
        with self._state_lock:
            self._was_hot_window_active_at_voice_start = self._state == ListeningState.HOT_WINDOW
        if self._was_hot_window_active_at_voice_start:
            debug_log("voice input started during active hot window", "state")

    def was_hot_window_active_at_voice_start(self) -> bool:
        """Check if hot window was active when current voice input started."""
        with self._state_lock:
            return self._was_hot_window_active_at_voice_start

    def clear_hot_window_voice_state(self) -> None:
        """Clear the hot window voice start state."""
        with self._state_lock:
            self._was_hot_window_active_at_voice_start = False

    def cancel_hot_window_activation(self) -> None:
        """Cancel any pending hot window activation timer.

        Call this when user starts a new query to prevent delayed activation
        from interfering with the current interaction.
        """
        with self._timer_lock:
            if self._hot_window_activation_timer is not None:
                self._hot_window_activation_timer.cancel()
                self._hot_window_activation_timer = None
                debug_log("cancelled pending hot window activation", "state")

    def _cancel_hot_window_expiry_timer(self) -> None:
        """Cancel the hot window expiry timer."""
        with self._timer_lock:
            if self._hot_window_expiry_timer is not None:
                self._hot_window_expiry_timer.cancel()
                self._hot_window_expiry_timer = None

    def _schedule_hot_window_expiry(self) -> None:
        """Schedule hot window expiry timer.

        This timer guarantees expiry will fire even if no audio is being processed.
        """
        self._cancel_hot_window_expiry_timer()

        def _expire():
            with self._state_lock:
                if self._state != ListeningState.HOT_WINDOW:
                    return
                self._state = ListeningState.WAKE_WORD

            expiry_time = time.time()
            duration = expiry_time - self._hot_window_start_time if self._hot_window_start_time > 0 else 0
            expiry_time_str = datetime.fromtimestamp(expiry_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"hot window expired (timer) at {expiry_time_str} after {duration:.2f}s", "state")

            # Set face state to IDLE
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window timer expiry)", "state")
            except ImportError:
                # Desktop app not available (headless mode)
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode\n", flush=True)
            except Exception:
                pass

        with self._timer_lock:
            self._hot_window_expiry_timer = threading.Timer(self.hot_window_seconds, _expire)
            self._hot_window_expiry_timer.daemon = True
            self._hot_window_expiry_timer.start()

        debug_log(f"scheduled hot window expiry in {self.hot_window_seconds}s", "state")

    def schedule_hot_window_activation(self, voice_debug: bool = False) -> None:
        """
        Schedule hot window activation after echo tolerance delay.

        Uses threading.Timer for reliable activation instead of daemon thread + sleep.

        Args:
            voice_debug: Whether to enable debug logging
        """
        schedule_time_str = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S.%f')[:-3]
        debug_log(f"scheduling hot window activation at {schedule_time_str} (delay={self.echo_tolerance}s, should_stop={self._should_stop})", "state")

        # Cancel any pending activation first
        self.cancel_hot_window_activation()

        # Cache voice_debug for use in timer callbacks
        self._voice_debug = voice_debug

        def _activate():
            # Check if we should still activate
            if self._should_stop:
                debug_log("hot window activation cancelled (should_stop=True)", "state")
                return

            with self._state_lock:
                # Don't overwrite COLLECTING state - user may have already started a new query
                if self._state == ListeningState.COLLECTING:
                    debug_log("hot window activation cancelled (already collecting)", "state")
                    return
                self._state = ListeningState.HOT_WINDOW
                self._hot_window_start_time = time.time()

            activation_time_str = datetime.fromtimestamp(self._hot_window_start_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"hot window activated at {activation_time_str} for {self.hot_window_seconds}s (after {self.echo_tolerance}s echo delay)", "state")

            # Set face state to LISTENING
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.LISTENING)
                debug_log("face state set to LISTENING (hot window activated)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to LISTENING: {e}", "state")

            # Always show user-facing output
            try:
                print(f"👂 Listening for follow-up ({int(self.hot_window_seconds)}s)...", flush=True)
            except Exception as e:
                debug_log(f"failed to print hot window message: {e}", "state")

            # Schedule the expiry timer now that hot window is active
            self._schedule_hot_window_expiry()

        # Use Timer for more reliable activation
        with self._timer_lock:
            self._hot_window_activation_timer = threading.Timer(self.echo_tolerance, _activate)
            self._hot_window_activation_timer.daemon = True
            self._hot_window_activation_timer.start()

        debug_log("hot window activation timer started", "state")

    def _should_expire_hot_window(self) -> bool:
        """Check if hot window should expire due to timeout.

        Note: With timer-based expiry, this is now mainly a fallback check.
        The timer should handle expiry automatically.
        """
        if not self.is_hot_window_active():
            return False
        current_time = time.time()
        return (current_time - self._hot_window_start_time) >= self.hot_window_seconds

    def check_hot_window_expiry(self, voice_debug: bool = False) -> bool:
        """
        Check and handle hot window expiry.

        Note: With timer-based expiry, this is now a fallback check.
        The timer should handle expiry automatically, but this method
        provides a synchronous check for the main audio processing loop.

        Args:
            voice_debug: Whether to enable debug logging

        Returns:
            True if hot window was expired
        """
        if self._should_expire_hot_window():
            # Cancel expiry timer since we're handling it here
            self._cancel_hot_window_expiry_timer()

            with self._state_lock:
                self._state = ListeningState.WAKE_WORD

            debug_log("hot window expired (poll)", "state")

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window poll expiry)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode\n", flush=True)
            except Exception:
                pass

            return True
        return False

    def expire_hot_window(self, voice_debug: bool = False) -> None:
        """
        Manually expire the hot window.

        Args:
            voice_debug: Whether to enable debug logging
        """
        # Cancel expiry timer since we're manually expiring
        self._cancel_hot_window_expiry_timer()

        if self.is_hot_window_active():
            with self._state_lock:
                self._state = ListeningState.WAKE_WORD

            debug_log("hot window manually expired", "state")

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window manually expired)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode", flush=True)
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the state manager and cancel all timers."""
        self._should_stop = True

        # Cancel all timers
        self.cancel_hot_window_activation()
        self._cancel_hot_window_expiry_timer()

        with self._state_lock:
            self._state = ListeningState.WAKE_WORD
