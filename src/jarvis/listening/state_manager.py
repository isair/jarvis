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
        """Capture whether hot window was active when voice input started."""
        self._was_hot_window_active_at_voice_start = (
            self.is_hot_window_active() and not self._should_expire_hot_window()
        )
        if self._was_hot_window_active_at_voice_start:
            debug_log("voice input started during active hot window", "state")
    
    def was_hot_window_active_at_voice_start(self) -> bool:
        """Check if hot window was active when current voice input started."""
        return self._was_hot_window_active_at_voice_start
    
    def clear_hot_window_voice_state(self) -> None:
        """Clear the hot window voice start state."""
        self._was_hot_window_active_at_voice_start = False
    
    def schedule_hot_window_activation(self, voice_debug: bool = False) -> None:
        """
        Schedule hot window activation after echo tolerance delay.
        
        Args:
            voice_debug: Whether to enable debug logging
        """
        def _delayed_activation():
            time.sleep(self.echo_tolerance)
            # Check if we should still activate (e.g., not interrupted)
            if not self._should_stop:
                with self._state_lock:
                    self._state = ListeningState.HOT_WINDOW
                    self._hot_window_start_time = time.time()
                
                debug_log(f"hot window activated for {self.hot_window_seconds}s (after {self.echo_tolerance}s echo delay)", "state")
                
                # Pretty output for non-debug mode
                if not voice_debug:
                    try:
                        print(f"👂 Listening for follow-up ({int(self.hot_window_seconds)}s)...")
                    except Exception:
                        pass
        
        # Start delayed activation in a separate thread
        activation_thread = threading.Thread(target=_delayed_activation, daemon=True)
        activation_thread.start()
    
    def _should_expire_hot_window(self) -> bool:
        """Check if hot window should expire due to timeout."""
        if not self.is_hot_window_active():
            return False
        current_time = time.time()
        return (current_time - self._hot_window_start_time) >= self.hot_window_seconds
    
    def check_hot_window_expiry(self, voice_debug: bool = False) -> bool:
        """
        Check and handle hot window expiry.
        
        Args:
            voice_debug: Whether to enable debug logging
            
        Returns:
            True if hot window was expired
        """
        if self._should_expire_hot_window():
            with self._state_lock:
                self._state = ListeningState.WAKE_WORD
            
            debug_log("hot window expired", "state")
            
            # Pretty output for non-debug mode
            if not voice_debug:
                try:
                    print("💤 Returning to wake word mode\n")
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
        if self.is_hot_window_active():
            with self._state_lock:
                self._state = ListeningState.WAKE_WORD
            
            debug_log("hot window manually expired", "state")
            
            # Pretty output for non-debug mode
            if not voice_debug:
                try:
                    print("💤 Returning to wake word mode")
                except Exception:
                    pass
    
    def stop(self) -> None:
        """Stop the state manager and any background threads."""
        self._should_stop = True
        with self._state_lock:
            self._state = ListeningState.WAKE_WORD
