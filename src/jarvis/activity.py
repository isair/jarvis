"""Shared assistant activity state without desktop or GUI dependencies."""

from dataclasses import dataclass
from enum import Enum
import os
import tempfile
import threading
from typing import Optional

from .debug import debug_log


class JarvisState(Enum):
    """Assistant activity shared with local observers."""
    ASLEEP = "asleep"          # Daemon not started yet
    IDLE = "idle"              # Awake and ready, waiting for wake word
    LISTENING = "listening"    # Actively listening (collecting or hot window)
    THINKING = "thinking"      # Processing query
    SPEAKING = "speaking"      # Speaking response
    DICTATING = "dictating"    # Hold-to-dictate recording active
    DICTATION_PROCESSING = "dictation_processing"  # Transcribing & pasting captured dictation


def _get_jarvis_state_file() -> str:
    """Get the path to the Jarvis state file."""
    return os.path.join(tempfile.gettempdir(), "jarvis_state")


@dataclass(frozen=True, eq=False)
class _StateOwnership:
    previous: JarvisState
    publication: tuple[int, int, int, int]


def _file_identity(info) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size


class JarvisStateManager:
    """File-backed activity shared by in-process and subprocess observers."""

    def __init__(self):
        self._state = JarvisState.ASLEEP  # Start asleep
        self._state_lock = threading.Lock()
        self._owner: Optional[_StateOwnership] = None
        self._state_file = _get_jarvis_state_file()
        # Always start fresh in ASLEEP state on app launch
        # (state file is for cross-process communication during a session,
        # not for persisting state across app restarts)
        self._write_state(JarvisState.ASLEEP)

    def _published_state(self) -> Optional[tuple[JarvisState, tuple[int, int, int, int]]]:
        try:
            with open(self._state_file, 'r') as published:
                state = JarvisState(published.read().strip())
                identity = _file_identity(os.fstat(published.fileno()))
            return state, identity
        except (ValueError, OSError):
            return None

    @property
    def state(self) -> JarvisState:
        """Read a complete publication, with the last local state as fallback."""
        published = self._published_state()
        if published is not None:
            return published[0]
        with self._state_lock:
            return self._state

    def _write_state(self, state: JarvisState) -> Optional[tuple[int, int, int, int]]:
        """Publish a complete snapshot without following the shared path."""
        pending_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".jarvis-state-",
                dir=os.path.dirname(self._state_file) or ".", delete=False,
            ) as pending:
                pending_path = pending.name
                pending.write(state.value)
                pending.flush()
                identity = _file_identity(os.fstat(pending.fileno()))
            os.replace(pending_path, self._state_file)
            return identity
        except OSError:
            # Publication is best-effort; retain the previous shared snapshot.
            pass
        finally:
            if pending_path is not None:
                try:
                    os.unlink(pending_path)
                except OSError:
                    pass

    def set_state(self, state: JarvisState) -> None:
        """Set the Jarvis state (thread-safe, cross-process)."""
        with self._state_lock:
            self._owner = None
            self._state = state
            self._write_state(state)


    def acquire_state(self, state: JarvisState, *,
                      protected_states: set[JarvisState] | frozenset[JarvisState] = frozenset(),
                      restore_state: Optional[JarvisState] = None) -> Optional[_StateOwnership]:
        """Own a temporary local activity unless protected work is visible."""
        with self._state_lock:
            published = self._published_state()
            previous = published[0] if published is not None else self._state
            if previous in protected_states:
                debug_log(f"Temporary {state.value} activity suppressed by {previous.value}", "activity")
                return None
            if restore_state is not None:
                previous = restore_state
            elif self._owner is not None and published is not None and published[1] == self._owner.publication:
                previous = self._owner.previous
            identity = self._write_state(state)
            if identity is None:
                return None
            owner = _StateOwnership(previous, identity)
            self._state = state
            self._owner = owner
            debug_log(f"Temporary activity acquired: {state.value}", "activity")
            return owner

    def release_state(self, owner: Optional[_StateOwnership]) -> bool:
        """Restore only the current local owner's unchanged publication."""
        with self._state_lock:
            if owner is None or self._owner is not owner:
                return False
            self._owner = None
            published = self._published_state()
            if published is None or published[1] != owner.publication:
                debug_log("Activity publication changed before completion", "activity")
                return False
            identity = self._write_state(owner.previous)
            if identity is None:
                return False
            self._state = owner.previous
            debug_log(f"Owned activity released: {owner.previous.value}", "activity")
            return True


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

