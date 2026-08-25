"""
Health registry — centralised service health tracking.

Each subsystem registers itself and publishes periodic status updates.
The daemon and desktop UI consume the registry to determine whether to
start in degraded mode and what to surface to the operator.

Design
------
* Thread-safe.
* Subsystems are identified by a canonical string name.
* Health states are one of: ``ready``, ``degraded``, ``unavailable``, ``initialising``.
* A ``detail`` string may carry a human-readable explanation.
* The registry is a module-level singleton initialised at daemon startup.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from ..debug import debug_log


# ---------------------------------------------------------------------------
# Enumerations and value types
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    """Operational status of a single subsystem."""
    INITIALISING = "initialising"
    """Subsystem is still starting up."""
    READY = "ready"
    """Subsystem is fully operational."""
    DEGRADED = "degraded"
    """Subsystem is partially operational — some features may be unavailable."""
    UNAVAILABLE = "unavailable"
    """Subsystem is not available — dependent features are disabled."""


# Well-known subsystem identifiers (callers may also register custom names).
class ServiceName:
    DATABASE    = "database"
    OLLAMA      = "ollama"
    WHISPER     = "whisper"
    MICROPHONE  = "microphone"
    TTS         = "tts"
    MCP         = "mcp"
    LOCATION    = "location"
    POLICY      = "policy"
    AUDIT       = "audit"
    VOICE       = "voice"


@dataclass
class ServiceHealth:
    """Snapshot of a single service's health at a point in time."""
    name: str
    status: HealthStatus = HealthStatus.INITIALISING
    detail: str = ""
    last_updated: float = field(default_factory=time.time)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class HealthRegistry:
    """
    Thread-safe registry for all Jarvis subsystem health states.

    Instantiate at daemon startup (or use the module-level singleton via
    :func:`get_registry`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._services: Dict[str, ServiceHealth] = {}
        self._listeners: List[Callable[[ServiceHealth], None]] = []

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def set(
        self,
        name: str,
        status: HealthStatus,
        detail: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Update the health state for *name*.

        Creates an entry if one does not yet exist.  Notifies registered listeners.
        """
        record = ServiceHealth(
            name=name,
            status=status,
            detail=detail,
            last_updated=time.time(),
            error=error,
        )
        with self._lock:
            self._services[name] = record
            listeners = list(self._listeners)

        debug_log(f"health: {name} → {status.value}" + (f" ({detail})" if detail else ""), "health")

        for listener in listeners:
            try:
                listener(record)
            except Exception:
                pass

    def ready(self, name: str, detail: str = "") -> None:
        """Convenience: mark *name* as READY."""
        self.set(name, HealthStatus.READY, detail)

    def degraded(self, name: str, detail: str = "", error: Optional[str] = None) -> None:
        """Convenience: mark *name* as DEGRADED."""
        self.set(name, HealthStatus.DEGRADED, detail, error)

    def unavailable(self, name: str, detail: str = "", error: Optional[str] = None) -> None:
        """Convenience: mark *name* as UNAVAILABLE."""
        self.set(name, HealthStatus.UNAVAILABLE, detail, error)

    def initialising(self, name: str, detail: str = "") -> None:
        """Convenience: mark *name* as INITIALISING."""
        self.set(name, HealthStatus.INITIALISING, detail)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ServiceHealth]:
        """Return the current health record for *name*, or ``None``."""
        with self._lock:
            return self._services.get(name)

    def is_ready(self, name: str) -> bool:
        """Return True only when *name* is READY."""
        record = self.get(name)
        return record is not None and record.status == HealthStatus.READY

    def is_operational(self, name: str) -> bool:
        """Return True when *name* is READY or DEGRADED."""
        record = self.get(name)
        return record is not None and record.status in (HealthStatus.READY, HealthStatus.DEGRADED)

    def all_statuses(self) -> Dict[str, ServiceHealth]:
        """Return a shallow copy of all registered service health records."""
        with self._lock:
            return dict(self._services)

    def summary(self) -> Dict[str, str]:
        """
        Return a simplified ``{name: status_value}`` dict suitable for
        serialisation or display.
        """
        with self._lock:
            return {name: h.status.value for name, h in self._services.items()}

    def has_critical_failures(self) -> bool:
        """
        Return True if any *critical* subsystem is UNAVAILABLE.

        Critical subsystems are: DATABASE, OLLAMA, WHISPER, MICROPHONE.
        """
        critical = {ServiceName.DATABASE, ServiceName.OLLAMA, ServiceName.WHISPER, ServiceName.MICROPHONE}
        with self._lock:
            for name, health in self._services.items():
                if name in critical and health.status == HealthStatus.UNAVAILABLE:
                    return True
        return False

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(self, callback: Callable[[ServiceHealth], None]) -> None:
        """Register a callback invoked whenever any service health changes."""
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[ServiceHealth], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[HealthRegistry] = None


def configure() -> HealthRegistry:
    """
    Create and store the module-level :class:`HealthRegistry`.

    Called once from the service container at daemon startup.
    """
    global _registry
    _registry = HealthRegistry()
    return _registry


def get_registry() -> Optional[HealthRegistry]:
    """Return the module-level registry, or ``None`` if not configured."""
    return _registry
