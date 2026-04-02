"""
Service container — owns all Jarvis subsystem instances.

The container is the single source of truth for what is alive and what
is degraded.  The daemon entry point constructs a container, calls
``start()``, waits for ``stop_event``, then calls ``shutdown()``.

Each service initialisation is wrapped in a try/except so that failures
result in degraded mode rather than a crash.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..debug import debug_log
from .health import HealthRegistry, HealthStatus, ServiceName


class ServiceContainer:
    """
    Holds references to every live Jarvis service and coordinates their
    lifecycle through the :class:`~jarvis.runtime.health.HealthRegistry`.

    Attributes are set by :func:`~jarvis.runtime.bootstrap.build_service_container`.
    Access them via the typed properties below.
    """

    def __init__(self, cfg, health: HealthRegistry) -> None:
        self._cfg = cfg
        self._health = health
        self._lock = threading.Lock()

        # Core services (set during build)
        self._db: Optional[Any] = None
        self._dialogue_memory: Optional[Any] = None
        self._tts: Optional[Any] = None
        self._voice_listener: Optional[Any] = None
        self._policy_engine: Optional[Any] = None
        self._audit_recorder: Optional[Any] = None
        self._approval_store: Optional[Any] = None
        self._shutdown_manager: Optional[Any] = None

        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        return self._cfg

    @property
    def health(self) -> HealthRegistry:
        return self._health

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value) -> None:
        self._db = value

    @property
    def dialogue_memory(self):
        return self._dialogue_memory

    @dialogue_memory.setter
    def dialogue_memory(self, value) -> None:
        self._dialogue_memory = value

    @property
    def tts(self):
        return self._tts

    @tts.setter
    def tts(self, value) -> None:
        self._tts = value

    @property
    def voice_listener(self):
        return self._voice_listener

    @voice_listener.setter
    def voice_listener(self, value) -> None:
        self._voice_listener = value

    @property
    def policy_engine(self):
        return self._policy_engine

    @policy_engine.setter
    def policy_engine(self, value) -> None:
        self._policy_engine = value

    @property
    def audit_recorder(self):
        return self._audit_recorder

    @audit_recorder.setter
    def audit_recorder(self, value) -> None:
        self._audit_recorder = value

    @property
    def approval_store(self):
        return self._approval_store

    @approval_store.setter
    def approval_store(self, value) -> None:
        self._approval_store = value

    @property
    def shutdown_manager(self):
        return self._shutdown_manager

    @shutdown_manager.setter
    def shutdown_manager(self, value) -> None:
        self._shutdown_manager = value

    @property
    def stop_event(self) -> threading.Event:
        """Set this event to trigger daemon shutdown."""
        return self._stop_event

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the daemon to begin shutdown."""
        debug_log("service container: stop requested", "runtime")
        self._stop_event.set()

    def is_policy_available(self) -> bool:
        return self._policy_engine is not None and self._health.is_operational(ServiceName.POLICY)

    def is_mcp_available(self) -> bool:
        return self._health.is_operational(ServiceName.MCP)

    def is_tts_available(self) -> bool:
        return self._tts is not None and self._health.is_operational(ServiceName.TTS)

    def is_voice_available(self) -> bool:
        return self._voice_listener is not None and self._health.is_operational(ServiceName.VOICE)
