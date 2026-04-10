"""
Shutdown manager — coordinates orderly daemon shutdown.

Responsibilities
----------------
1. Stop the voice listener (sets stop flag and joins thread).
2. Stop the TTS engine.
3. Flush pending audit records.
4. Perform the diary update (with configurable timeout).
5. Close databases.
6. Report final health status.

The manager is agnostic to the desktop-app IPC layer; callbacks
for diary progress can be injected so the desktop layer can display
live update progress without coupling to this module.

Usage::

    manager = ShutdownManager(cfg, services)
    manager.add_diary_callbacks(on_token=..., on_complete=...)
    manager.shutdown(timeout_sec=60.0)
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from ..debug import debug_log
from .health import HealthRegistry, HealthStatus, ServiceName


class ShutdownManager:
    """
    Coordinates a graceful daemon shutdown.

    Args:
        cfg: Settings object.
        health: :class:`~jarvis.runtime.health.HealthRegistry` instance.
    """

    def __init__(self, cfg, health: Optional[HealthRegistry] = None) -> None:
        self._cfg = cfg
        self._health = health
        self._lock = threading.Lock()
        self._shutdown_complete = threading.Event()

        # Optional service references set by the service container
        self._voice_listener = None
        self._tts_engine = None
        self._dialogue_memory = None
        self._db = None
        self._audit_recorder = None

        # Optional callbacks for diary update progress (used by desktop app)
        self._on_token: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_chunks: Optional[Callable[[list], None]] = None
        self._on_complete: Optional[Callable[[bool], None]] = None

    # ------------------------------------------------------------------
    # Service registration
    # ------------------------------------------------------------------

    def register_voice_listener(self, listener) -> None:
        self._voice_listener = listener

    def register_tts(self, engine) -> None:
        self._tts_engine = engine

    def register_dialogue_memory(self, memory) -> None:
        self._dialogue_memory = memory

    def register_db(self, db) -> None:
        self._db = db

    def register_audit_recorder(self, recorder) -> None:
        self._audit_recorder = recorder

    def add_diary_callbacks(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_chunks: Optional[Callable[[list], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Register callbacks for diary update progress (used by desktop layer)."""
        self._on_token = on_token
        self._on_status = on_status
        self._on_chunks = on_chunks
        self._on_complete = on_complete

    # ------------------------------------------------------------------
    # Main shutdown entry point
    # ------------------------------------------------------------------

    def shutdown(self, timeout_sec: float = 60.0) -> None:
        """
        Perform a full coordinated shutdown within *timeout_sec* seconds.

        Steps are executed in order.  Each step catches its own exceptions
        so that a failure in one step does not prevent later steps from
        running.
        """
        debug_log("shutdown manager: beginning shutdown sequence", "shutdown")
        start = time.time()

        self._stop_voice_listener()
        self._stop_tts()
        self._flush_diary(timeout_remaining=max(0.0, timeout_sec - (time.time() - start)))
        self._flush_audit()
        self._close_db()

        self._shutdown_complete.set()
        debug_log(
            f"shutdown manager: complete in {(time.time() - start):.1f}s", "shutdown"
        )

    def wait_complete(self, timeout: float = 70.0) -> bool:
        """Block until shutdown is complete or *timeout* elapses."""
        return self._shutdown_complete.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Individual shutdown steps
    # ------------------------------------------------------------------

    def _stop_voice_listener(self) -> None:
        if not self._voice_listener:
            return
        try:
            debug_log("shutdown: stopping voice listener", "shutdown")
            if hasattr(self._voice_listener, "stop"):
                self._voice_listener.stop()
        except Exception as exc:
            debug_log(f"shutdown: voice listener stop error: {exc}", "shutdown")
        finally:
            if self._health:
                self._health.set(ServiceName.VOICE, HealthStatus.UNAVAILABLE, "stopped")

    def _stop_tts(self) -> None:
        if not self._tts_engine:
            return
        try:
            debug_log("shutdown: stopping TTS", "shutdown")
            if hasattr(self._tts_engine, "stop"):
                self._tts_engine.stop()
        except Exception as exc:
            debug_log(f"shutdown: TTS stop error: {exc}", "shutdown")
        finally:
            if self._health:
                self._health.set(ServiceName.TTS, HealthStatus.UNAVAILABLE, "stopped")

    def _flush_diary(self, timeout_remaining: float = 45.0) -> None:
        """
        Attempt a diary update with a bounded timeout.

        Falls back gracefully if the LLM is unavailable or the timeout elapses.
        """
        if not self._dialogue_memory or not self._db:
            debug_log("shutdown: no dialogue memory — skipping diary flush", "shutdown")
            if self._on_complete:
                try:
                    self._on_complete(False)
                except Exception:
                    pass
            return

        timeout = min(timeout_remaining, getattr(self._cfg, "shutdown_diary_timeout_sec", 5.0))

        debug_log(f"shutdown: flushing diary (timeout={timeout:.0f}s)", "shutdown")

        # Notify UI about pending chunks before blocking LLM call
        if self._on_chunks:
            try:
                pending = self._dialogue_memory.get_pending_chunks() if self._dialogue_memory else []
                self._on_chunks(pending)
            except Exception:
                pass
        if self._on_status:
            try:
                self._on_status("Writing diary entry…")
            except Exception:
                pass

        result = [False]
        done_event = threading.Event()

        def _do_flush():
            try:
                from ..memory.conversation import update_diary_from_dialogue_memory
                cfg = self._cfg
                update_diary_from_dialogue_memory(
                    db=self._db,
                    dialogue_memory=self._dialogue_memory,
                    ollama_base_url=getattr(cfg, "ollama_base_url", "http://localhost:11434"),
                    ollama_chat_model=getattr(cfg, "ollama_chat_model", ""),
                    ollama_embed_model=getattr(cfg, "ollama_embed_model", ""),
                    source_app="voice",
                    voice_debug=getattr(cfg, "voice_debug", False),
                    timeout_sec=timeout,
                    force=True,
                    on_token=self._on_token,
                )
                result[0] = True
                debug_log("shutdown: diary flush succeeded", "shutdown")
            except Exception as exc:
                debug_log(f"shutdown: diary flush failed: {exc}", "shutdown")
            finally:
                done_event.set()

        t = threading.Thread(target=_do_flush, daemon=True, name="diary-flush")
        t.start()
        completed = done_event.wait(timeout=timeout)
        if not completed:
            debug_log("shutdown: diary flush timed out — proceeding without full diary", "shutdown")

        if self._on_complete:
            try:
                self._on_complete(result[0])
            except Exception:
                pass

    def _flush_audit(self) -> None:
        """Close the audit recorder gracefully."""
        if not self._audit_recorder:
            return
        try:
            debug_log("shutdown: closing audit recorder", "shutdown")
            if hasattr(self._audit_recorder, "close"):
                self._audit_recorder.close()
        except Exception as exc:
            debug_log(f"shutdown: audit recorder close error: {exc}", "shutdown")

    def _close_db(self) -> None:
        if not self._db:
            return
        try:
            debug_log("shutdown: closing database", "shutdown")
            if hasattr(self._db, "close"):
                self._db.close()
        except Exception as exc:
            debug_log(f"shutdown: db close error: {exc}", "shutdown")
