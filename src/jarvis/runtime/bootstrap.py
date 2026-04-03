"""
Bootstrap — assembles all Jarvis services into a :class:`~jarvis.runtime.service_container.ServiceContainer`.

Each service initialisation is graceful: failures mark the service as
DEGRADED or UNAVAILABLE in the health registry and execution continues.
This ensures Jarvis can start (with limited functionality) even when
optional dependencies are unavailable.

Entry point used by the daemon::

    from jarvis.runtime.bootstrap import build_service_container

    container = build_service_container(cfg)
    # container.health describes what started successfully
    container.stop_event.wait()
    container.shutdown_manager.shutdown()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..debug import debug_log
from .health import HealthRegistry, HealthStatus, ServiceName, configure as configure_health
from .service_container import ServiceContainer
from .shutdown_manager import ShutdownManager


def build_service_container(cfg) -> ServiceContainer:
    """
    Initialise all services and return a populated :class:`ServiceContainer`.

    Services are started in dependency order.  Each step is wrapped in a
    try/except so that a failure degrades individual services rather than
    aborting the whole startup.

    Args:
        cfg: :class:`~jarvis.config.Settings` instance.

    Returns:
        :class:`ServiceContainer` with health registry populated.
    """
    health = configure_health()
    container = ServiceContainer(cfg, health)

    debug_log("bootstrap: starting service initialisation", "runtime")

    _init_audit(container)
    _init_policy(container)
    _init_database(container)
    _init_tts(container)
    _init_mcp(container)
    _init_location(container)

    # Shutdown manager wires all services together
    manager = ShutdownManager(cfg, health)
    manager.register_db(container.db)
    manager.register_tts(container.tts)
    manager.register_audit_recorder(container.audit_recorder)
    container.shutdown_manager = manager

    debug_log(f"bootstrap: complete — health summary: {health.summary()}", "runtime")
    return container


# ---------------------------------------------------------------------------
# Individual service initialisers
# ---------------------------------------------------------------------------

def _init_audit(container: ServiceContainer) -> None:
    """Initialise the audit recorder."""
    health = container.health
    health.initialising(ServiceName.AUDIT)
    try:
        from ..audit.recorder import configure as configure_audit
        cfg = container.cfg
        audit_db_path = getattr(cfg, "audit_db_path", None)
        if not audit_db_path:
            # Default to a sibling file of the main DB
            main_db = getattr(cfg, "db_path", str(Path.home() / ".local/share/jarvis/jarvis.db"))
            audit_db_path = str(Path(main_db).parent / "audit.db")
        recorder = configure_audit(audit_db_path)
        container.audit_recorder = recorder
        health.ready(ServiceName.AUDIT, f"db={audit_db_path}")
    except Exception as exc:
        health.degraded(ServiceName.AUDIT, "audit db unavailable", error=str(exc))
        debug_log(f"bootstrap: audit init failed: {exc}", "runtime")


def _init_policy(container: ServiceContainer) -> None:
    """Initialise the policy engine."""
    health = container.health
    health.initialising(ServiceName.POLICY)
    try:
        from ..policy.approvals import ApprovalStore
        from ..policy.engine import configure as configure_policy
        cfg = container.cfg
        # Reuse audit db path for approval persistence when available
        audit_db_path: Optional[str] = None
        if container.audit_recorder is not None:
            try:
                audit_db_path = container.audit_recorder._db._db_path  # type: ignore[attr-defined]
            except Exception:
                pass
        store = ApprovalStore(db_path=audit_db_path)
        engine = configure_policy(cfg, store)
        container.policy_engine = engine
        container.approval_store = store
        health.ready(ServiceName.POLICY, f"mode={getattr(cfg, 'policy_mode', 'ask_destructive')}")
    except Exception as exc:
        health.degraded(ServiceName.POLICY, "policy engine unavailable", error=str(exc))
        debug_log(f"bootstrap: policy init failed: {exc}", "runtime")


def _init_database(container: ServiceContainer) -> None:
    """Initialise the main Jarvis database and dialogue memory."""
    health = container.health
    health.initialising(ServiceName.DATABASE)
    try:
        from ..memory.db import Database
        from ..memory.conversation import DialogueMemory
        cfg = container.cfg
        db = Database(cfg.db_path, sqlite_vss_path=cfg.sqlite_vss_path)
        container.db = db

        dialogue_memory = DialogueMemory(
            inactivity_timeout=cfg.dialogue_memory_timeout,
            max_interactions=20,
        )
        container.dialogue_memory = dialogue_memory

        health.ready(ServiceName.DATABASE, cfg.db_path)
        if container.shutdown_manager:
            container.shutdown_manager.register_db(db)
            container.shutdown_manager.register_dialogue_memory(dialogue_memory)
    except Exception as exc:
        health.unavailable(ServiceName.DATABASE, error=str(exc))
        debug_log(f"bootstrap: database init failed: {exc}", "runtime")


def _init_tts(container: ServiceContainer) -> None:
    """Initialise the TTS engine (optional — degrades gracefully)."""
    health = container.health
    health.initialising(ServiceName.TTS)
    try:
        from ..output.tts import create_tts_engine
        cfg = container.cfg
        if not getattr(cfg, "tts_enabled", True):
            health.unavailable(ServiceName.TTS, "disabled in config")
            return
        engine = create_tts_engine(
            engine=getattr(cfg, "tts_engine", "piper"),
            enabled=getattr(cfg, "tts_enabled", True),
            voice=getattr(cfg, "tts_voice", None),
            rate=getattr(cfg, "tts_rate", None),
            device=getattr(cfg, "tts_chatterbox_device", "cuda"),
            audio_prompt_path=getattr(cfg, "tts_chatterbox_audio_prompt", None),
            exaggeration=getattr(cfg, "tts_chatterbox_exaggeration", 0.5),
            cfg_weight=getattr(cfg, "tts_chatterbox_cfg_weight", 0.5),
            piper_model_path=getattr(cfg, "tts_piper_model_path", None),
            piper_speaker=getattr(cfg, "tts_piper_speaker", None),
            piper_length_scale=getattr(cfg, "tts_piper_length_scale", 1.0),
            piper_noise_scale=getattr(cfg, "tts_piper_noise_scale", 0.667),
            piper_noise_w=getattr(cfg, "tts_piper_noise_w", 0.8),
            piper_sentence_silence=getattr(cfg, "tts_piper_sentence_silence", 0.2),
        )
        container.tts = engine
        if engine is None:
            health.degraded(ServiceName.TTS, "engine returned None")
        else:
            health.ready(ServiceName.TTS, getattr(cfg, "tts_engine", "piper"))
        if container.shutdown_manager:
            container.shutdown_manager.register_tts(engine)
    except Exception as exc:
        health.degraded(ServiceName.TTS, "TTS unavailable — text-only mode", error=str(exc))
        debug_log(f"bootstrap: TTS init failed: {exc}", "runtime")


def _init_mcp(container: ServiceContainer) -> None:
    """Initialise MCP tool discovery (optional — degrades gracefully)."""
    health = container.health
    health.initialising(ServiceName.MCP)
    try:
        cfg = container.cfg
        mcps = getattr(cfg, "mcps", {})
        if not mcps:
            health.unavailable(ServiceName.MCP, "no MCP servers configured")
            return
        from ..tools.registry import initialize_mcp_tools
        discovered = initialize_mcp_tools(mcps, verbose=True)
        if discovered:
            health.ready(ServiceName.MCP, f"{len(discovered)} tools discovered")
        else:
            health.degraded(ServiceName.MCP, "no tools discovered from configured servers")
    except Exception as exc:
        health.degraded(ServiceName.MCP, "MCP discovery failed", error=str(exc))
        debug_log(f"bootstrap: MCP init failed: {exc}", "runtime")


def _init_location(container: ServiceContainer) -> None:
    """Probe location service availability."""
    health = container.health
    health.initialising(ServiceName.LOCATION)
    try:
        cfg = container.cfg
        if not getattr(cfg, "location_enabled", True):
            health.unavailable(ServiceName.LOCATION, "disabled in config")
            return
        from ..utils.location import is_location_available
        if is_location_available():
            health.ready(ServiceName.LOCATION)
        else:
            health.degraded(ServiceName.LOCATION, "location service unreachable")
    except Exception as exc:
        health.degraded(ServiceName.LOCATION, "location check failed", error=str(exc))
        debug_log(f"bootstrap: location init failed: {exc}", "runtime")
