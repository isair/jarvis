"""Home Assistant Voice: Preview Edition integration (stock firmware).

Product mode: ``Stock Voice PE / push-to-talk + continued conversation``.
Jarvis is a direct Native API client on TCP 6053 and takes the satellite-side
role: onboarding, Noise PSK, discovery, Voice Assistant session protocol,
microphone ingress, streamed TTS, media player, LED ring and button events.

The existing ``aioesphomeapi`` package is the only protocol implementation;
the local module set adds the Jarvis-side glue. See ``voice_pe.spec.md``.
"""

from .models import (
    DEFAULT_API_PORT,
    DeviceState,
    SessionState,
    VoiceInputSession,
    VoicePEConfig,
)
from .manager import VoicePEManager
from .cli import handle as _handle_cli

_MANAGER: VoicePEManager | None = None


def start(settings, listener, tts_engine) -> VoicePEManager | None:
    """Create and start the manager; ``None`` when the feature is disabled."""
    global _MANAGER
    manager = VoicePEManager(settings, listener, tts_engine)
    if not manager.enabled:
        return None
    manager.start()
    _MANAGER = manager
    return manager


def stop() -> None:
    """Stop the running manager, if any."""
    global _MANAGER
    if _MANAGER is not None:
        _MANAGER.stop()
        _MANAGER = None


def get_manager() -> VoicePEManager | None:
    return _MANAGER


def mirror_local(text: str) -> int:
    """Hand one locally spoken line to the idle satellites too.

    Returns the number of satellites that received it, ``0`` when the
    integration is off. Non-blocking: every device posts its own coroutine onto
    the manager loop, so the caller's (listener or proactive) thread keeps going.
    """
    if _MANAGER is None:
        return 0
    try:
        return int(_MANAGER.mirror_local(text))
    except Exception:
        return 0


def run_cli(argv: list) -> int:
    """Entry point for ``jarvis voice-pe ...`` (hand-rolled argv style)."""
    from jarvis.config import load_settings

    settings = load_settings()
    transient = None
    if _MANAGER is None:
        # No live daemon: a short-lived manager owns one connection per
        # subcommand so status/announce/play/set-led still work standalone.
        transient = VoicePEManager(settings, None, None)
        if transient.enabled:
            transient.start()
    try:
        return _handle_cli(list(argv or []), settings, transient or _MANAGER)
    finally:
        if transient is not None:
            transient.stop()


def health_snapshot() -> dict:
    if _MANAGER is None:
        return {"enabled": False, "devices": [], "metrics": {}}
    return _MANAGER.health()


__all__ = [
    "DEFAULT_API_PORT",
    "DeviceState",
    "SessionState",
    "VoiceInputSession",
    "VoicePEConfig",
    "VoicePEManager",
    "get_manager",
    "health_snapshot",
    "mirror_local",
    "run_cli",
    "start",
    "stop",
]
