"""Media-player and announcement control for the Voice PE integration.

One controller per device over the published media player entity. Volume comes
from the device as an absolute value (the rotary encoder publishes no raw
wheel impulses), so the UI mirrors the published number and never re-echoes a
command. Announcements use the Voice Assistant announce RPC when the ``ANNOUNCE``
feature flag is present, otherwise the media player's announcement flag.
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:  # pragma: no cover - trivial import shim
    from aioesphomeapi import MediaPlayerCommand
except ImportError:  # pragma: no cover
    MediaPlayerCommand = None  # type: ignore[assignment]

#: State names of the stock media player, incl. ``announcing``.
MEDIA_STATE_NAMES = {
    0: "none",
    1: "idle",
    2: "playing",
    3: "paused",
    4: "announcing",
    5: "off",
    6: "on",
}

#: Echo suppression window for volume changes Jarvis itself performed.
VOLUME_ECHO_WINDOW_S = 1.5


class VoicePEMediaController:
    """Thin command layer over one ``MediaPlayerInfo`` key."""

    def __init__(self, client, key: Optional[int], mute_key: Optional[int] = None) -> None:
        self._client = client
        self._key = key
        self._mute_key = mute_key
        self.state: str = "none"
        self.volume: float = 0.0
        self.muted: bool = False
        self._last_local_change: float = 0.0

    # -- state -----------------------------------------------------------

    def update_state(self, state: Any) -> dict:
        """Store one ``MediaPlayerEntityState``; report whether it is new."""
        raw_state = int(getattr(state, "state", 0) or 0)
        name = MEDIA_STATE_NAMES.get(raw_state, "none")
        volume = float(getattr(state, "volume", 0.0) or 0.0)
        muted = bool(getattr(state, "muted", False))
        changed = (name, round(volume, 3), muted) != (
            self.state,
            round(self.volume, 3),
            self.muted,
        )
        self.state = name
        self.volume = volume
        self.muted = muted
        if changed and (time.time() - self._last_local_change) > VOLUME_ECHO_WINDOW_S:
            return {"source": "device", "state": name, "volume": volume, "muted": muted}
        if changed:
            return {"source": "jarvis", "state": name, "volume": volume, "muted": muted}
        return {}

    @property
    def volume_source(self) -> str:
        """`jarvis` while an issued command still covers the echo window."""
        if (time.time() - self._last_local_change) <= VOLUME_ECHO_WINDOW_S:
            return "jarvis"
        return "device"

    def is_active(self) -> bool:
        return self.state in {"playing", "announcing"}

    # -- commands --------------------------------------------------------

    def _command(self, **kwargs) -> None:
        if self._key is None:
            return
        self._client.media_player_command(key=self._key, **kwargs)

    def play_url(self, url: str) -> None:
        self._last_local_change = time.time()
        self._command(media_url=url, command=_cmd("PLAY"))

    def pause(self) -> None:
        self._last_local_change = time.time()
        self._command(command=_cmd("PAUSE"))

    def resume(self) -> None:
        self._last_local_change = time.time()
        self._command(command=_cmd("PLAY"))

    def stop(self) -> None:
        self._last_local_change = time.time()
        self._command(command=_cmd("STOP"))

    def set_volume(self, volume: float) -> None:
        self._last_local_change = time.time()
        self.volume = float(volume)
        self._command(volume=float(volume))

    def set_muted(self, muted: bool) -> None:
        """Soft mute through the published switch; hardware mute is read-only."""
        self._last_local_change = time.time()
        if self._mute_key is not None:
            self._client.switch_command(key=self._mute_key, state=bool(muted))
            return
        self._command(command=_cmd("UNMUTE" if not muted else "MUTE"))

    def supports_announce(self, capability_snapshot) -> bool:
        return bool(getattr(capability_snapshot, "announce", False))

    async def announce(
        self,
        media_id: str,
        *,
        text: str = "",
        timeout: float = 300.0,
        start_conversation: bool = False,
        preannounce_media_id: str = "",
    ) -> bool:
        """Announce and wait for the finished reply from the device."""
        response = await self._client.send_voice_assistant_announcement_await_response(
            media_id,
            timeout,
            text=text,
            start_conversation=start_conversation,
            preannounce_media_id=preannounce_media_id,
        )
        return bool(getattr(response, "success", False))

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "volume": round(self.volume, 3),
            "muted": self.muted,
            "volume_source": self.volume_source,
        }


def _cmd(name: str):
    """Resolve a media-player command enum member with a stable fallback."""
    if MediaPlayerCommand is not None:
        member = getattr(MediaPlayerCommand, name, None)
        if member is not None:
            return member
    return {
        "PLAY": 0,
        "PAUSE": 1,
        "STOP": 2,
        "MUTE": 3,
        "UNMUTE": 4,
    }[name]
