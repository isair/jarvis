"""TTS arbiter for the modern chat UI (Phase 3A).

There is exactly ONE shared Piper engine with a single completion-callback slot
and a single FIFO worker (``PiperTTS``). When two producers (voice + chat) hand
it callbacks, the newer ``speak()`` overwrites the older slot and a global
``interrupt()`` stops whatever is playing — the cross-channel callback clobber
documented as finding 5a.

``TTSCoordinator`` wraps the engine so that:
  * each playback is stamped with a monotonic ``generation`` and a *wrapper*
    completion callback that fires the caller's callback ONLY while its
    generation is still current — so a superseded (clobbered) callback becomes a
    harmless no-op and can never finalise a newer playback;
  * ``interrupt`` is scoped by ``turn_id`` and ``channel`` — a chat Stop cannot
    silence a voice turn, and vice-versa;
  * "Speak OFF"/disabled/blank text returns ``SKIPPED`` without touching the
    engine, and an engine failure returns ``FAILED`` (the text reply still stands).

SAFETY NOTE (Phase 3A scope): only the CHAT channel is routed through this
coordinator; the live, owner-validated voice path in ``listener.py`` is left
byte-for-byte unchanged. To make the coordinator safe to run *alongside* an
unmanaged voice path, preemption keys on coordinator-*owned* playback
(``active_turn_id``), NOT on ``engine.is_speaking()`` — so a chat ``speak`` never
force-interrupts a voice utterance the coordinator does not own. Full voice
routing is designed and forward-compatible, but deferred to a separately
reviewed phase.

Locking: this module owns exactly one lock (L2). It NEVER acquires
``daemon._reply_lock`` (L1) and is never held across an engine call
(``speak``/``interrupt`` run outside the lock), so the L1->L2 order can never
form a cycle. See the design's deadlock proof.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

__all__ = ["TtsChannel", "SpeakStatus", "SpeakToken", "TTSCoordinator"]


class TtsChannel(str, Enum):
    VOICE = "voice"
    CHAT = "chat"


class SpeakStatus(str, Enum):
    SPEAKING = "speaking"
    SKIPPED = "skipped"   # Speak OFF / engine disabled / blank -> no TTS
    FAILED = "failed"     # engine.speak raised -> text reply still stands


@dataclass(frozen=True)
class SpeakToken:
    generation: Optional[int]   # None when SKIPPED / FAILED-before-playback
    status: SpeakStatus


class TTSCoordinator:
    """Single arbiter over the one shared engine (duck-typed: ``enabled``,
    ``speak(text, completion_callback, duration_callback)``, ``is_speaking()``,
    ``interrupt()``)."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lock = threading.RLock()
        self.active_turn_id: Optional[str] = None
        self.active_channel: Optional[TtsChannel] = None
        self._on_complete: Optional[Callable[[], None]] = None
        self._generation: int = 0  # monotonic token

    # -- queries -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        eng = self._engine
        return bool(eng is not None and getattr(eng, "enabled", False))

    def is_speaking(self) -> bool:
        eng = self._engine
        try:
            return bool(eng is not None and eng.is_speaking())
        except Exception:
            return False

    def owns_active_playback(self) -> bool:
        with self._lock:
            return self.active_turn_id is not None

    # -- speak -------------------------------------------------------------
    def speak(
        self,
        *,
        turn_id: str,
        channel: TtsChannel,
        text: str,
        on_complete: Optional[Callable[[], None]] = None,
        on_duration: Optional[Callable[[float], None]] = None,
        speak_enabled: bool = True,
    ) -> SpeakToken:
        eng = self._engine
        if (
            eng is None
            or not getattr(eng, "enabled", False)
            or not speak_enabled
            or not text
            or not text.strip()
        ):
            return SpeakToken(None, SpeakStatus.SKIPPED)

        # Do NOT overwrite a live playback this coordinator does not own (e.g. a
        # voice utterance on the shared Piper engine, which has a single
        # completion-callback slot). Speaking over it would clobber the other
        # channel's callback. Skip -> the chat text still shows, just unspoken.
        try:
            if eng.is_speaking() and self.active_turn_id is None:
                return SpeakToken(None, SpeakStatus.SKIPPED)
        except Exception:
            pass

        with self._lock:
            # NB: chat is single-flight, so by the time a new turn reaches speak()
            # the previous turn's audio has finished — there is nothing of ours to
            # preempt, so we never call eng.interrupt() here (removing that avoids
            # ever force-interrupting an unrelated/voice playback).
            self._generation += 1
            gen = self._generation
            self.active_turn_id = turn_id
            self.active_channel = channel
            self._on_complete = on_complete

            def _guarded_complete() -> None:  # runs on the Piper worker thread
                cb = None
                with self._lock:
                    if gen != self._generation:
                        return  # superseded -> stale callback is a no-op
                    cb = self._on_complete
                    self.active_turn_id = None
                    self.active_channel = None
                    self._on_complete = None
                if cb is not None:
                    cb()  # OUTSIDE the lock (callback may re-enter speak)

            # Enqueue UNDER the lock so a concurrent interrupt() (UI-thread Stop)
            # cannot slip between registration and enqueue and let audio leak past
            # the Stop. eng.speak() is non-blocking (queue put) and never re-enters
            # the coordinator or acquires L1, so holding L2 across it is deadlock-safe.
            try:
                eng.speak(text, completion_callback=_guarded_complete,
                          duration_callback=on_duration)
            except Exception:
                if gen == self._generation:
                    self.active_turn_id = None
                    self.active_channel = None
                    self._on_complete = None
                return SpeakToken(gen, SpeakStatus.FAILED)
        return SpeakToken(gen, SpeakStatus.SPEAKING)

    def release_if_owned(self, turn_id: str) -> bool:
        """Clear ownership of a turn's playback WITHOUT interrupting audio.

        Called when a turn's worker finishes, so a clobbered (never-fired)
        guarded callback cannot leave ``active_turn_id`` stranded — a stranded id
        would make a later chat turn's interrupt/Stop force-interrupt an unrelated
        (possibly voice) playback. Bumps the generation so any late guarded
        callback for this turn is neutralised."""
        with self._lock:
            if self.active_turn_id == turn_id:
                self._generation += 1
                self.active_turn_id = None
                self.active_channel = None
                self._on_complete = None
                return True
        return False

    # -- interrupt ---------------------------------------------------------
    def interrupt(
        self,
        *,
        turn_id: Optional[str] = None,
        channel: Optional[TtsChannel] = None,
    ) -> bool:
        """Stop the current owned playback if it matches the given identity.

        ``turn_id``/``channel`` are filters: a mismatch is a no-op (returns
        False) so a cross-turn or cross-channel Stop cannot kill the wrong audio.
        Bumping ``generation`` neutralises the pending guarded callback so the
        interrupted turn's ``on_complete`` never fires (matching Piper's
        ``not interrupted`` semantics)."""
        eng = self._engine
        with self._lock:
            if self.active_turn_id is None:
                return False
            if turn_id is not None and turn_id != self.active_turn_id:
                return False
            if channel is not None and channel != self.active_channel:
                return False
            self._generation += 1  # neutralise pending guarded callback
            self.active_turn_id = None
            self.active_channel = None
            self._on_complete = None
        if eng is not None:
            try:
                eng.interrupt()  # OUTSIDE _lock
            except Exception:
                pass
        return True
