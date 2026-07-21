"""Unified turn foundation for Cora — shared by text and voice input.

Phase 1 of the unified-local-voice-chat integration. These are pure, dependency
-free data types: a typed ``UserMessage`` (from either a typed chat box or a
transcribed voice utterance), an ``AssistantMessage``, a per-turn ``TurnContext``
carrying identity/cancellation/provider selection, and a ``TurnResult`` describing
the outcome.

Design invariants (enforced here, not wired into runtime yet):
  * A turn is created by exactly one factory (``new_turn``) that mints a single
    ``turn_id`` and ``correlation_id`` and one inbound ``UserMessage``.
  * ``source`` is only ``text`` or ``voice`` — the same core consumes both.
  * The TTS status is a *secondary* output: it never controls the semantic
    status of a turn (a failed/late/muted TTS does not fail a completed turn).
  * No I/O, no network, no globals, no external dependencies — stdlib only.

Everything is JSON-serialisable via ``to_dict()`` so a turn can be logged,
sent over an event bus, or diffed in tests without special handling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "TurnSource",
    "ConversationState",
    "AssistantStatus",
    "VerificationStatus",
    "TtsStatus",
    "MemoryStatus",
    "TurnStatus",
    "StructuredError",
    "CancellationToken",
    "UserMessage",
    "AssistantMessage",
    "TurnContext",
    "TurnResult",
    "Turn",
    "new_turn",
    "new_user_message",
    "new_assistant_message",
]


# ---------------------------------------------------------------------------
# Enums — string-valued so they serialise predictably (matches memory/learning)
# ---------------------------------------------------------------------------

class TurnSource(str, Enum):
    """Where the inbound message came from. The core treats both identically."""

    TEXT = "text"
    VOICE = "voice"


class ConversationState(str, Enum):
    """Coarse per-turn lifecycle state (explicit transitions come in later phases)."""

    CREATED = "created"
    PERCEIVING = "perceiving"
    THINKING = "thinking"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESPONDING = "responding"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AssistantStatus(str, Enum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"


class VerificationStatus(str, Enum):
    """Result of the (future) ResultVerifier — success is claimed only when VERIFIED."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


class TtsStatus(str, Enum):
    """Secondary output channel status. Never gates the semantic turn status."""

    IDLE = "idle"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SKIPPED = "skipped"  # speak_response was False, or muted


class MemoryStatus(str, Enum):
    PENDING = "pending"
    WRITTEN = "written"
    SKIPPED = "skipped"
    FAILED = "failed"


class TurnStatus(str, Enum):
    """Semantic outcome of a whole turn."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _to_serialisable(value: Any) -> Any:
    """Recursively coerce a value into JSON-friendly primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        # Note: str/Enum members are ``str`` instances; return their plain value.
        if isinstance(value, Enum):
            return value.value
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {f.name: _to_serialisable(getattr(value, f.name)) for f in fields(value)}
    return value


class _Serialisable:
    """Mixin giving every dataclass a predictable ``to_dict()``."""

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _to_serialisable(getattr(self, f.name)) for f in fields(self)}


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------

@dataclass
class StructuredError(_Serialisable):
    """An error carried as data, so it never destroys the surrounding context."""

    kind: str
    message: str
    detail: Optional[str] = None


@dataclass
class CancellationToken(_Serialisable):
    """Minimal cooperative-cancellation flag (barge-in / Stop wire it up later)."""

    cancelled: bool = False
    reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        self.cancelled = True
        if reason is not None:
            self.reason = reason

    def __bool__(self) -> bool:  # ``if token:`` reads as "is cancelled"
        return self.cancelled


# ---------------------------------------------------------------------------
# Identity / time helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """A fresh, locally-generated unique id (no network, no coordination)."""
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    """Timezone-aware UTC timestamp as ISO-8601 (matches memory/graph style)."""
    return datetime.now(timezone.utc).isoformat()


def _normalise_language(language: Optional[str]) -> Optional[str]:
    """Trim + lowercase a language code, preserving codes like ``ro``.

    Returns ``None`` for empty/blank input. Never rewrites the code itself —
    ``ro`` stays ``ro`` — it only strips whitespace and normalises case.
    """
    if language is None:
        return None
    normalised = str(language).strip().lower()
    return normalised or None


def _coerce_source(source: Any) -> TurnSource:
    """Coerce a raw value into a ``TurnSource``; raise ValueError if invalid."""
    if isinstance(source, TurnSource):
        return source
    # TurnSource("text") / TurnSource("voice") validate; anything else raises.
    return TurnSource(str(source).strip().lower())


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserMessage(_Serialisable):
    """An immutable inbound message — typed or transcribed — entering the core."""

    message_id: str
    conversation_id: str
    turn_id: str
    source: TurnSource
    text: str
    language: Optional[str]
    created_at: str
    transcript_confidence: Optional[float] = None
    audio_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # frozen dataclass: normalise/validate via object.__setattr__.
        object.__setattr__(self, "source", _coerce_source(self.source))
        object.__setattr__(self, "language", _normalise_language(self.language))
        if self.text is None or not str(self.text).strip():
            raise ValueError("UserMessage.text must be non-empty (empty/whitespace rejected)")


@dataclass
class AssistantMessage(_Serialisable):
    """The assistant's reply for a turn. Mutable as the turn progresses."""

    message_id: str
    conversation_id: str
    turn_id: str
    text: str = ""
    status: AssistantStatus = AssistantStatus.PENDING
    created_at: str = field(default_factory=_utc_now_iso)
    tool_results: List[Any] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    tts_status: TtsStatus = TtsStatus.IDLE
    error: Optional[StructuredError] = None


# ---------------------------------------------------------------------------
# Turn context + result
# ---------------------------------------------------------------------------

@dataclass
class TurnContext(_Serialisable):
    """Per-turn coordination state: identity, cancellation, provider selection."""

    conversation_id: str
    turn_id: str
    correlation_id: str
    source: TurnSource
    created_at: str = field(default_factory=_utc_now_iso)
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    selected_stt_provider: Optional[str] = None
    selected_tts_provider: Optional[str] = None
    speak_response: bool = True
    state: ConversationState = ConversationState.CREATED
    latency_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TurnResult(_Serialisable):
    """The outcome of a turn. ``status`` is semantic and independent of TTS."""

    status: TurnStatus
    assistant_message: Optional[AssistantMessage] = None
    tool_results: List[Any] = field(default_factory=list)
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    memory_status: MemoryStatus = MemoryStatus.PENDING
    tts_status: TtsStatus = TtsStatus.IDLE
    structured_error: Optional[StructuredError] = None

    def __post_init__(self) -> None:
        # Coerce/validate status — invalid statuses are rejected here.
        if isinstance(self.status, TurnStatus):
            return
        self.status = TurnStatus(str(self.status).strip().lower())

    @property
    def is_success(self) -> bool:
        """Semantic success — reflects ``status`` only, never ``tts_status``."""
        return self.status == TurnStatus.COMPLETED


@dataclass(frozen=True)
class Turn(_Serialisable):
    """Aggregate returned by ``new_turn``: the inbound message plus its context."""

    context: TurnContext
    user_message: UserMessage


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def new_user_message(
    source: Any,
    text: str,
    language: Optional[str],
    conversation_id: str,
    turn_id: str,
    *,
    message_id: Optional[str] = None,
    transcript_confidence: Optional[float] = None,
    audio_metadata: Optional[Dict[str, Any]] = None,
) -> UserMessage:
    """Build a validated, immutable ``UserMessage`` bound to an existing turn."""
    return UserMessage(
        message_id=message_id or _new_id(),
        conversation_id=conversation_id,
        turn_id=turn_id,
        source=_coerce_source(source),
        text=text,
        language=language,
        created_at=_utc_now_iso(),
        transcript_confidence=transcript_confidence,
        audio_metadata=audio_metadata,
    )


def new_assistant_message(context: TurnContext) -> AssistantMessage:
    """Create an empty assistant message wired to a turn's identity."""
    return AssistantMessage(
        message_id=_new_id(),
        conversation_id=context.conversation_id,
        turn_id=context.turn_id,
    )


def new_turn(
    source: Any,
    text: str,
    language: Optional[str],
    conversation_id: Optional[str] = None,
    speak_response: bool = True,
    *,
    transcript_confidence: Optional[float] = None,
    audio_metadata: Optional[Dict[str, Any]] = None,
) -> Turn:
    """Create a new turn from a text or voice input.

    Mints a single ``turn_id`` and ``correlation_id`` and one inbound
    ``UserMessage``. A ``conversation_id`` is preserved when supplied, otherwise
    generated. ``speak_response`` applies to both text and voice inputs and does
    not depend on the source.

    Raises ``ValueError`` for an invalid ``source`` or empty ``text``.
    """
    turn_source = _coerce_source(source)  # validates before minting ids
    conv_id = conversation_id or _new_id()
    turn_id = _new_id()
    correlation_id = _new_id()

    user_message = new_user_message(
        source=turn_source,
        text=text,
        language=language,
        conversation_id=conv_id,
        turn_id=turn_id,
        transcript_confidence=transcript_confidence,
        audio_metadata=audio_metadata,
    )

    context = TurnContext(
        conversation_id=conv_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        source=turn_source,
        speak_response=speak_response,
    )

    return Turn(context=context, user_message=user_message)
