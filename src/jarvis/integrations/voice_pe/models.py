"""Data models and protocol constants for the ESPHome Voice PE transport.

Product mode of a stock retail device: ``Stock Voice PE / push-to-talk +
continued conversation``. The centre button opens the first wake-free
session and ``continue_conversation`` carries the follow-up turns.

Hard stock-firmware limits represented here and in the UI:
- the rotary encoder has no public raw-wheel entity, the authoritative
  value is the media-player volume the device publishes;
- the centrebutton single click is handled locally on the device;
- ``voice_assistant_leds`` is ``internal: true`` and is driven only through
  standard Voice Assistant events;
- ``STREAMING_MICROPHONE`` and ``STREAMING_RESPONSE`` are separate states, so
  turn taking is fast but not simultaneous mic+TTS barge-in.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Native API + audio constants (see the ESPHome ``api.proto`` reference)
# ---------------------------------------------------------------------------

DEFAULT_API_PORT = 6053

# TTS wire format for a device that declares SPEAKER | API_AUDIO.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SAMPLE_CHANNELS = 1
SAMPLES_PER_CHUNK = 512
PAYLOAD_BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * SAMPLE_WIDTH * SAMPLE_CHANNELS  # 1024

# The device-side speaker ring buffer is a fixed 512 ms and is kept near
# 384 ms (75% full) by the pacing rule in ``tts_stream``.
SPEAKER_RING_BUFFER_S = 0.512
SPEAKER_BUFFER_TARGET_S = 0.384

# Announcement wait + satellite configuration fetch budgets.
ANNOUNCEMENT_TIMEOUT_S = 300.0
VA_CONFIG_TIMEOUT_S = 5.0

# mDNS service type used by every ESPHome node.
ESPHOME_MDNS_TYPE = "_esphomelib._tcp.local."

# Feature bits from ``voice_assistant_feature_flags`` (API < 1.15) and from
# ``DeviceCapabilitiesResponse.voice_assistant`` (API >= 1.15).
FEATURE_VOICE_ASSISTANT = 1 << 0
FEATURE_SPEAKER = 1 << 1
FEATURE_API_AUDIO = 1 << 2
FEATURE_TIMERS = 1 << 3
FEATURE_ANNOUNCE = 1 << 4
FEATURE_START_CONVERSATION = 1 << 5
FEATURE_MULTI_CHANNEL_AUDIO = 1 << 6

FEATURE_NAMES: Dict[int, str] = {
    FEATURE_VOICE_ASSISTANT: "voice_assistant",
    FEATURE_SPEAKER: "speaker",
    FEATURE_API_AUDIO: "api_audio",
    FEATURE_TIMERS: "timers",
    FEATURE_ANNOUNCE: "announce",
    FEATURE_START_CONVERSATION: "start_conversation",
    FEATURE_MULTI_CHANNEL_AUDIO: "multi_channel_audio",
}

# Flags the device sends in ``VoiceAssistantRequest.flags``. Diagnostic only:
# with ``disable_wake_words=True`` the pipeline always starts at STT.
COMMAND_FLAG_USE_VAD = 1 << 0
COMMAND_FLAG_USE_WAKE_WORD = 1 << 1


def feature_list(flags: int) -> list[str]:
    """Expand a feature bitmask into stable names for UI and diagnostics."""
    return [name for bit, name in FEATURE_NAMES.items() if flags & bit]


def make_client(
    host: str,
    port: int,
    psk: Optional[str],
    *,
    device_name: Optional[str] = None,
    mac: Optional[str] = None,
):
    """One ``APIClient`` construction shape for every library release.

    ``password`` is a required positional in 38.x and a defaulted keyword in
    newer releases, so it is always passed positionally as ``None`` (the
    Native API dropped password auth in 2026.1.0 and Noise carries the PSK).
    """
    from aioesphomeapi import APIClient

    return APIClient(
        host,
        port,
        None,
        noise_psk=psk,
        expected_name=device_name,
        expected_mac=mac,
    )


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

#: Source tag on every item the listener queue receives. One microphone is
#: readable at a time, so each frame carries the source it came from.
AUDIO_SOURCE_LOCAL = "local"
AUDIO_SOURCE_VOICE_PE = "voice_pe"


#: Full identity of one microphone stream: which satellite, which connection of
#: it, and which run inside that connection. Two satellites that both number
#: their first run ``1`` stay distinguishable through ``device_id``.
StreamId = namedtuple("StreamId", "device_id connection_generation session_generation")

#: Identity of the local microphone, which has no satellite numbering at all.
LOCAL_STREAM = StreamId("", 0, 0)

#: One satellite microphone block with the identity of the stream it belongs
#: to. The listener drops a frame whose stream is no longer the open one, so
#: audio of a cancelled run or of another satellite cannot widen a newer
#: utterance.
#: ``channel`` is 0 (``data``/enhanced) or 1 (``data2``/raw), always passed
#: explicitly from the ``VoiceAssistantAudio`` callback. A missing channel is
#: a ``TypeError`` from the dataclass itself (no default). The local microphone
#: uses a different dataclass, ``LocalMicFrame``, so it cannot be confused
#: with Voice PE channel 0.
AUDIO_CHANNEL_ENHANCED = 0
AUDIO_CHANNEL_RAW = 1

#: Host AEC modes (``voice_pe_dsp_mode``). ``host_raw_aec`` is the production
#: default: channel 1 (``data2``/raw) goes through the native engine's own
#: AEC3 lane using the Windows loopback as far-end reference.
#: ``device_enhanced`` keeps channel 0 (XMOS) without host processing;
#: ``shadow_compare`` delivers channel 0 while lane B cleans channel 1 for
#: diagnostics only — the two signals are never mixed.
DSP_MODE_HOST_RAW_AEC = "host_raw_aec"
DSP_MODE_DEVICE_ENHANCED = "device_enhanced"
DSP_MODE_SHADOW_COMPARE = "shadow_compare"
DSP_MODES = (DSP_MODE_HOST_RAW_AEC, DSP_MODE_DEVICE_ENHANCED, DSP_MODE_SHADOW_COMPARE)


@dataclass(frozen=True, slots=True)
class SatelliteAudioFrame:
    """One PCM block on the satellite side; every field is required."""

    stream: StreamId
    source: str
    samples: bytes
    channel: int  # 0 or 1; missing or out-of-range is a TypeError from the dataclass.


@dataclass(frozen=True, slots=True)
class LocalMicFrame:
    """The local PC microphone's own frame type — never confused with channel 0."""

    samples: bytes
    source: str = AUDIO_SOURCE_LOCAL
    stream: StreamId = LOCAL_STREAM


#: Back-compatible aliases so the transport and the listener can import either
#: name — the same dataclass instance keeps the ``isinstance`` check simple.
AudioFrame = SatelliteAudioFrame


#: Per-packet metadata of one ``VoiceAssistantAudio`` callback, kept per
#: ``(StreamId, channel)`` so the two channels never mix inside one utterance.
PacketStats = namedtuple(
    "PacketStats",
    "index channel timestamp_ns byte_length sha1 rms peak all_zero generation",
)


@dataclass(frozen=True)
class SpeechEvidenceCandidate:
    """Candidate view of one ``(StreamId, channel)`` inside the auto window.

    The window is the same time range for both channels, and the numeric
    fields feed the ``_rank`` step of ``AudioIngress._maybe_close_window``:
    exactly one admissible wins; both admissible → higher SNR, then higher
    voiced RMS, then channel 0. ``silent_rms_dbfs`` is the median of the
    per-frame RMSs inside the window — the only noise floor available before
    VAD has run — and ``admissible`` mirrors the hard ``SpeechEvidence`` gates
    as far as they already hold: non-zero, ``speech_span_ms >= 200``,
    ``voiced_frame_count >= 10`` and ``snr_db >= 6.0``.
    """

    channel: int
    nonzero_samples: int
    voiced_frame_count: int
    speech_span_ms: int
    voiced_rms_dbfs: Optional[float]
    silent_rms_dbfs: Optional[float]
    snr_db: Optional[float]
    clipped_ratio: float
    admissible: bool


@dataclass(frozen=True)
class SpeechEvidence:
    """Hard acoustic gate for one ``(StreamId, channel)``.

    ``admissible`` is true only when every numeric gate is satisfied:
    PCM non-empty and not all-zero, ``speech_span_ms >= 200``,
    ``voiced_frame_count >= 10`` on the 20 ms grid, no ``acoustic_silence``,
    no ``acoustic_low_energy``, and SNR >= 6 dB when computable. Auto-gain
    is applied after this evidence and cannot lift a rejected frame.
    ``rejection_reason`` is one of: ``empty_pcm``, ``all_zero_pcm``,
    ``speech_span_too_short``, ``voiced_frames_too_sparse``,
    ``acoustic_silence``, ``acoustic_low_energy``, ``snr_below_threshold``,
    ``no_audio_channel_with_speech``.
    """

    stream: "StreamId"
    channel: int
    total_samples: int
    voiced_frame_count: int
    speech_span_ms: int
    rms_dbfs: Optional[float]
    peak_dbfs: Optional[float]
    vad_voiced_rms: float
    vad_silent_rms: float
    snr_db: Optional[float]
    source_channel: Optional[int]
    admissible: bool
    rejection_reason: str


@dataclass(frozen=True)
class TurnContext:
    """Immutable identity of one pipeline turn, carried to its terminal event.

    Taken when the utterance opens and passed with the transcript, the reply and
    the error, so a callback of an older generation stays recognisable as old
    after the next button press, with no fallback to the lease in force then.
    """

    source: str
    device_id: str
    connection_generation: int
    session_generation: int

    @property
    def stream(self) -> "StreamId":
        """The microphone-stream identity this turn owns."""
        return StreamId(
            str(self.device_id), int(self.connection_generation), int(self.session_generation)
        )


@dataclass
class PendingPlayback:
    """One delivered reply waiting for the device's own end-of-playback report.

    Announcements complete in order on the device, so the finished callbacks are
    matched to these entries first-in-first-out; that is what ties a callback to
    the generation that produced the audio instead of to the moment it arrives.
    """

    generation: int
    session_generation: int
    media_id: str
    egress: str


def is_current_turn(
    context: Optional[TurnContext],
    device_id: str,
    connection_generation: int,
    session_generation: int,
    source: Optional[str] = None,
) -> bool:
    """Whether ``context`` still names the open run, every field compared.

    Fail-closed: no context, a missing field or a mismatch is a stale identity.
    """
    if context is None:
        return False
    if source is not None and str(context.source) != str(source):
        return False
    if str(context.device_id) != str(device_id):
        return False
    if int(context.connection_generation) != int(connection_generation):
        return False
    return int(context.session_generation) == int(session_generation)


def is_current_stream(
    stream: "StreamId", context: Optional[TurnContext]
) -> bool:
    """Whether ``stream`` is the one the given turn owns; local always is."""
    if stream is None:
        return False
    if str(stream.device_id or "") == "" and int(stream.session_generation) == 0:
        return True
    if context is None:
        return False
    return stream == context.stream




class DeviceState(str, Enum):
    """Per-device connection lifecycle."""

    DISABLED = "disabled"
    DISCOVERING = "discovering"
    CONNECTING = "connecting"
    PROVISIONING = "provisioning"
    AUTHENTICATING = "authenticating"
    SYNCING_CAPABILITIES = "syncing_capabilities"
    READY = "ready"
    VOICE_ACTIVE = "voice_active"
    RECONNECTING = "reconnecting"
    # Noise PSK rejected: keep the stored key and offer an import.
    AUTH_REQUIRED = "auth_required"
    ERROR = "error"


#: Stock per-session state machine. ``IDLE`` closes every run.
class SessionState(str, Enum):
    IDLE = "idle"
    BUTTON_TRIGGERED = "button_triggered"
    LISTENING = "listening"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CONTINUE_PENDING = "continue_pending"


#: Stock LED phases the firmware derives from Voice Assistant events. The
#: ``voice_assistant_leds`` light is internal, these ids are the contract.
LED_PHASES: Dict[str, int] = {
    "idle": 1,
    "waiting_for_command": 2,
    "listening_for_command": 3,
    "thinking": 4,
    "replying": 5,
    "not_ready": 10,
    "error": 11,
}

#: One bridge from the satellite LED phase to the toaster avatar state, so the
#: ring and the desktop face show the same phase of the same run. Values are the
#: ``face_widget.JarvisState`` names.
LED_PHASE_JARVIS_STATE: Dict[str, str] = {
    "idle": "idle",
    "waiting_for_command": "listening",
    "listening_for_command": "listening",
    "thinking": "thinking",
    "replying": "speaking",
    "not_ready": "asleep",
    "error": "error",
}


# ---------------------------------------------------------------------------
# Config + identity models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoicePEConfig:
    """Voice PE view of the flat ``jarvis.config.Settings`` fields."""

    enabled: bool = False
    discovery_enabled: bool = True
    host: Optional[str] = None
    port: int = DEFAULT_API_PORT
    device_name: Optional[str] = None
    mac_address: Optional[str] = None
    noise_psk_secret_id: Optional[str] = None
    room: Optional[str] = None
    disable_wake_words: bool = True
    prefer_api_audio: bool = True
    #: Legacy integer view of the same decision, kept for on-disk compatibility.
    preferred_input_channel: int = 0
    #: Explicit channel selection: ``enhanced`` (``data``/ch0), ``raw``
    #: (``data2``/ch1) or ``auto``. Locked per StreamId once chosen.
    audio_channel: str = "enhanced"
    #: One of :data:`DSP_MODES`. ``host_raw_aec`` routes the raw satellite
    #: channel through the native AEC3 lane (production default under loud
    #: Windows playback).
    voice_pe_dsp_mode: str = "host_raw_aec"
    #: Jitter buffer target in milliseconds.
    voice_pe_jitter_target_ms: int = 80
    #: Jitter buffer hard ceiling in milliseconds.
    voice_pe_jitter_max_ms: int = 250
    #: Delay-acquisition budget in milliseconds before a named failure.
    voice_pe_aec_acquire_max_ms: int = 1500
    continued_conversation: bool = True
    conversation_timeout_s: float = 300.0
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 30.0
    audio_queue_ms: int = 300
    led_brightness: float = 0.66
    led_rgb: Tuple[float, float, float] = (0.55, 0.0, 1.0)
    #: Event-entity mapping (``button_press_event`` -> Jarvis action name).
    button_actions: Dict[str, str] = field(default_factory=dict)
    #: Persisted per-device metadata keyed by MAC address.
    devices: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DeviceIdentity:
    """Stable device identity. IP is mutable, MAC plus node name is not."""

    mac_address: str = ""
    node_name: str = ""
    friendly_name: str = ""
    project_name: str = ""
    project_version: str = ""
    model: str = ""
    manufacturer: str = ""
    api_version: str = ""
    voice_feature_flags: int = 0
    addresses: list[str] = field(default_factory=list)
    last_connected: float = 0.0

    @property
    def compact_mac(self) -> str:
        """MAC without separators, the form ``APIClient(expected_mac=)`` takes."""
        return self.mac_address.replace(":", "").replace("-", "").lower()


@dataclass
class VoiceInputSession:
    """Input abstraction handed to the existing Jarvis VAD/STT path."""

    source_id: str
    room: Optional[str]
    conversation_id: str
    sample_rate: int = SAMPLE_RATE
    channels: int = SAMPLE_CHANNELS
    sample_width: int = SAMPLE_WIDTH
    enhanced: bool = True
