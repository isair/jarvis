"""OpenAI Realtime premium voice backend (audio-to-audio) — GA protocol.

Uses the official Realtime WebSocket API (no OpenAI-Beta header).
Designed for unit tests with injected transport mocks — never opens a real
socket unless ``allow_network=True`` (production path only).
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from ..debug import debug_log
from .openai_credentials import (
    MissingOpenAICredential,
    require_openai_api_key,
)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
OPENAI_PCM_RATE = 24000  # official pcm16 input/output rate

CORA_INSTRUCTIONS = (
    "Numele tău este Cora. Ești asistenta vocală personală a utilizatorului. "
    "Vorbește exclusiv în limba română, natural, clar, calm și profesionist. "
    "Răspunde scurt și direct dacă utilizatorul nu cere detalii. "
    "Ascultă atent numerele, numele și cuvintele românești. "
    "Dacă nu ești sigură ce ai auzit, repetă ce ai înțeles și cere confirmare. "
    "Nu inventa informații și nu pretinde că ai executat acțiuni neexecutate. "
    "Nu activa unelte și nu pretinde că ai acces la shell, git, instalări sau "
    "modificări de sistem fără o confirmare locală explicită."
)


@dataclass
class RealtimeServerError(Exception):
    """Structured Realtime server / protocol failure (never includes secrets)."""

    error_type: str = ""
    error_code: str = ""
    error_param: Optional[str] = None
    event_id: Optional[str] = None
    close_code: Optional[int] = None
    phase: str = ""

    def __str__(self) -> str:
        parts = [p for p in (self.error_type, self.error_code, self.phase) if p]
        return ":".join(parts) if parts else "RealtimeServerError"

    @property
    def code(self) -> str:
        """Compact code for RealtimeTurnResult.error (no sensitive payload)."""
        if self.error_code:
            return str(self.error_code)
        if self.error_type:
            return str(self.error_type)
        if self.close_code is not None:
            return f"close:{self.close_code}"
        return "realtime_error"


@dataclass
class RealtimeTurnResult:
    """Outcome of one premium turn."""

    ok: bool
    user_transcript: str = ""
    assistant_transcript: str = ""
    audio_bytes: bytes = b""
    error: str = ""
    error_type: str = ""
    error_code: str = ""
    error_param: Optional[str] = None
    event_id: Optional[str] = None
    close_code: Optional[int] = None
    used_premium: bool = False
    fallback_needed: bool = False
    ignored_no_wake: bool = False
    item_deleted: bool = False
    response_created: bool = False


class RealtimeTransport(Protocol):
    """Minimal WS transport for production or mocks."""

    def connect(self, url: str, headers: dict[str, str], timeout: float) -> None: ...
    def send_json(self, payload: dict) -> None: ...
    def recv_json(self, timeout: float) -> Optional[dict]: ...
    def close(self) -> None: ...


class AiohttpRealtimeTransport:
    """Production transport using aiohttp WebSocket (sync façade)."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._session = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._recv_q: asyncio.Queue | None = None
        self.close_code: Optional[int] = None

    def connect(self, url: str, headers: dict[str, str], timeout: float) -> None:
        self._loop = asyncio.new_event_loop()
        self._error = None
        self._ready.clear()
        self.close_code = None

        def _runner():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._async_connect(url, headers, timeout))
                self._ready.set()
                self._loop.run_forever()
            except BaseException as e:
                self._error = e
                self._ready.set()

        self._thread = threading.Thread(target=_runner, name="openai-realtime-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout + 5):
            raise TimeoutError("realtime websocket connect timed out")
        if self._error:
            raise self._error

    async def _async_connect(self, url: str, headers: dict[str, str], timeout: float):
        import aiohttp

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            url,
            headers=headers,
            heartbeat=20,
            timeout=aiohttp.ClientTimeout(total=timeout),
        )
        self._recv_q = asyncio.Queue()

        async def _reader():
            assert self._ws is not None and self._recv_q is not None
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._recv_q.put(json.loads(msg.data))
                    except Exception:
                        await self._recv_q.put(
                            {"type": "error", "error": {"message": "invalid_json"}}
                        )
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    self.close_code = self._ws.close_code
                    await self._recv_q.put(
                        {
                            "type": "_ws_closed",
                            "close_code": self._ws.close_code,
                        }
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.close_code = self._ws.close_code
                    await self._recv_q.put(
                        {
                            "type": "error",
                            "error": {
                                "type": "ws_error",
                                "code": "ws_error",
                            },
                            "close_code": self._ws.close_code,
                        }
                    )
                    break

        asyncio.ensure_future(_reader(), loop=self._loop)

    def send_json(self, payload: dict) -> None:
        if not self._loop or not self._ws:
            raise RuntimeError("transport not connected")
        fut = asyncio.run_coroutine_threadsafe(
            self._ws.send_str(json.dumps(payload)), self._loop
        )
        fut.result(timeout=10)

    def recv_json(self, timeout: float) -> Optional[dict]:
        if not self._loop or self._recv_q is None:
            raise RuntimeError("transport not connected")

        async def _get():
            return await asyncio.wait_for(self._recv_q.get(), timeout=timeout)

        fut = asyncio.run_coroutine_threadsafe(_get(), self._loop)
        try:
            return fut.result(timeout=timeout + 1)
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._loop and self._ws:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
                try:
                    fut.result(timeout=3)
                except Exception:
                    pass
                try:
                    self.close_code = self._ws.close_code
                except Exception:
                    pass
            if self._loop and self._session:
                fut = asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
                try:
                    fut.result(timeout=3)
                except Exception:
                    pass
        finally:
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=3)
            self._ws = None
            self._session = None
            self._loop = None


@dataclass
class MockRealtimeTransport:
    """Test double — records traffic, yields scripted events."""

    script: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)
    connected: bool = False
    connect_calls: int = 0
    closed: bool = False
    connect_error: Optional[BaseException] = None
    close_code: Optional[int] = None
    _idx: int = 0
    _url: str = ""
    _headers: dict[str, str] = field(default_factory=dict)

    def connect(self, url: str, headers: dict[str, str], timeout: float) -> None:
        self.connect_calls += 1
        if self.connect_error:
            raise self.connect_error
        self.connected = True
        self._url = url
        # Never expose Authorization value.
        self._headers = {
            k: ("<redacted>" if k.lower() == "authorization" else v)
            for k, v in headers.items()
        }

    def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv_json(self, timeout: float) -> Optional[dict]:
        if self._idx >= len(self.script):
            return None
        item = self.script[self._idx]
        self._idx += 1
        return item

    def close(self) -> None:
        self.closed = True
        self.connected = False


def float32_mono_to_pcm16_24k(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono mic PCM to little-endian pcm16 @ 24 kHz (one-shot)."""
    if audio is None or getattr(audio, "size", 0) == 0:
        return b""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16).tobytes()
    if int(sample_rate) == OPENAI_PCM_RATE:
        return pcm16
    converted, _ = audioop.ratecv(pcm16, 2, 1, int(sample_rate), OPENAI_PCM_RATE, None)
    return converted


class StatefulPcm24kResampler:
    """Continuous float32 mono → pcm16 @ 24 kHz; ratecv state is never reset per frame.

    Silent and voiced frames are converted alike — no energy/VAD gating here.
    """

    def __init__(self, source_rate: int) -> None:
        self.source_rate = int(source_rate)
        self._state: Any = None
        self.samples_in = 0
        self.samples_out = 0

    def reset(self, source_rate: Optional[int] = None) -> None:
        if source_rate is not None:
            self.source_rate = int(source_rate)
        self._state = None
        self.samples_in = 0
        self.samples_out = 0

    def convert_float32(self, audio: np.ndarray) -> bytes:
        if audio is None or getattr(audio, "size", 0) == 0:
            return b""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        x = np.clip(x, -1.0, 1.0)
        self.samples_in += int(x.size)
        pcm16 = (x * 32767.0).astype(np.int16).tobytes()
        if self.source_rate == OPENAI_PCM_RATE:
            self.samples_out += int(x.size)
            return pcm16
        converted, self._state = audioop.ratecv(
            pcm16, 2, 1, self.source_rate, OPENAI_PCM_RATE, self._state
        )
        self.samples_out += len(converted) // 2
        return converted

    @property
    def duration_in_sec(self) -> float:
        return float(self.samples_in) / max(self.source_rate, 1)

    @property
    def duration_out_sec(self) -> float:
        return float(self.samples_out) / float(OPENAI_PCM_RATE)


@dataclass
class PendingTranscript:
    """Server-side turn ready for wake gate (transcription.completed)."""

    transcript: str
    item_id: str = ""


# Premium Audio V4 — documented GA schema (developers.openai.com realtime-vad).
# Prefer semantic_vad + eagerness=low; create_response/interrupt_response false.
SEMANTIC_VAD_TURN_DETECTION: dict[str, Any] = {
    "type": "semantic_vad",
    "eagerness": "low",
    "create_response": False,
    "interrupt_response": False,
}
NEAR_FIELD_NOISE_REDUCTION: dict[str, Any] = {"type": "near_field"}

# Bounded backpressure: ~10 s of pcm16 @ 24 kHz awaiting send acknowledgement.
STREAM_PENDING_BYTES_LIMIT = OPENAI_PCM_RATE * 2 * 10


def play_pcm16_24k(pcm: bytes, *, player: Optional[Callable[[np.ndarray, int], None]] = None) -> None:
    """Play OpenAI pcm16 mono @ 24 kHz exactly once."""
    if not pcm:
        return
    samples = np.frombuffer(pcm, dtype=np.int16)
    if player is not None:
        player(samples, OPENAI_PCM_RATE)
        return
    import sounddevice as sd

    sd.play(samples, samplerate=OPENAI_PCM_RATE, blocking=True)


def build_ga_session_update(
    *,
    model: str,
    voice: str,
    transcription_model: str,
    language: str,
    instructions: str = CORA_INSTRUCTIONS,
) -> dict[str, Any]:
    """GA Realtime session.update (Premium Audio V4) — no beta fields / no tools.

    Schema source: OpenAI Realtime VAD guide (semantic_vad + noise_reduction).
    Live rejection must be reported before inventing alternate parameters.
    """
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OPENAI_PCM_RATE,
                    },
                    "transcription": {
                        "model": transcription_model,
                        "language": language,
                    },
                    "noise_reduction": dict(NEAR_FIELD_NOISE_REDUCTION),
                    "turn_detection": dict(SEMANTIC_VAD_TURN_DETECTION),
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OPENAI_PCM_RATE,
                    },
                    "voice": voice,
                },
            },
        },
    }


def normalize_openai_transcript(text: str) -> str:
    """Normalize OpenAI transcript for wake matching (not for memory rewriting)."""
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def transcript_starts_with_wake(
    transcript: str,
    wake_word: str,
    aliases: Optional[Sequence[str]] = None,
) -> bool:
    """True if normalized transcript begins with wake word or a configured alias."""
    t = normalize_openai_transcript(transcript).lower()
    t = re.sub(r"^[\s\.\,\!\?\;\:\"\'«»\-—–]+", "", t)
    if not t:
        return False
    names = {str(wake_word or "").strip().lower()}
    for a in aliases or []:
        a = str(a or "").strip().lower()
        if a:
            names.add(a)
    names.discard("")
    # Longer aliases first (e.g. "hey cora" before "cora")
    for name in sorted(names, key=len, reverse=True):
        if t == name:
            return True
        if t.startswith(name):
            rest = t[len(name):]
            if not rest or rest[0] in " ,.!?;:":
                return True
    return False


def _error_from_event(evt: dict, *, phase: str, close_code: Optional[int] = None) -> RealtimeServerError:
    err = evt.get("error") if isinstance(evt.get("error"), dict) else {}
    return RealtimeServerError(
        error_type=str(err.get("type") or evt.get("type") or ""),
        error_code=str(err.get("code") or ""),
        error_param=err.get("param"),
        event_id=evt.get("event_id") or err.get("event_id"),
        close_code=close_code if close_code is not None else evt.get("close_code"),
        phase=phase,
    )


class OpenAIRealtimeSession:
    """Reusable Realtime session with idle timeout (GA protocol)."""

    def __init__(
        self,
        cfg,
        *,
        transport_factory: Optional[Callable[[], RealtimeTransport]] = None,
        allow_network: bool = False,
        audio_player: Optional[Callable[[np.ndarray, int], None]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self._transport_factory = transport_factory
        self._allow_network = allow_network
        self._audio_player = audio_player
        self._clock = clock
        self._transport: Optional[RealtimeTransport] = None
        self._last_activity: Optional[float] = None
        self._lock = threading.RLock()
        self._fallback_used_for_turn = False
        self._handshake_done = False
        self._last_session_snapshot: dict = {}
        # Premium Audio V4 streaming state
        self._resampler: Optional[StatefulPcm24kResampler] = None
        self._stream_bytes_sent = 0
        self._stream_frames_appended = 0
        self._stream_frames_dropped = 0
        self._pending_append_bytes = 0
        self._stream_item_id = ""
        self._stream_partial_transcript = ""
        self._streaming_armed = False

    @property
    def is_open(self) -> bool:
        return self._transport is not None and self._handshake_done

    def _idle_timeout(self) -> float:
        return float(getattr(self.cfg, "openai_realtime_idle_timeout_sec", 60.0) or 60.0)

    def _model(self) -> str:
        return str(getattr(self.cfg, "openai_realtime_model", "gpt-realtime-2.1"))

    def _transcription_model(self) -> str:
        return str(
            getattr(self.cfg, "openai_realtime_transcription_model", "gpt-4o-transcribe")
        )

    def _voice(self) -> str:
        return str(getattr(self.cfg, "openai_realtime_voice", "marin"))

    def _language(self) -> str:
        return str(getattr(self.cfg, "openai_realtime_language", "ro") or "ro")

    def close(self) -> None:
        with self._lock:
            if self._transport is not None:
                try:
                    self._transport.close()
                except Exception as e:
                    debug_log(f"realtime close error: {type(e).__name__}", "openai")
                self._transport = None
            self._last_activity = None
            self._handshake_done = False
            self._streaming_armed = False
            self._pending_append_bytes = 0
            if self._resampler is not None:
                self._resampler.reset()

    def stream_counters(self) -> dict[str, int]:
        """Explicit counters for sent / dropped streaming frames (tests + diag)."""
        return {
            "bytes_sent": int(self._stream_bytes_sent),
            "frames_appended": int(self._stream_frames_appended),
            "frames_dropped": int(self._stream_frames_dropped),
            "pending_append_bytes": int(self._pending_append_bytes),
            "samples_in": int(self._resampler.samples_in) if self._resampler else 0,
            "samples_out": int(self._resampler.samples_out) if self._resampler else 0,
        }

    def ensure_streaming(self, source_rate: int) -> bool:
        """Open GA session for continuous mic streaming (LISTENING only)."""
        try:
            self._ensure_session()
        except Exception as e:
            debug_log(f"realtime ensure_streaming failed: {type(e).__name__}", "openai")
            return False
        with self._lock:
            if self._resampler is None or self._resampler.source_rate != int(source_rate):
                self._resampler = StatefulPcm24kResampler(int(source_rate))
            if not self._streaming_armed:
                try:
                    assert self._transport is not None
                    self._transport.send_json({"type": "input_audio_buffer.clear"})
                except Exception:
                    pass
                self._streaming_armed = True
                self._stream_item_id = ""
                self._stream_partial_transcript = ""
            return True

    def stream_append_float32(self, frame, sample_rate: int) -> bool:
        """Append one mic frame (all samples, including silence). True if sent."""
        with self._lock:
            if not self._handshake_done or self._transport is None:
                self._stream_frames_dropped += 1
                return False
            if self._pending_append_bytes >= STREAM_PENDING_BYTES_LIMIT:
                self._stream_frames_dropped += 1
                debug_log(
                    f"realtime backpressure drop "
                    f"(pending={self._pending_append_bytes} "
                    f"dropped={self._stream_frames_dropped})",
                    "openai",
                )
                return False
            if self._resampler is None or self._resampler.source_rate != int(sample_rate):
                self._resampler = StatefulPcm24kResampler(int(sample_rate))
            try:
                pcm = self._resampler.convert_float32(frame)
            except Exception:
                self._stream_frames_dropped += 1
                return False
            if not pcm:
                return True
            try:
                b64 = base64.b64encode(pcm).decode("ascii")
                self._transport.send_json({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                })
            except Exception:
                self._stream_frames_dropped += 1
                return False
            self._stream_bytes_sent += len(pcm)
            self._stream_frames_appended += 1
            self._pending_append_bytes += len(pcm)
            if self._pending_append_bytes > STREAM_PENDING_BYTES_LIMIT // 2:
                self._pending_append_bytes = max(
                    0, self._pending_append_bytes - len(pcm)
                )
            self._last_activity = self._clock()
            return True

    def poll_pending_transcript(self, timeout: float = 0.0) -> Optional[PendingTranscript]:
        """Short poll for server transcription.completed (never sends response.create)."""
        with self._lock:
            transport = self._transport
            if transport is None or not self._handshake_done:
                return None
        deadline = self._clock() + max(0.0, float(timeout))
        while True:
            remaining = deadline - self._clock()
            wait = 0.0 if timeout <= 0 else max(0.0, remaining)
            try:
                evt = transport.recv_json(timeout=wait if wait > 0 else 0.001)
            except Exception:
                return None
            if evt is None:
                if timeout <= 0 or self._clock() >= deadline:
                    return None
                continue
            et = str(evt.get("type") or "")
            if et == "_ws_closed":
                self.close()
                return None
            if et in ("error", "response.failed") or et.endswith(".failed"):
                debug_log(f"realtime stream error event: {et}", "openai")
                continue
            if et == "conversation.item.created":
                item = evt.get("item") or {}
                if item.get("id"):
                    self._stream_item_id = str(item.get("id"))
                continue
            if et == "input_audio_buffer.committed":
                self._pending_append_bytes = 0
                continue
            if et in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.done",
            ):
                tr = normalize_openai_transcript(str(evt.get("transcript") or ""))
                item_id = str(evt.get("item_id") or self._stream_item_id or "")
                self._stream_partial_transcript = ""
                self._streaming_armed = False
                return PendingTranscript(transcript=tr, item_id=item_id)
            if et == "conversation.item.input_audio_transcription.delta":
                delta = evt.get("delta") or evt.get("transcript") or ""
                if delta:
                    self._stream_partial_transcript = (
                        (self._stream_partial_transcript + str(delta))
                        if self._stream_partial_transcript
                        else str(delta)
                    )
                if evt.get("item_id"):
                    self._stream_item_id = str(evt.get("item_id"))
                continue
            if timeout <= 0:
                return None
            if self._clock() >= deadline:
                return None

    def finish_premium_turn(self, pending: PendingTranscript) -> RealtimeTurnResult:
        """Wake gate on OpenAI transcript only; at most one response.create."""
        transport = self._transport
        if transport is None:
            return RealtimeTurnResult(ok=False, error="no_transport", fallback_needed=False)

        user_tr = normalize_openai_transcript(pending.transcript)
        item_id = pending.item_id or self._stream_item_id

        if not user_tr:
            if item_id:
                self._delete_conversation_item(transport, item_id)
            return RealtimeTurnResult(
                ok=False,
                error="empty_transcript",
                user_transcript="",
                item_deleted=bool(item_id),
                fallback_needed=False,
            )

        require_wake = self._require_wake_each_turn()
        wake_ok = True
        if require_wake:
            wake_ok = transcript_starts_with_wake(
                user_tr, self._wake_word(), self._wake_aliases()
            )
        if not wake_ok:
            deleted = self._delete_conversation_item(transport, item_id) if item_id else False
            debug_log("premium ignored — no OpenAI wake word", "openai")
            print("  🔇 premium ignored — no OpenAI wake word", flush=True)
            self._last_activity = self._clock()
            return RealtimeTurnResult(
                ok=False,
                user_transcript=user_tr,
                ignored_no_wake=True,
                item_deleted=deleted,
                response_created=False,
                fallback_needed=False,
            )

        transport.send_json({"type": "response.create"})
        asst_tr = ""
        asst_parts: list[str] = []
        audio_chunks: list[bytes] = []
        deadline = self._clock() + 45.0
        empty_recvs = 0
        while self._clock() < deadline:
            remaining = max(0.05, deadline - self._clock())
            evt = transport.recv_json(timeout=min(2.0, remaining))
            if evt is None:
                empty_recvs += 1
                if empty_recvs >= 3:
                    break
                continue
            empty_recvs = 0
            et = str(evt.get("type") or "")
            if et == "_ws_closed":
                self.close()
                return RealtimeTurnResult(
                    ok=False,
                    user_transcript=user_tr,
                    error="connection_closed",
                    response_created=True,
                    fallback_needed=False,
                )
            if et in ("error", "response.failed") or et.endswith(".failed"):
                err = _error_from_event(evt, phase="response")
                self.close()
                return RealtimeTurnResult(
                    ok=False,
                    user_transcript=user_tr,
                    error=err.code,
                    error_type=err.error_type,
                    error_code=err.error_code,
                    response_created=True,
                    fallback_needed=False,
                )
            if et == "response.output_audio.delta":
                delta = evt.get("delta") or ""
                if delta:
                    try:
                        audio_chunks.append(base64.b64decode(delta))
                    except Exception:
                        pass
            elif et == "response.output_audio_transcript.delta":
                piece = evt.get("delta") or ""
                if piece:
                    asst_parts.append(str(piece))
            elif et == "response.output_audio_transcript.done":
                done_tr = evt.get("transcript")
                if done_tr:
                    asst_tr = str(done_tr)
                elif asst_parts:
                    asst_tr = "".join(asst_parts)
            elif et == "response.audio.delta":
                delta = evt.get("delta") or ""
                if delta and not audio_chunks:
                    try:
                        audio_chunks.append(base64.b64decode(delta))
                    except Exception:
                        pass
            elif et == "response.done":
                break

        if not asst_tr and asst_parts:
            asst_tr = "".join(asst_parts)
        audio_out = b"".join(audio_chunks)
        if not audio_out:
            return RealtimeTurnResult(
                ok=False,
                user_transcript=user_tr,
                error="empty_audio_response",
                response_created=True,
                fallback_needed=False,
            )
        play_pcm16_24k(audio_out, player=self._audio_player)
        self._last_activity = self._clock()
        return RealtimeTurnResult(
            ok=True,
            user_transcript=user_tr,
            assistant_transcript=asst_tr,
            audio_bytes=audio_out,
            used_premium=True,
            response_created=True,
        )

    def _recv_until(
        self,
        transport: RealtimeTransport,
        wanted: Sequence[str],
        *,
        timeout: float,
        phase: str,
    ) -> dict:
        """Receive events until one of ``wanted`` types, or raise structured error."""
        deadline = self._clock() + timeout
        wanted_set = set(wanted)
        while self._clock() < deadline:
            remaining = max(0.05, deadline - self._clock())
            evt = transport.recv_json(timeout=min(2.0, remaining))
            if evt is None:
                continue
            et = str(evt.get("type") or "")
            if et == "_ws_closed":
                raise RealtimeServerError(
                    error_type="connection_closed",
                    error_code="connection_closed",
                    close_code=evt.get("close_code"),
                    phase=phase,
                )
            if et == "error" or et == "response.failed" or et.endswith(".failed"):
                raise _error_from_event(evt, phase=phase)
            if et in wanted_set:
                return evt
            # Ignore unrelated lifecycle noise during handshake / turn wait.
        raise RealtimeServerError(
            error_type="timeout",
            error_code=f"timeout_waiting_{wanted[0] if wanted else 'event'}",
            phase=phase,
        )

    def _build_session_update(self) -> dict[str, Any]:
        return build_ga_session_update(
            model=self._model(),
            voice=self._voice(),
            transcription_model=self._transcription_model(),
            language=self._language(),
            instructions=CORA_INSTRUCTIONS,
        )

    def _handshake(self, transport: RealtimeTransport) -> None:
        """Wait session.created → session.update (GA) → session.updated. No audio."""
        created = self._recv_until(
            transport, ("session.created",), timeout=20.0, phase="await_session_created"
        )
        self._last_session_snapshot = created.get("session") or {}
        transport.send_json(self._build_session_update())
        updated = self._recv_until(
            transport, ("session.updated",), timeout=20.0, phase="await_session_updated"
        )
        self._last_session_snapshot = updated.get("session") or self._last_session_snapshot
        self._handshake_done = True
        debug_log("realtime GA handshake complete", "openai")

    def _ensure_session(self) -> RealtimeTransport:
        with self._lock:
            now = self._clock()
            if self._transport is not None and self._last_activity is not None:
                if (now - self._last_activity) > self._idle_timeout():
                    debug_log("realtime idle timeout — closing session", "openai")
                    self.close()

            if self._transport is not None and self._handshake_done:
                return self._transport

            if self._transport_factory is not None:
                transport = self._transport_factory()
            else:
                if not self._allow_network:
                    raise RuntimeError("realtime network disabled (tests)")
                transport = AiohttpRealtimeTransport()

            api_key = require_openai_api_key()
            url = f"{REALTIME_URL}?model={self._model()}"
            # GA: Authorization only — never OpenAI-Beta.
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
            transport.connect(url, headers, timeout=20.0)
            try:
                self._handshake(transport)
            except RealtimeServerError:
                try:
                    transport.close()
                except Exception:
                    pass
                raise
            except Exception as e:
                try:
                    transport.close()
                except Exception:
                    pass
                raise RealtimeServerError(
                    error_type=type(e).__name__,
                    error_code=type(e).__name__,
                    phase="handshake",
                ) from e

            self._transport = transport
            self._last_activity = now
            return transport

    def _require_wake_each_turn(self) -> bool:
        return getattr(self.cfg, "openai_realtime_require_wake_each_turn", True) is True

    def _wake_word(self) -> str:
        return str(getattr(self.cfg, "wake_word", "cora") or "cora")

    def _wake_aliases(self) -> list[str]:
        raw = getattr(self.cfg, "wake_aliases", None) or []
        try:
            return [str(a) for a in raw]
        except Exception:
            return []

    def _delete_conversation_item(self, transport: RealtimeTransport, item_id: str) -> bool:
        if not item_id:
            return False
        transport.send_json({
            "type": "conversation.item.delete",
            "item_id": item_id,
        })
        return True

    def handle_utterance(
        self,
        audio_f32: np.ndarray,
        sample_rate: int,
    ) -> RealtimeTurnResult:
        """Two-phase premium turn: transcribe+wake gate, then optional response.

        Phase A: clear/append/commit → wait OpenAI input transcript (no response.create).
        Phase B: if wake accepted → response.create → one audio playback.
        No-wake / empty transcript → fail closed (no Piper, no response.create).
        """
        self._fallback_used_for_turn = False
        try:
            transport = self._ensure_session()
        except MissingOpenAICredential:
            print("MISSING_OPENAI_CREDENTIAL", flush=True)
            return RealtimeTurnResult(
                ok=False,
                error="MISSING_OPENAI_CREDENTIAL",
                error_code="MISSING_OPENAI_CREDENTIAL",
                fallback_needed=True,
            )
        except RealtimeServerError as e:
            debug_log(
                f"realtime handshake failed: type={e.error_type} code={e.error_code}",
                "openai",
            )
            self.close()
            return RealtimeTurnResult(
                ok=False,
                error=e.code,
                error_type=e.error_type,
                error_code=e.error_code,
                error_param=e.error_param,
                event_id=e.event_id,
                close_code=e.close_code,
                fallback_needed=True,
            )
        except Exception as e:
            debug_log(f"realtime session open failed: {type(e).__name__}", "openai")
            self.close()
            return RealtimeTurnResult(
                ok=False,
                error=type(e).__name__,
                error_type=type(e).__name__,
                fallback_needed=True,
            )

        try:
            pcm = float32_mono_to_pcm16_24k(audio_f32, sample_rate)
            if not pcm:
                return RealtimeTurnResult(
                    ok=False,
                    error="empty_audio",
                    fallback_needed=False,
                )

            # ---- Phase A: commit audio and wait for OpenAI transcription only ----
            transport.send_json({"type": "input_audio_buffer.clear"})
            b64 = base64.b64encode(pcm).decode("ascii")
            transport.send_json({
                "type": "input_audio_buffer.append",
                "audio": b64,
            })
            transport.send_json({"type": "input_audio_buffer.commit"})

            user_tr = ""
            item_id = ""
            deadline = self._clock() + 30.0
            empty_recvs = 0
            while self._clock() < deadline:
                remaining = max(0.05, deadline - self._clock())
                evt = transport.recv_json(timeout=min(2.0, remaining))
                if evt is None:
                    empty_recvs += 1
                    if empty_recvs >= 3:
                        break
                    continue
                empty_recvs = 0
                et = str(evt.get("type") or "")

                if et == "_ws_closed":
                    self.close()
                    return RealtimeTurnResult(
                        ok=False,
                        error="connection_closed",
                        error_type="connection_closed",
                        close_code=evt.get("close_code"),
                        fallback_needed=False,
                    )
                if et in ("error", "response.failed") or et.endswith(".failed"):
                    err = _error_from_event(evt, phase="transcribe")
                    self.close()
                    return RealtimeTurnResult(
                        ok=False,
                        error=err.code,
                        error_type=err.error_type,
                        error_code=err.error_code,
                        error_param=err.error_param,
                        event_id=err.event_id,
                        close_code=err.close_code,
                        fallback_needed=False,
                    )
                if et == "conversation.item.created":
                    item = evt.get("item") or {}
                    if item.get("id"):
                        item_id = str(item.get("id"))
                if et in (
                    "conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.done",
                ):
                    user_tr = str(evt.get("transcript") or "")
                    if evt.get("item_id"):
                        item_id = str(evt.get("item_id"))
                    break
                if et == "conversation.item.input_audio_transcription.delta":
                    delta = evt.get("delta") or evt.get("transcript") or ""
                    if delta:
                        user_tr = (user_tr + str(delta)) if user_tr else str(delta)
                    if evt.get("item_id"):
                        item_id = str(evt.get("item_id"))
                if et == "conversation.item.input_audio_transcription.failed":
                    return RealtimeTurnResult(
                        ok=False,
                        error="transcription_failed",
                        error_type="transcription_failed",
                        fallback_needed=False,
                    )

            user_tr = normalize_openai_transcript(user_tr)
            if not user_tr:
                debug_log("OpenAI transcript unavailable", "openai")
                if item_id:
                    self._delete_conversation_item(transport, item_id)
                return RealtimeTurnResult(
                    ok=False,
                    error="empty_transcript",
                    user_transcript="",
                    item_deleted=bool(item_id),
                    fallback_needed=False,
                )

            # ---- Wake gate (authoritative OpenAI transcript) ----
            require_wake = self._require_wake_each_turn()
            wake_ok = True
            if require_wake:
                wake_ok = transcript_starts_with_wake(
                    user_tr, self._wake_word(), self._wake_aliases()
                )
            if not wake_ok:
                deleted = self._delete_conversation_item(transport, item_id) if item_id else False
                debug_log("premium ignored — no OpenAI wake word", "openai")
                print("  🔇 premium ignored — no OpenAI wake word", flush=True)
                self._last_activity = self._clock()
                return RealtimeTurnResult(
                    ok=False,
                    user_transcript=user_tr,
                    ignored_no_wake=True,
                    item_deleted=deleted,
                    response_created=False,
                    fallback_needed=False,
                )

            # ---- Phase B: response.create only after wake accepted ----
            transport.send_json({"type": "response.create"})
            asst_tr = ""
            asst_parts: list[str] = []
            audio_chunks: list[bytes] = []
            deadline = self._clock() + 45.0
            empty_recvs = 0
            while self._clock() < deadline:
                remaining = max(0.05, deadline - self._clock())
                evt = transport.recv_json(timeout=min(2.0, remaining))
                if evt is None:
                    empty_recvs += 1
                    if empty_recvs >= 3:
                        break
                    continue
                empty_recvs = 0
                et = str(evt.get("type") or "")

                if et == "_ws_closed":
                    self.close()
                    return RealtimeTurnResult(
                        ok=False,
                        user_transcript=user_tr,
                        error="connection_closed",
                        error_type="connection_closed",
                        close_code=evt.get("close_code"),
                        response_created=True,
                        fallback_needed=False,
                    )
                if et in ("error", "response.failed") or et.endswith(".failed"):
                    err = _error_from_event(evt, phase="response")
                    self.close()
                    return RealtimeTurnResult(
                        ok=False,
                        user_transcript=user_tr,
                        error=err.code,
                        error_type=err.error_type,
                        error_code=err.error_code,
                        error_param=err.error_param,
                        event_id=err.event_id,
                        close_code=err.close_code,
                        response_created=True,
                        fallback_needed=False,
                    )
                if et == "response.output_audio.delta":
                    delta = evt.get("delta") or ""
                    if delta:
                        try:
                            audio_chunks.append(base64.b64decode(delta))
                        except Exception:
                            pass
                elif et == "response.output_audio_transcript.delta":
                    piece = evt.get("delta") or ""
                    if piece:
                        asst_parts.append(str(piece))
                elif et == "response.output_audio_transcript.done":
                    done_tr = evt.get("transcript")
                    if done_tr:
                        asst_tr = str(done_tr)
                    elif asst_parts:
                        asst_tr = "".join(asst_parts)
                elif et == "response.audio.delta":
                    delta = evt.get("delta") or ""
                    if delta and not audio_chunks:
                        try:
                            audio_chunks.append(base64.b64decode(delta))
                        except Exception:
                            pass
                elif et == "response.audio_transcript.done":
                    if not asst_tr:
                        asst_tr = str(evt.get("transcript") or "")
                elif et == "response.done":
                    break

            if not asst_tr and asst_parts:
                asst_tr = "".join(asst_parts)

            audio_out = b"".join(audio_chunks)
            if not audio_out:
                return RealtimeTurnResult(
                    ok=False,
                    user_transcript=user_tr,
                    error="empty_audio_response",
                    response_created=True,
                    fallback_needed=False,
                )

            play_pcm16_24k(audio_out, player=self._audio_player)
            self._last_activity = self._clock()
            return RealtimeTurnResult(
                ok=True,
                user_transcript=user_tr,
                assistant_transcript=asst_tr,
                audio_bytes=audio_out,
                used_premium=True,
                response_created=True,
            )
        except RealtimeServerError as e:
            debug_log(
                f"realtime turn failed: type={e.error_type} code={e.error_code}",
                "openai",
            )
            self.close()
            return RealtimeTurnResult(
                ok=False,
                error=e.code,
                error_type=e.error_type,
                error_code=e.error_code,
                error_param=e.error_param,
                event_id=e.event_id,
                close_code=e.close_code,
                fallback_needed=False,
            )
        except Exception as e:
            debug_log(f"realtime turn failed: {type(e).__name__}", "openai")
            self.close()
            return RealtimeTurnResult(
                ok=False,
                error=type(e).__name__,
                error_type=type(e).__name__,
                fallback_needed=False,
            )


_SESSION: Optional[OpenAIRealtimeSession] = None
_SESSION_LOCK = threading.Lock()


def get_realtime_session(cfg, **kwargs) -> OpenAIRealtimeSession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = OpenAIRealtimeSession(cfg, **kwargs)
        else:
            _SESSION.cfg = cfg
        return _SESSION


def reset_realtime_session_for_tests() -> None:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION.close()
        _SESSION = None


def premium_enabled(cfg) -> bool:
    # Exact True only — avoids MagicMock truthiness in tests.
    return getattr(cfg, "openai_realtime_enabled", False) is True
