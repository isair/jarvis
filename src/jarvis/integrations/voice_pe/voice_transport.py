"""Microphone audio ingress for the Voice PE transport.

One bounded path per connection. The device pushes PCM16LE chunks in, the pump
hands them to the existing Jarvis ``VoiceListener`` queue, which owns the VAD,
the endpointing and Whisper. No second STT stack is created here.

Channel plan of the retail device (multi-channel firmware):
- channel 0: enhanced (XMOS processed) speech audio;
- channel 1: less processed audio.
Both are PCM16LE, 16 kHz, mono.

Strict types: satellite frames use ``SatelliteAudioFrame`` with an explicit
``channel`` field, local-mic frames use ``LocalMicFrame`` — a 3-argument
``AudioFrame(...)`` construction is impossible because the dataclass rejects
it (no default).

Auto channel selection is driven by the speech itself, not by 1 packet:
- both channels are buffered as a rolling pre-roll of at most ``2 s``;
- the main VAD does not run until the lock;
- a per-channel analyser looks for the first ``admissible speech evidence``;
- once at least one channel is admissible, a 300 ms comparison window opens
  and the other channel fills it too;
- rank inside that window as ``admissible → higher SNR → higher voiced RMS →
  enhanced (channel 0)``;
- if no channel becomes admissible within 5 s from the microphone start, the
  stream closes as ``no_audio_channel_with_speech``;
- after the lock, the *chosen* channel's rolling buffer is replayed from
  ``first_voiced_index - 500 ms`` into the main VAD, so the first second is
  not lost. The selection analysis is its own per-channel analyser and does
  not touch the listener's state machine at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .models import (
    AUDIO_SOURCE_VOICE_PE,
    AUDIO_SOURCE_LOCAL,
    AUDIO_CHANNEL_ENHANCED,
    AUDIO_CHANNEL_RAW,
    LOCAL_STREAM,
    SAMPLE_RATE,
    LocalMicFrame,
    PacketStats,
    SatelliteAudioFrame,
    SpeechEvidenceCandidate,
    StreamId,
    VoicePEConfig,
)
from ...listening.clean_audio_bus import get_bus, upsample_to_48k

#: Back-compat alias so the tests and the listener can import either name.
AudioFrame = SatelliteAudioFrame

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

SAMPLES_PER_MS = SAMPLE_RATE / 1000.0

#: Smallest useful budget, in samples (one 20 ms VAD frame at 16 kHz).
SAMPLES_PER_CHUNK_MIN = 320

#: Rolling pre-roll per channel: 2 s at 16 kHz is 32000 samples.
PRE_ROLL_S = 2.0
PRE_ROLL_SAMPLES = int(PRE_ROLL_S * SAMPLE_RATE)

#: After the first admissible candidate, how long to wait for the other one.
COMPARISON_WINDOW_MS = 300.0
#: Hard cut for the whole selection window: 5 s from ``microphone_start``.
NO_SPEECH_HARD_MS = 5000.0

#: Voiced-frame threshold on the 20 ms grid: a frame is voiced when its own
#: RMS exceeds by this factor the median of the frame RMSs inside the buffer.
AUTO_VOICED_RATIO = 1.5
AUTO_MIN_SNR_DB = 6.0
AUTO_MIN_VOICED_FRAMES = 10
AUTO_MIN_SPEECH_SPAN_MS = 200
#: 500 ms of history before the first voiced frame is preserved on replay.
REPLAY_HEADROOM_MS = 500


@dataclass(frozen=True)
class EndOfStream:
    """In-queue marker: this stream's PCM is complete.

    Travels in the same FIFO as the PCM blocks, so the pump sees it only after
    every block of the run has been handed to the listener queue. It carries
    no weight: the sample budget only accounts for PCM blocks.
    """

    stream: StreamId
    source: str = AUDIO_SOURCE_VOICE_PE


def pcm16_to_float32(payload: bytes):
    """Convert PCM16LE bytes to the float32 shape the listener expects."""
    if np is None:  # pragma: no cover
        return payload
    data = np.frombuffer(payload, dtype=np.int16)
    if data.ndim > 1:
        data = data.flatten()
    return (data.astype(np.float32) / 32768.0).reshape(-1)


def _frame_stats(buf, frame_samples: int, frame_ms: int) -> dict:
    """Aggregate a per-channel buffer into an evidence dict."""
    out = {
        "nonzero_samples": 0,
        "voiced_frame_count": 0,
        "speech_span_ms": 0,
        "first_voiced_index": None,
        "voiced_rms_dbfs": None,
        "silent_rms_dbfs": None,
        "snr_db": None,
        "clipped_ratio": 0.0,
    }
    if np is None or frame_samples <= 0 or not buf:
        return out
    parts: list = []
    for frame in buf:
        payload = getattr(frame, "samples", frame)
        if payload is None:
            continue
        if isinstance(payload, bytes):
            parts.append(
                np.frombuffer(payload, dtype=np.int16).astype(np.float64) / 32768.0
            )
        else:
            parts.append(np.asarray(payload, dtype=np.float64).reshape(-1))
    if not parts:
        return out
    flat = np.concatenate(parts)
    total = int(flat.size)
    if total == 0:
        return out
    out["nonzero_samples"] = int(np.count_nonzero(flat))
    grid = int(flat.size // frame_samples)
    if grid <= 0:
        return out
    tail = flat[: grid * frame_samples].reshape(grid, frame_samples)
    rms_per = np.sqrt(np.mean(np.square(tail), axis=1))
    positive = rms_per[rms_per > 0.0]
    if positive.size == 0:
        return out
    median = float(np.median(positive))
    threshold = median * AUTO_VOICED_RATIO
    voiced_index = [i for i, v in enumerate(rms_per) if float(v) > threshold]
    out["voiced_frame_count"] = len(voiced_index)
    if voiced_index:
        out["first_voiced_index"] = voiced_index[0]
        out["speech_span_ms"] = int((voiced_index[-1] - voiced_index[0] + 1) * frame_ms)
        voiced_slice = tail[voiced_index[0] : voiced_index[-1] + 1]
        voiced_rms = float(np.sqrt(np.mean(np.square(voiced_slice)))) * 32768.0
        out["voiced_rms_dbfs"] = 20.0 * float(np.log10(voiced_rms)) if voiced_rms > 0 else None
    out["silent_rms_dbfs"] = 20.0 * float(np.log10(median * 32768.0)) if median > 0 else None
    if out["voiced_rms_dbfs"] is not None and median > 0.0:
        out["snr_db"] = round(out["voiced_rms_dbfs"] - out["silent_rms_dbfs"], 3)
    out["clipped_ratio"] = round(float(np.count_nonzero(np.abs(flat) >= 0.999)) / total, 6)
    return out


def _admitted(stats: dict) -> bool:
    """Same gates as ``SpeechEvidence``, evaluated on a per-buffer dict."""
    if not stats.get("nonzero_samples"):
        return False
    if int(stats.get("voiced_frame_count") or 0) < AUTO_MIN_VOICED_FRAMES:
        return False
    if int(stats.get("speech_span_ms") or 0) < AUTO_MIN_SPEECH_SPAN_MS:
        return False
    snr = stats.get("snr_db")
    if snr is not None and float(snr) < AUTO_MIN_SNR_DB:
        return False
    return True


def _snr_diff(available: list) -> float:
    if len(available) >= 2 and available[0].snr_db is not None and available[1].snr_db is not None:
        return round(float(available[0].snr_db) - float(available[1].snr_db), 6)
    return 0.0


def _voiced_diff(available: list) -> float:
    if len(available) >= 2 and available[0].voiced_rms_dbfs is not None and available[1].voiced_rms_dbfs is not None:
        return round(float(available[0].voiced_rms_dbfs) - float(available[1].voiced_rms_dbfs), 6)
    return 0.0


class AudioIngress:
    """Bounded microphone queue with a per-channel analyser and evidence lock."""

    def __init__(self, listener, config: VoicePEConfig, metrics: dict) -> None:
        self._listener = listener
        self._config = config
        self._metrics = metrics
        budget_samples = max(
            SAMPLES_PER_CHUNK_MIN, config.audio_queue_ms * SAMPLES_PER_MS
        )
        self._budget_samples = int(budget_samples)
        #: FIFO of PCM blocks / ``EndOfStream`` items, in arrival order.
        self._items: deque = deque()
        self._pending_samples = 0
        self._closed = False
        self._stream: StreamId = LOCAL_STREAM
        self._wake: Optional[asyncio.Event] = None
        #: ``(key, channel) -> PacketStats`` rows for the ``--two-channel`` diag.
        self._packet_stats: dict = {}
        #: ``(key, channel) -> deque[SatelliteAudioFrame]`` rolling 2 s buffer.
        self._buffers: dict = {}
        #: ``(key) -> first-packet monotonic_ns``; the selection window start.
        self._first_ns: dict = {}
        #: ``(key) -> int`` of the first voiced frame index, per ``_frame_stats``.
        self._first_voiced: dict = {}
        #: ``(key) -> int`` of the first admissible candidate's monotonic_ns,
        #: used as the 300 ms comparison window's anchor.
        self._first_admissible_ns: dict = {}
        #: ``(key) -> int`` locked channel or ``None`` for no-admissible stop.
        self._selected: dict = {}
        #: ``(key) -> str`` exact reason that names the lock.
        self._selected_reason: dict = {}
        #: ``(key) -> int`` monotonic_ns of the lock.

def _item_sample_len(payload) -> int:
    """Sample count of a queue item's payload (int16 bytes or float32 array)."""
    if isinstance(payload, (bytes, bytearray)):
        return len(payload) // 2
    size = getattr(payload, "size", None)
    if isinstance(size, int):
        return int(size)
    try:
        return len(payload)
    except Exception:
        return 0
        self._selected_at_ns: dict = {}
        #: ``(key, channel) -> total samples`` in the whole current buffer.
        self._window_samples: dict = {}
        self._packet_index: dict = {}
        #: ``(key, channel) -> deque[arrival_ns]`` aligned with ``_buffers``.
        self._arrival: dict = {}
        #: ``key -> native lane handle`` for the host AEC lane.
        self._lane_handles: dict = {}
        #: ``key -> str`` last named DSP status of the lane.
        self.last_dsp_status: dict = {}
        #: ``key -> list`` shadow-compare cleaned blocks (diagnostics only).
        self._shadow: dict = {}
        #: False on firmware without ``multi_channel_audio`` (single channel).
        self._multi: bool = True

    # -- host AEC lane (native ABI v2) ----------------------------------

    def _mode(self) -> str:
        cfg = self._config
        return str(
            getattr(cfg, "voice_pe_dsp_mode", None)
            or getattr(cfg, "dsp_mode", None)
            or "host_raw_aec"
        ).strip().lower()

    def _lane(self, key: tuple) -> Optional[int]:
        """Create/return the native AEC lane of one satellite StreamId."""
        handle = self._lane_handles.get(key)
        if handle is not None:
            return handle
        from ...listening import audio_io as _aio

        if not _aio.has_native():
            self.last_dsp_status[key] = "audio_dsp_error"
            return None
        handle = _aio.get_or_create_pe_lane(
            self._config,
            str(key[0]),
            int(key[1]),
            int(key[2]),
        )
        if handle is None:
            self.last_dsp_status[key] = "audio_dsp_error"
            return None
        self._lane_handles[key] = int(handle)
        return int(handle)

    def _lane_feed(self, frame: SatelliteAudioFrame, ch: int) -> list:
        """Push one raw satellite block through the lane; return cleaned."""
        from ...listening import audio_io as _aio

        key = self._stream_key(frame.stream)
        handle = self._lane(key)
        if handle is None:
            return []
        at = self._last_arrival(key, ch)
        arr = pcm16_to_float32(frame.samples)
        st = _aio.push_capture(handle, arr, SAMPLE_RATE, at)
        _ = st
        cleaned: list = []
        while True:
            rate, block = _aio.pop_clean(key)
            if not rate or block is None or int(block.size) == 0:
                break
            cleaned.append(block)
        tel = _aio.lane_telemetry(key)
        aec_state = "acquiring"
        reference_active = False
        if isinstance(tel, dict):
            state = int(tel.get("aec_state", 1) or 1)
            aec_state = str(
                tel.get("aec_named_status") or tel.get("aec_state_label") or "acquiring"
            )
            self.last_dsp_status[key] = aec_state
            if int(tel.get("reference_active", 0) or 0) == 0:
                self.last_dsp_status[key] = "reference_alignment_failed"
                aec_state = "reference_alignment_failed"
            else:
                reference_active = True
            self._metrics.setdefault("aec", {})[str(key)] = tel
        self._publish_clean(key, cleaned, aec_state, reference_active)
        return cleaned

    # -- CleanAudioBus fan-out (per satellite, after its AEC lane) --------

    def _publish_clean(
        self,
        key: tuple,
        cleaned: list,
        aec_state: str,
        reference_active: bool,
    ) -> None:
        """Publish the lane's cleaned 16 kHz blocks as canonical frames.

        Each drained 160-sample/10 ms block is upsampled to the canonical
        48 kHz/480-sample format and stamped with the same monotonic QPC
        clock the publisher and the VAD adapter read from. No content is
        printed; only the named AEC state travels with the frame.
        """
        bus = get_bus()
        for block in cleaned or ():
            view = upsample_to_48k(block)
            try:
                bus.publish_frame(
                    view,
                    source_id=str(key[0]),
                    source_kind="voice_pe",
                    connection_generation=int(key[1]),
                    session_generation=int(key[2]),
                    aec_state=aec_state,
                    reference_active=reference_active,
                )
            except Exception:
                pass

    def _publish_raw(self, frame: SatelliteAudioFrame, ch: int) -> None:
        """Publish one un-cleaned block of the ``device_enhanced`` mode."""
        arr = pcm16_to_float32(frame.samples)
        if np is not None and arr.ndim > 1:
            arr = arr.reshape(-1)
        self._publish_clean(
            self._stream_key(frame.stream),
            [arr],
            "disabled",
            True,
        )

    def source_status(self, stream) -> dict:
        """Named AEC state of one satellite stream, for the UI/publisher."""
        key = self._stream_key(
            stream if isinstance(stream, StreamId) else (
                StreamId(str(stream), 0, 0) if stream else None
            )
        )
        return {
            "source_id": str(key[0]),
            "aec_state": self.last_dsp_status.get(key, "unknown"),
        }

    def _last_arrival(self, key: tuple, ch: int) -> int:
        q = self._arrival.get(key + (int(ch),))
        if q:
            return int(q[-1])
        return int(time.monotonic_ns())

    def _flush_lane(self, key: tuple) -> list:
        """Drain everything the lane still holds for one stream."""
        from ...listening import audio_io as _aio

        if key not in self._lane_handles:
            return []
        out: list = []
        for _ in range(256):  # bounded drain per EOS flush
            rate, block = _aio.pop_clean(key)
            if not rate or block is None or int(block.size) == 0:
                break
            out.append(block)
        self._publish_clean(
            key, out,
            self.last_dsp_status.get(key, "unknown"),
            bool(out),
        )
        return out

    def _destroy_lane(self, key: tuple) -> None:
        from ...listening import audio_io as _aio

        handle = self._lane_handles.pop(key, None)
        if handle is not None:
            _aio.lane_destroy(key)

    # -- producer side (ESPHome callback, same loop as the pump) ----------

    @staticmethod
    def _stream_key(stream: Optional[StreamId]) -> tuple:
        """Hashable identity of any stream shape (named StreamId or legacy int)."""
        if stream is None:
            return ("", 0, 0)
        if isinstance(stream, tuple):
            try:
                return (
                    str(stream.device_id),
                    int(stream.connection_generation),
                    int(stream.session_generation),
                )
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            return ("", 0, int(stream))
        except (TypeError, ValueError):
            return ("", 0, 0)

    @staticmethod
    def _stream_from_key(key: tuple) -> StreamId:
        return StreamId(str(key[0]), int(key[1]), int(key[2]))

    def _wake_pump(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def _trim_to_budget(self) -> None:
        """Drop the oldest PCM blocks in place; EOS keeps its position."""
        index = 0
        while self._pending_samples > self._budget_samples and index < len(self._items):
            item = self._items[index]
            if isinstance(item, EndOfStream):
                index += 1
                continue
            del self._items[index]
            self._pending_samples = max(
                0, self._pending_samples - _item_sample_len(item.samples)
            )
            self._metrics["audio_dropped_chunks"] = (
                int(self._metrics.get("audio_dropped_chunks", 0)) + 1
            )

    def push_frame(
        self,
        payload: bytes,
        stream: Optional[StreamId] = None,
        channel: int = AUDIO_CHANNEL_ENHANCED,
    ) -> None:
        """Append one SatelliteAudioFrame; satellite ``channel`` is required."""
        if self._closed or not payload:
            return
        active = stream or self._stream
        key = self._stream_key(active)
        ch = int(channel)
        if ch not in (AUDIO_CHANNEL_ENHANCED, AUDIO_CHANNEL_RAW):
            raise ValueError(f"Voice PE channel must be 0 or 1, got {channel!r}")
        # The strict dataclass raises if ``channel`` is missing.
        frame = SatelliteAudioFrame(active, AUDIO_SOURCE_VOICE_PE, payload, ch)
        buf = self._buffers.setdefault(key + (ch,), deque())
        buf.append(frame)
        # Rolling 2 s pre-roll per channel.
        cap = PRE_ROLL_SAMPLES
        total = self._window_samples.get(key + (ch,), 0) + len(payload) // 2
        while total > cap and len(buf) > 1:
            oldest = buf.popleft()
            total -= len(oldest.samples) // 2
        self._window_samples[key + (ch,)] = total
        if key not in self._first_ns:
            self._first_ns[key] = time.monotonic_ns()
        self._record_packet(active, payload, ch, time.monotonic_ns())
        if key in self._selected:
            self._deliver_now(frame, ch)
        elif (self._config.audio_channel or "enhanced").lower() == "auto":
            self._step_auto_window(key)
        else:
            configured = (self._config.audio_channel or "enhanced").lower()
            # Single-channel firmware never carries data2: the raw lock would
            # starve, so channel 0 is the only selectable one there.
            if self._multi is False:
                configured = "enhanced"
            target = (
                AUDIO_CHANNEL_RAW
                if configured == "raw"
                else AUDIO_CHANNEL_ENHANCED
            )
            self._lock(
                key,
                target,
                f"fixed:{configured}",
            )
            self._replay_from(key, self._first_voiced.get(key))

    def _deliver_now(self, frame: SatelliteAudioFrame, ch: int) -> None:
        """The lock is done: only the chosen signal reaches the main VAD.

        ``host_raw_aec``: the locked raw block is pushed through the native
        lane and the *cleaned* 16 kHz frames replace it on the queue.
        ``shadow_compare``: cleaned raw drives the VAD, the enhanced copy is
        kept in ``_shadow`` for diagnostics only. ``device_enhanced`` keeps
        the plain per-channel delivery.
        """
        key = self._stream_key(frame.stream)
        if self._selected.get(key) != ch:
            return
        mode = self._mode()
        if mode == "shadow_compare" and ch == AUDIO_CHANNEL_ENHANCED:
            self._shadow.setdefault(key, []).append(pcm16_to_float32(frame.samples))
            return
        if mode != "device_enhanced":
            cleaned = self._lane_feed(frame, ch)
            if cleaned:
                for block in cleaned:
                    self._pending_samples += int(block.size)
                    self._items.append(
                        SatelliteAudioFrame(
                            frame.stream, frame.source, block, int(ch)
                        )
                    )
                self._metrics["audio_chunks"] = int(
                    self._metrics.get("audio_chunks", 0)
                ) + len(cleaned)
                self._metrics["audio_bytes"] = int(
                    self._metrics.get("audio_bytes", 0)
                ) + sum(int(b.size) * 2 for b in cleaned)
                self._wake_pump()
                return
        self._pending_samples += len(frame.samples) // 2
        self._items.append(frame)
        self._publish_raw(frame, ch)
        self._trim_to_budget()
        self._metrics["audio_chunks"] = int(self._metrics.get("audio_chunks", 0)) + 1
        self._metrics["audio_bytes"] = int(
            self._metrics.get("audio_bytes", 0)
        ) + len(frame.samples)
        self._wake_pump()

    def _lock(self, key: tuple, channel: Optional[int], reason: str) -> None:
        self._selected[key] = None if channel is None else int(channel)
        self._selected_reason[key] = str(reason)
        self._selected_at_ns[key] = int(time.monotonic_ns())

    def _step_auto_window(self, key: tuple) -> None:
        """Speech-driven auto close, not "first packet then count"."""
        if key in self._selected:
            return
        first = self._first_ns.get(key)
        if first is None:
            return  # nothing has arrived yet
        now = int(time.monotonic_ns())
        elapsed_ms = (now - int(first)) / 1_000_000.0

        # Analyse both channels inside the 2 s rolling buffer.
        cands: dict = {}
        for ch in (AUDIO_CHANNEL_ENHANCED, AUDIO_CHANNEL_RAW):
            buf = self._buffers.get(key + (ch,), ())
            if not buf:
                continue
            stats = _frame_stats(
                buf,
                self._listener_frame_samples(),
                self._listener_vad_frame_ms(),
            )
            cands[ch] = SpeechEvidenceCandidate(
                channel=ch,
                nonzero_samples=int(stats["nonzero_samples"]),
                voiced_frame_count=int(stats["voiced_frame_count"]),
                speech_span_ms=int(stats["speech_span_ms"]),
                voiced_rms_dbfs=stats["voiced_rms_dbfs"],
                silent_rms_dbfs=stats["silent_rms_dbfs"],
                snr_db=stats["snr_db"],
                clipped_ratio=float(stats["clipped_ratio"]),
                admissible=_admitted(stats),
            )
            if stats.get("first_voiced_index") is not None:
                self._first_voiced.setdefault(key + (ch,), stats["first_voiced_index"])

        available = [c for c in cands.values() if c.admissible]
        if available:
            # First admissible opens a 300 ms comparison window once.
            if key not in self._first_admissible_ns:
                self._first_admissible_ns[key] = time.monotonic_ns()
            # Give the *other* channel exactly 300 ms; do it after that time
            # even if it stays silent (that is what ``no_audio_channel`` covers).
            cmp_ms = (now - self._first_admissible_ns[key]) / 1_000_000.0
            if cmp_ms >= COMPARISON_WINDOW_MS:
                chosen = self._rank(available)
                reason = self._reason_for(available, cmp_ms)
                self._lock(key, chosen.channel, reason)
                self._replay_from(key, self._first_voiced.get(key + (chosen.channel,)))
            return

        # No admissible yet: hard cut at 5 s from microphone start.
        if elapsed_ms >= NO_SPEECH_HARD_MS:
            self._lock(key, None, "no_audio_channel_with_speech@5s")
            self._close_no_speech(key)

    def _rank(self, available: list) -> SpeechEvidenceCandidate:
        ordered = sorted(
            available,
            key=lambda c: (
                -(float(c.snr_db) if c.snr_db is not None else -1.0),
                -(float(c.voiced_rms_dbfs) if c.voiced_rms_dbfs is not None else -1.0),
                int(c.channel),
            ),
        )
        return ordered[0]

    @staticmethod
    def _reason_for(available: list, cmp_ms: float) -> str:
        prefix = f"speech@{int(cmp_ms)}ms:evidence_by="
        if len(available) == 1:
            return prefix + "single_admissible"
        snr = _snr_diff(available)
        voiced = _voiced_diff(available)
        if abs(snr) >= 0.5:
            return prefix + "snr"
        if abs(voiced) >= 0.0 and voiced != 0.0:
            return prefix + "voiced_rms_db"
        av = sorted(available, key=lambda c: int(c.channel))
        if int(av[0].channel) == AUDIO_CHANNEL_ENHANCED:
            return prefix + "enhanced_default"
        return prefix + "raw_only"

    def _listener_vad_frame_ms(self) -> int:
        return int(
            getattr(getattr(self._listener, "cfg", None), "vad_frame_ms", 20) or 20
        )

    def _listener_frame_samples(self) -> int:
        return int(getattr(self._listener, "_frame_samples", 0) or 320)

    def _replay_from(self, key: tuple, first_voiced_index: Optional[int]) -> None:
        """Replay the chosen buffer starting 500 ms before ``first_voiced``.

        With a host lane active the locked raw blocks are pushed through the
        native AEC and the cleaned 16 kHz frames are what reaches the queue.
        """
        ch = self._selected.get(key)
        if ch is None:
            return
        buf = self._buffers.get(key + (int(ch),), ())
        if not buf:
            return
        n = len(buf)
        # 500 ms of history = 500/20 = 25 frames on the 20 ms grid.
        head = max(0, 500 // self._listener_vad_frame_ms())
        start = max(0, (first_voiced_index or 0) - head) if first_voiced_index else 0
        mode = self._mode()
        use_lane = mode != "device_enhanced"
        replayed = 0
        for i in range(start, n):
            frame = buf[i]
            if use_lane:
                cleaned = self._lane_feed(frame, int(ch))
                if cleaned:
                    for block in cleaned:
                        if self._pending_samples >= self._budget_samples and self._items:
                            drop = self._items.popleft()
                            self._pending_samples = max(
                                0,
                                self._pending_samples
                                - (len(drop.samples) // 2 if isinstance(
                                    getattr(drop, "samples", None), (bytes, bytearray))
                                   else (int(drop.samples.size)
                                         if getattr(drop, "samples", None) is not None else 0)),
                            )
                        self._pending_samples += int(block.size)
                        self._items.append(
                            SatelliteAudioFrame(
                                frame.stream, frame.source, block, int(ch)
                            )
                        )
                        replayed += 1
                    continue
            total = self._window_samples.get(key + (int(ch),), 0)
            if total > 0 and self._pending_samples >= self._budget_samples:
                # Drop the oldest from the main FIFO on overflow.
                self._pending_samples = max(
                    0, self._pending_samples - len(frame.samples) // 2
                )
                if self._items:
                    self._items.popleft()
            self._pending_samples += len(frame.samples) // 2
            self._items.append(frame)
            replayed += 1
            if not use_lane:
                self._publish_raw(frame, int(ch))
        self._trim_to_budget()
        self._metrics["audio_chunks"] = int(self._metrics.get("audio_chunks", 0)) + (
            n - start
        )
        self._metrics["audio_bytes"] = int(self._metrics.get("audio_bytes", 0)) + sum(
            len(buf[i].samples) for i in range(start, n)
        )
        self._wake_pump()

    def _close_no_speech(self, key: tuple) -> None:
        """Signal the shared queue that this stream has no usable audio."""
        if self._listener is None:
            return
        close = getattr(self._listener, "pad_until_endpoint", None)
        if callable(close):
            try:
                close(self._stream_from_key(key), AUDIO_SOURCE_VOICE_PE)
            except Exception:
                pass

    def put(
        self,
        data: bytes,
        data2: Optional[bytes] = None,
        stream: Optional[StreamId] = None,
    ) -> None:
        """Non-blocking ingest of both channels of one ``VoiceAssistantAudio``."""
        active = stream or self._stream
        if data:
            self.push_frame(data, active, AUDIO_CHANNEL_ENHANCED)
        if data2 is not None and len(data2) > 0:
            self.push_frame(data2, active, AUDIO_CHANNEL_RAW)

    def push_local(self, payload: bytes) -> None:
        """Append exactly one ``LocalMicFrame`` (not a satellite 4-field)."""
        if self._closed or not payload:
            return
        self._items.append(
            LocalMicFrame(
                pcm16_to_float32(payload) if np is not None else payload
            )
        )
        self._wake_pump()

    # -- stream identity --------------------------------------------------

    def set_stream(self, stream: StreamId) -> StreamId:
        """Bind the queue to one satellite stream and return it.

        A change of the full ``StreamId`` destroys the old lane's adaptive
        state (never reused across generations); rebinding the very same id
        keeps it alive for that one run.
        """
        new = StreamId(
            str(stream.device_id),
            int(stream.connection_generation),
            int(stream.session_generation),
        )
        old = self._stream_key(self._stream)
        new_key = self._stream_key(new)
        if old != new_key and old in self._lane_handles:
            self._destroy_lane(old)
        self._stream = new
        return self._stream

    @property
    def stream(self) -> StreamId:
        return self._stream

    def _record_packet(
        self, stream: StreamId, payload: bytes, channel: int, at_ns: int
    ) -> None:
        key = self._stream_key(stream)
        n = int(self._packet_index.get("n", 0)) + 1
        self._packet_index["n"] = n
        rms = peak = 0.0
        all_zero = False
        try:
            if np is not None:
                data = np.frombuffer(payload, dtype=np.int16)
                as_float = data.astype(np.float64) / 32768.0
                rms = float(np.sqrt(np.mean(np.square(as_float)))) if data.size else 0.0
                peak = float(np.max(np.abs(as_float))) if data.size else 0.0
                all_zero = bool(data.size and np.all(data == 0))
            else:  # pragma: no cover
                all_zero = not any(payload)
        except Exception:  # pragma: no cover
            rms = peak = 0.0
        try:
            generation = int(stream.session_generation)
        except (AttributeError, TypeError, ValueError):
            generation = int(key[2])
        self._packet_stats[key + (int(channel), int(time.monotonic_ns() % 1e12))] = PacketStats(
            index=n,
            channel=int(channel),
            timestamp_ns=at_ns,
            byte_length=len(payload),
            sha1=hashlib.sha1(payload).hexdigest(),
            rms=round(rms, 8),
            peak=round(peak, 8),
            all_zero=all_zero,
            generation=generation,
        )
        q = self._arrival.setdefault(key + (int(channel),), deque())
        q.append(int(at_ns))
        while len(q) > PRE_ROLL_SAMPLES // SAMPLES_PER_CHUNK_MIN + 4:
            q.popleft()

    def mark_end_of_stream(
        self, stream: StreamId, source: str = AUDIO_SOURCE_VOICE_PE
    ) -> None:
        """Append the EOS marker of one stream behind its PCM blocks.

        The marker is weightless: the sample budget only counts PCM blocks.
        With a host lane active, the block is appended behind the *last*
        cleaned frame the lane still holds for this stream.
        """
        if self._closed:
            return
        key = self._stream_key(stream)
        ch = int(self._selected.get(key, AUDIO_CHANNEL_ENHANCED) or 0)
        if key in self._lane_handles:
            cleaned = self._flush_lane(key)
            for block in cleaned:
                self._pending_samples += int(block.size)
                self._items.append(
                    SatelliteAudioFrame(stream, source, block, ch)
                )
        # Weightless: the marker takes no budget slot and is never refused.
        self._items.append(EndOfStream(stream, source))
        self._wake_pump()

    def push_tts_reference(
        self, stream: Optional[StreamId], samples, rate_hz: int, arrival_ns: int
    ) -> None:
        """Model the exact satellite-bound TTS payload as the lane far-end."""
        key = self._stream_key(stream or self._stream)
        if key not in self._lane_handles:
            return
        from ...listening import audio_io as _aio

        _aio.push_reference(key, samples, int(rate_hz), int(arrival_ns))

    # -- consumer side (pump task) ----------------------------------------

    async def _next_item(self):
        while True:
            if self._items:
                item = self._items.popleft()
                if self._wake is not None:
                    self._wake.clear()
                return item
            if self._closed:
                return None
            if self._wake is None:
                self._wake = asyncio.Event()
            if self._wake.is_set():
                self._wake.clear()
                continue
            await self._wake.wait()

    async def pump(self) -> None:
        """Feed the shared listener queue until the session closes."""
        while not self._closed:
            item = await self._next_item()
            if item is None:
                if self._closed:
                    return
                continue
            if isinstance(item, EndOfStream):
                self._close_stream(item)
                continue
            if not getattr(item, "samples", None):
                self._close_stream(
                    EndOfStream(item.stream if isinstance(item, (LocalMicFrame, SatelliteAudioFrame)) else LOCAL_STREAM,
                             getattr(item, "source", AUDIO_SOURCE_VOICE_PE))
                )
                continue
            self._pending_samples = max(
                0, self._pending_samples - _item_sample_len(item.samples)
            )
            self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
            if isinstance(item, LocalMicFrame):
                delivered = item
            else:
                if isinstance(item.samples, (bytes, bytearray)):
                    delivered = SatelliteAudioFrame(
                        item.stream,
                        item.source,
                        pcm16_to_float32(item.samples),
                        int(item.channel),
                    )
                else:
                    # Cleaned float32 frame straight off the native AEC lane.
                    delivered = item
            try:
                self._listener._audio_q.put_nowait(delivered)
            except Exception:
                self._metrics["audio_dropped_chunks"] = (
                    int(self._metrics.get("audio_dropped_chunks", 0)) + 1
                )

    def _close_stream(self, marker: "EndOfStream") -> None:
        """State the end of this stream on the shared listener queue.

        Also the terminal event of the stream: its host AEC lane is destroyed
        so the next ``StreamId`` starts with fresh adaptive state.
        """
        self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
        close = getattr(self._listener, "pad_until_endpoint", None)
        if callable(close):
            try:
                close(marker.stream, marker.source)
            except Exception:
                pass
        self._destroy_lane(self._stream_key(marker.stream))

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Drop stale items for a new session or a cancel, in place."""
        self._items.clear()
        self._pending_samples = 0
        self._packet_stats.clear()
        self._selected.clear()
        self._selected_reason.clear()
        self._selected_at_ns.clear()
        self._packet_index.clear()
        for buf in self._buffers.values():
            buf.clear()
        self._buffers.clear()
        self._window_samples.clear()
        self._first_ns.clear()
        self._first_voiced.clear()
        self._first_admissible_ns.clear()
        for key in list(self._lane_handles):
            from ...listening import audio_io as _aio

            _aio.lane_reset(key)
        self._arrival.clear()
        self._shadow.clear()
        self._wake_pump()

    def close(self) -> None:
        self._closed = True
        self.reset()

    def depth_ms(self) -> float:
        return round(self._pending_samples / SAMPLES_PER_MS, 1)

    def is_empty(self) -> bool:
        return not self._items and all(not b for b in self._buffers.values())

    def items(self) -> list:
        return list(self._items)

    def queue(self) -> "AudioIngress":
        return self

    # -- per-channel readbacks, used by ``--two-channel`` -----------------

    def selected_audio_channel(self, stream: Optional[StreamId] = None) -> Optional[int]:
        key = self._stream_key(stream)
        raw = self._selected.get(key, ...)
        return None if raw is ... or raw is None else int(raw)

    def selection_reason(self, stream: Optional[StreamId] = None) -> Optional[str]:
        key = self._stream_key(stream)
        return self._selected_reason.get(key)

    def selection_at_ns(self, stream: Optional[StreamId] = None) -> Optional[int]:
        key = self._stream_key(stream)
        return int(self._selected_at_ns[key]) if key in self._selected_at_ns else None

    def packet_stats(self, stream=None, channel: Optional[int] = None) -> list:
        key = self._stream_key(stream)
        out: list = []
        for full_key, stat in self._packet_stats.items():
            if full_key[:3] == key and (channel is None or full_key[3] == int(channel)):
                out.append(stat)
        return out

    def channel_stats(self) -> list:
        rows: list = []
        for (dev, conn, sess, ch), buf in sorted(
            self._buffers.items(), key=lambda kr: (kr[0][0], kr[0][3])
        ):
            if not buf:
                continue
            stats = _frame_stats(
                buf,
                self._listener_frame_samples(),
                self._listener_vad_frame_ms(),
            )
            c = SpeechEvidenceCandidate(
                channel=int(ch),
                nonzero_samples=int(stats["nonzero_samples"]),
                voiced_frame_count=int(stats["voiced_frame_count"]),
                speech_span_ms=int(stats["speech_span_ms"]),
                voiced_rms_dbfs=stats["voiced_rms_dbfs"],
                silent_rms_dbfs=stats["silent_rms_dbfs"],
                snr_db=stats["snr_db"],
                clipped_ratio=float(stats["clipped_ratio"]),
                admissible=_admitted(stats),
            )
            rows.append((str(dev), int(conn), int(sess), c))
        return rows

    def offline_candidates(self, stream=None) -> list:
        """Per ``(key, channel)`` ``SpeechEvidenceCandidate``s for the readback."""
        key = self._stream_key(stream)
        rows: list = []
        for ch in (AUDIO_CHANNEL_ENHANCED, AUDIO_CHANNEL_RAW):
            buf = self._buffers.get(key + (ch,), ())
            if not buf:
                continue
            stats = _frame_stats(
                buf,
                self._listener_frame_samples(),
                self._listener_vad_frame_ms(),
            )
            rows.append(
                SpeechEvidenceCandidate(
                    channel=int(ch),
                    nonzero_samples=int(stats["nonzero_samples"]),
                    voiced_frame_count=int(stats["voiced_frame_count"]),
                    speech_span_ms=int(stats["speech_span_ms"]),
                    voiced_rms_dbfs=stats["voiced_rms_dbfs"],
                    silent_rms_dbfs=stats["silent_rms_dbfs"],
                    snr_db=stats["snr_db"],
                    clipped_ratio=float(stats["clipped_ratio"]),
                    admissible=_admitted(stats),
                )
            )
        return rows

    def dump_wavs(self, prefix: str, stream=None) -> dict:
        """``channel-0-enhanced.wav``, ``channel-1-raw.wav`` + native lane 3."""
        import json as _json
        import wave as _wave

        key = self._stream_key(stream)
        paths: dict = {}
        names = {
            AUDIO_CHANNEL_ENHANCED: "channel-0-enhanced.wav",
            AUDIO_CHANNEL_RAW: "channel-1-raw.wav",
        }
        for ch in (AUDIO_CHANNEL_ENHANCED, AUDIO_CHANNEL_RAW):
            buf = self._buffers.get(key + (ch,), ())
            if not buf:
                continue
            joined = b"".join(bytes(f.samples) for f in buf)
            path = f"{prefix}-{names[int(ch)]}"
            try:
                with _wave.open(path, "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(SAMPLE_RATE)
                    handle.writeframes(joined)
            except Exception:
                continue
            paths[int(ch)] = {"path": path, "bytes": len(joined), "samples": len(joined) // 2}
        # Native multitrack: cleaned lane 3 from the same engine + sidecar JSON.
        from ...listening import audio_io as _aio

        if key in self._lane_handles:
            clean_path = f"{prefix}-lane-2-cleaned.wav"
            tel = _aio.lane_telemetry(key) or {}
            status = self.last_dsp_status.get(key) or str(
                tel.get("aec_named_status") or "unknown"
            )
            try:
                with _wave.open(clean_path, "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(SAMPLE_RATE)
                    rows = self._shadow.get(key) or []
                    joined = b"".join(
                        r.astype("<i2", copy=False).tobytes(order="C")
                        if hasattr(r, "tobytes") else bytes(r)
                        for r in rows
                    )
                    if joined:
                        handle.writeframes(joined)
                paths["cleaned"] = {
                    "path": clean_path,
                    "named_status": status,
                }
            except Exception:
                pass
            try:
                side = {
                    "stream": {"device_id": str(key[0]),
                               "connection_generation": int(key[1]),
                               "session_generation": int(key[2])},
                    "named_status": status,
                    "selected_channel": self._selected.get(key),
                    "selection_reason": self._selected_reason.get(key),
                    "engine_telemetry": _aio.engine_telemetry(),
                    "lane_telemetry": tel,
                }
                with open(f"{prefix}-lane.json", "w", encoding="utf-8") as f:
                    _json.dump(side, f, ensure_ascii=False, indent=1)
            except Exception:
                pass
        return paths


class UdpAudioServer(asyncio.DatagramProtocol):
    """UDP fallback for firmware without the API_AUDIO flag."""

    def __init__(self, ingress: AudioIngress) -> None:
        super().__init__()
        self._ingress = ingress
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.remote_addr = None

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data, addr) -> None:  # type: ignore[override]
        if self.remote_addr is None:
            self.remote_addr = addr
        # The UDP path is one stream only; the enhanced channel is explicit.
        self._ingress.push_frame(data or b"", self._ingress.stream, AUDIO_CHANNEL_ENHANCED)

    def error_received(self, exc) -> None:  # type: ignore[override]
        self._ingress.push_frame(b"", self._ingress.stream, AUDIO_CHANNEL_ENHANCED)

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.remote_addr = None
