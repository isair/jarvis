"""Microphone audio ingress for the Voice PE transport.

One bounded path per generation. The device pushes PCM16LE chunks in, the pump
hands them to the existing Jarvis ``VoiceListener`` queue, which owns the VAD,
the endpointing and Whisper. No second STT stack is created here.

Channel plan of the retail device (multi-channel firmware):
- channel 0: enhanced (XMOS processed) speech audio;
- channel 1: less processed audio.
Both are PCM16LE, 16 kHz, mono.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from .models import (
    AUDIO_SOURCE_VOICE_PE,
    SAMPLE_RATE,
    AudioFrame,
    VoicePEConfig,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard requirement
    np = None  # type: ignore[assignment]

#: 16 kHz mono 16-bit: one millisecond is 16 samples.
SAMPLES_PER_MS = SAMPLE_RATE / 1000.0


@dataclass(frozen=True)
class EndOfStream:
    """In-queue marker: this source's PCM for ``generation`` is complete.

    It travels in the same FIFO as the PCM blocks, so the pump can only see it
    after every block of the run has been handed to the listener queue.
    """

    source: str
    generation: int


def pcm16_to_float32(payload: bytes):
    """Convert PCM16LE bytes to the float32 array shape the listener expects."""
    if np is None:  # pragma: no cover
        return payload
    data = np.frombuffer(payload, dtype=np.int16)
    if data.ndim > 1:
        data = data.flatten()
    return (data.astype(np.float32) / 32768.0).reshape(-1)


class AudioIngress:
    """Bounded microphone queue: PCM frames plus one weightless EOS marker.

    Queue items are ``AudioFrame(source, generation, pcm)`` or ``EndOfStream``,
    so both the pump and the listener can tell the two kinds apart and every
    block carries the generation that owns it.
    """

    def __init__(self, listener, config: VoicePEConfig, metrics: dict) -> None:
        self._listener = listener
        self._config = config
        self._metrics = metrics
        budget_samples = max(
            SAMPLES_PER_CHUNK_MIN, config.audio_queue_ms * SAMPLES_PER_MS
        )
        self._budget_samples = int(budget_samples)
        # One slot beside the PCM budget belongs to the EOS marker, so the
        # marker can always be appended without displacing audio.
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(5, int(budget_samples // 512) + 3)
        )
        self._pending_samples = 0
        self._closed = False

    # -- producer side (ESPHome callback thread) -------------------------

    def _drop_oldest_pcm(self) -> bool:
        """Drop the oldest PCM block, keeping the weightless marker in place."""
        held: list = []
        dropped = False
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, EndOfStream):
                held.append(item)
                continue
            self._pending_samples = max(0, self._pending_samples - len(item.samples) // 2)
            self._metrics["audio_dropped_chunks"] = (
                int(self._metrics.get("audio_dropped_chunks", 0)) + 1
            )
            dropped = True
            break
        for marker in held:
            try:
                self._queue.put_nowait(marker)
            except asyncio.QueueFull:  # pragma: no cover - weightless, rare
                pass
        return dropped

    def put(self, data: bytes, data2: Optional[bytes] = None, generation: int = 0) -> None:
        """Never-blocking ingest from ``handle_audio``, stamped with a turn."""
        if self._closed:
            return
        payload = data
        if (
            self._config.preferred_input_channel == 1
            and data2 is not None
            and len(data2) > 0
        ):
            payload = data2
        if not payload:
            return

        samples = len(payload) // (2 * 1)
        while self._pending_samples + samples > self._budget_samples:
            if not self._drop_oldest_pcm():
                break
        self._pending_samples += samples
        frame = AudioFrame(AUDIO_SOURCE_VOICE_PE, int(generation), payload)
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Freshest audio wins: make room once, then the fresh block stands.
            self._drop_oldest_pcm()
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:  # pragma: no cover - marker-only queue
                self._pending_samples = max(0, self._pending_samples - samples)
        self._metrics["audio_chunks"] = int(self._metrics.get("audio_chunks", 0)) + 1
        self._metrics["audio_bytes"] = int(self._metrics.get("audio_bytes", 0)) + len(
            payload
        )

    # -- consumer side (pump task) --------------------------------------

    def mark_end_of_stream(self, source: str, generation: int) -> None:
        """Append the EOS marker of this run behind the queued PCM blocks.

        Never refused: the marker is weightless, so a full queue gives way by
        dropping the oldest PCM block, which is the same freshest-audio rule the
        budget already applies.
        """
        if self._closed:
            return
        marker = EndOfStream(source, int(generation))
        for _ in range(self._queue.maxsize + 1):
            try:
                self._queue.put_nowait(marker)
                return
            except asyncio.QueueFull:
                if not self._drop_oldest_pcm():
                    break
        # Still full (only markers in front): replace the front marker in place.
        try:
            self._queue.get_nowait()
            self._queue.put_nowait(marker)
        except Exception:  # pragma: no cover - queue touched only here
            pass

    async def pump(self) -> None:
        """Feed the shared listener queue until the session closes."""
        while not self._closed:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:  # pragma: no cover
                raise
            if isinstance(item, EndOfStream):
                # FIFO: every PCM block of this run is already on the listener
                # queue, so this source's stream is closed in order here.
                self._close_stream(item)
                continue
            if not isinstance(item, AudioFrame):
                # A bare block from a queue written before the stamping shape.
                item = AudioFrame(AUDIO_SOURCE_VOICE_PE, 0, item)
            if not item.samples:
                # The UDP path states the close with one empty datagram.
                self._close_stream(EndOfStream(item.source, item.generation))
                continue
            self._pending_samples = max(
                0, self._pending_samples - len(item.samples) // 2
            )
            self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
            delivered = AudioFrame(
                item.source, item.generation, pcm16_to_float32(item.samples)
            )
            try:
                self._listener._audio_q.put_nowait(delivered)
            except Exception:
                # The listener queue is bounded too: freshest audio wins.
                self._metrics["audio_dropped_chunks"] = (
                    int(self._metrics.get("audio_dropped_chunks", 0)) + 1
                )

    def _close_stream(self, marker: "EndOfStream") -> None:
        """State the end of this source's stream on the shared listener queue."""
        self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
        close = getattr(self._listener, "pad_until_endpoint", None)
        if not callable(close):
            return
        try:
            close(marker.source, marker.generation)
        except Exception:
            pass

    def reset(self) -> None:
        """Drop stale audio for a new generation / cancel; markers are ignored."""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, EndOfStream):
                continue
            self._pending_samples = max(0, self._pending_samples - len(item.samples) // 2)
        self._pending_samples = 0

    def close(self) -> None:
        self._closed = True
        self.reset()

    def depth_ms(self) -> float:
        return round(self._pending_samples / SAMPLES_PER_MS, 1)

    def is_empty(self) -> bool:
        return self._queue.empty()

    def queue(self) -> asyncio.Queue:
        """The raw queue, shared with the UDP fallback receiver."""
        return self._queue


#: Smallest useful budget, in samples (one 20 ms VAD frame at 16 kHz).
SAMPLES_PER_CHUNK_MIN = 320


class UdpAudioServer(asyncio.DatagramProtocol):
    """UDP microphone receiver for firmware without the API_AUDIO flag."""

    def __init__(self, queue: asyncio.Queue, generation: int = 0) -> None:
        super().__init__()
        self._queue = queue
        self.generation = int(generation)
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.remote_addr = None

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self.transport = transport

    def _stamp(self, payload: bytes) -> None:
        try:
            self._queue.put_nowait(
                AudioFrame(AUDIO_SOURCE_VOICE_PE, self.generation, payload)
            )
        except asyncio.QueueFull:
            pass

    def datagram_received(self, data, addr) -> None:  # type: ignore[override]
        if self.remote_addr is None:
            self.remote_addr = addr
        self._stamp(data)

    def error_received(self, exc) -> None:  # type: ignore[override]
        self._stamp(b"")

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.remote_addr = None
