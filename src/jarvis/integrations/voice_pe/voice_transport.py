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
from typing import Optional

from .models import AUDIO_SOURCE_VOICE_PE, SAMPLE_RATE, VoicePEConfig

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard requirement
    np = None  # type: ignore[assignment]

#: 16 kHz mono 16-bit: one millisecond is 16 samples.
SAMPLES_PER_MS = SAMPLE_RATE / 1000.0


def pcm16_to_float32(payload: bytes):
    """Convert PCM16LE bytes to the float32 array shape the listener expects."""
    if np is None:  # pragma: no cover
        return payload
    data = np.frombuffer(payload, dtype=np.int16)
    if data.ndim > 1:
        data = data.flatten()
    return (data.astype(np.float32) / 32768.0).reshape(-1)


class AudioIngress:
    """Bounded microphone queue with oldest-first drop on overflow."""

    def __init__(self, listener, config: VoicePEConfig, metrics: dict) -> None:
        self._listener = listener
        self._config = config
        self._metrics = metrics
        budget_samples = max(
            SAMPLES_PER_CHUNK_MIN, config.audio_queue_ms * SAMPLES_PER_MS
        )
        self._budget_samples = int(budget_samples)
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(4, int(budget_samples // 512) + 2)
        )
        self._pending_samples = 0
        self._closed = False

    # -- producer side (ESPHome callback thread) -------------------------

    def put(self, data: bytes, data2: Optional[bytes] = None) -> None:
        """Never-blocking ingest from ``handle_audio``."""
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
            try:
                oldest = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending_samples -= len(oldest) // 2
            self._metrics["audio_dropped_chunks"] = (
                int(self._metrics.get("audio_dropped_chunks", 0)) + 1
            )
        self._pending_samples += samples
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover - budget keeps it rare
            self._metrics["audio_dropped_chunks"] = (
                int(self._metrics.get("audio_dropped_chunks", 0)) + 1
            )
            self._pending_samples = max(0, self._pending_samples - samples)
        self._metrics["audio_chunks"] = int(self._metrics.get("audio_chunks", 0)) + 1
        self._metrics["audio_bytes"] = int(self._metrics.get("audio_bytes", 0)) + len(
            payload
        )

    # -- consumer side (pump task) --------------------------------------

    async def pump(self) -> None:
        """Feed the shared listener queue until the session closes."""
        while not self._closed:
            try:
                payload = await self._queue.get()
            except asyncio.CancelledError:  # pragma: no cover
                raise
            self._pending_samples = max(
                0, self._pending_samples - len(payload) // 2
            )
            self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
            try:
                # Tagged item: exactly one microphone owns an utterance.
                self._listener._audio_q.put_nowait(
                    (
                        AUDIO_SOURCE_VOICE_PE,
                        pcm16_to_float32(payload),
                    )
                )
            except Exception:
                # The listener queue is bounded too: freshest audio wins.
                self._metrics["audio_dropped_chunks"] = (
                    int(self._metrics.get("audio_dropped_chunks", 0)) + 1
                )

    def reset(self) -> None:
        """Drop stale audio for a new generation / cancel."""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending_samples = max(0, self._pending_samples - len(item) // 2)
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

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._queue = queue
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.remote_addr = None

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data, addr) -> None:  # type: ignore[override]
        if self.remote_addr is None:
            self.remote_addr = addr
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def error_received(self, exc) -> None:  # type: ignore[override]
        try:
            self._queue.put_nowait(b"")
        except asyncio.QueueFull:
            pass

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.remote_addr = None
