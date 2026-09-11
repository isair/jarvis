"""Microphone audio ingress for the Voice PE transport.

One bounded path per connection. The device pushes PCM16LE chunks in, the pump
hands them to the existing Jarvis ``VoiceListener`` queue, which owns the VAD,
the endpointing and Whisper. No second STT stack is created here.

Channel plan of the retail device (multi-channel firmware):
- channel 0: enhanced (XMOS processed) speech audio;
- channel 1: less processed audio.
Both are PCM16LE, 16 kHz, mono.

Every queued item names the full stream it belongs to, so two satellites that
both number their first run ``1`` stay distinguishable and stale audio cannot be
appended to a newer run:

- ``AudioFrame(stream, source, samples)`` for one PCM block;
- ``EndOfStream(stream, source)`` for the microphone end of that stream.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .models import (
    AUDIO_SOURCE_VOICE_PE,
    LOCAL_STREAM,
    SAMPLE_RATE,
    AudioFrame,
    StreamId,
    VoicePEConfig,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard requirement
    np = None  # type: ignore[assignment]

#: 16 kHz mono 16-bit: one millisecond is 16 samples.
SAMPLES_PER_MS = SAMPLE_RATE / 1000.0

#: Smallest useful budget, in samples (one 20 ms VAD frame at 16 kHz).
SAMPLES_PER_CHUNK_MIN = 320


@dataclass(frozen=True)
class EndOfStream:
    """In-queue marker: this stream's PCM is complete.

    It travels in the same FIFO as the PCM blocks, so the pump can only see it
    after every block of the run has been handed to the listener queue. It is
    weightless: only the PCM blocks count against the sample budget, so the
    marker is never refused and never displaces audio.
    """

    stream: StreamId
    source: str = AUDIO_SOURCE_VOICE_PE


def pcm16_to_float32(payload: bytes):
    """Convert PCM16LE bytes to the float32 array shape the listener expects."""
    if np is None:  # pragma: no cover
        return payload
    data = np.frombuffer(payload, dtype=np.int16)
    if data.ndim > 1:
        data = data.flatten()
    return (data.astype(np.float32) / 32768.0).reshape(-1)


class AudioIngress:
    """Bounded microphone queue: PCM frames plus one weightless EOS marker."""

    def __init__(self, listener, config: VoicePEConfig, metrics: dict) -> None:
        self._listener = listener
        self._config = config
        self._metrics = metrics
        budget_samples = max(
            SAMPLES_PER_CHUNK_MIN, config.audio_queue_ms * SAMPLES_PER_MS
        )
        self._budget_samples = int(budget_samples)
        #: FIFO of ``AudioFrame`` and ``EndOfStream`` items, in arrival order.
        self._items: deque = deque()
        self._pending_samples = 0
        self._closed = False
        self._stream: StreamId = LOCAL_STREAM
        self._wake: Optional[asyncio.Event] = None

    # -- stream identity --------------------------------------------------

    def set_stream(self, stream: StreamId) -> StreamId:
        """Bind the queue to one satellite stream and return it."""
        self._stream = StreamId(
            str(stream.device_id),
            int(stream.connection_generation),
            int(stream.session_generation),
        )
        return self._stream

    @property
    def stream(self) -> StreamId:
        return self._stream

    # -- producer side (ESPHome callback, same loop as the pump) ----------

    def _wake_pump(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def _trim_to_budget(self) -> None:
        """Drop the oldest PCM blocks in place; markers keep their position."""
        index = 0
        while self._pending_samples > self._budget_samples and index < len(self._items):
            item = self._items[index]
            if isinstance(item, EndOfStream):
                index += 1
                continue
            del self._items[index]
            self._pending_samples = max(
                0, self._pending_samples - len(item.samples) // 2
            )
            self._metrics["audio_dropped_chunks"] = (
                int(self._metrics.get("audio_dropped_chunks", 0)) + 1
            )

    def push_frame(self, payload: bytes, stream: Optional[StreamId] = None) -> None:
        """Append one stamped PCM block, the single writer of the queue."""
        if self._closed or not payload:
            return
        active = stream or self._stream
        samples = len(payload) // 2
        self._pending_samples += samples
        self._items.append(AudioFrame(active, AUDIO_SOURCE_VOICE_PE, payload))
        self._trim_to_budget()
        self._metrics["audio_chunks"] = int(self._metrics.get("audio_chunks", 0)) + 1
        self._metrics["audio_bytes"] = int(self._metrics.get("audio_bytes", 0)) + len(
            payload
        )
        self._wake_pump()

    def put(
        self,
        data: bytes,
        data2: Optional[bytes] = None,
        stream: Optional[StreamId] = None,
    ) -> None:
        """Never-blocking ingest from ``handle_audio``, stamped with a stream."""
        payload = data
        if (
            self._config.preferred_input_channel == 1
            and data2 is not None
            and len(data2) > 0
        ):
            payload = data2
        self.push_frame(payload, stream)

    def mark_end_of_stream(
        self, stream: StreamId, source: str = AUDIO_SOURCE_VOICE_PE
    ) -> None:
        """Append the EOS marker of one stream behind its queued PCM blocks."""
        if self._closed:
            return
        # Weightless: the marker takes no budget slot and is never refused.
        self._items.append(EndOfStream(stream, source))
        self._wake_pump()

    # -- consumer side (pump task) ----------------------------------------

    async def _next_item(self):
        """Pop the next item, waiting on the event only when the queue is empty.

        The flag is cleared on use, in this order: pop-and-clear when an item is
        there, clear-and-retry when only a stale flag is there, and only then
        suspend. Every push appends and sets, so an empty queue with a cleared
        flag is exactly "nothing pending", and each pass of the loop does one
        unit of work instead of spinning on an already-set flag.
        """
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
                # FIFO: every PCM block of this stream is already on the
                # listener queue, so the stream is closed in order here.
                self._close_stream(item)
                continue
            if not item.samples:
                # The UDP path states the microphone end with one empty block.
                self._close_stream(EndOfStream(item.stream, item.source))
                continue
            self._pending_samples = max(
                0, self._pending_samples - len(item.samples) // 2
            )
            self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
            delivered = AudioFrame(
                item.stream, item.source, pcm16_to_float32(item.samples)
            )
            try:
                self._listener._audio_q.put_nowait(delivered)
            except Exception:
                # The listener queue is bounded too: freshest audio wins.
                self._metrics["audio_dropped_chunks"] = (
                    int(self._metrics.get("audio_dropped_chunks", 0)) + 1
                )

    def _close_stream(self, marker: "EndOfStream") -> None:
        """State the end of this stream on the shared listener queue."""
        self._metrics["microphone_queue_depth_ms"] = self.depth_ms()
        close = getattr(self._listener, "pad_until_endpoint", None)
        if not callable(close):
            return
        try:
            close(marker.stream, marker.source)
        except Exception:
            pass

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Drop stale items for a new session or a cancel, in place."""
        self._items.clear()
        self._pending_samples = 0
        self._wake_pump()

    def close(self) -> None:
        self._closed = True
        self.reset()

    def depth_ms(self) -> float:
        return round(self._pending_samples / SAMPLES_PER_MS, 1)

    def is_empty(self) -> bool:
        return not self._items

    def items(self) -> list:
        """Snapshot of the queued items, for the health view and the tests."""
        return list(self._items)

    def queue(self) -> "AudioIngress":
        """The ingress itself, shared with the UDP fallback receiver."""
        return self


class UdpAudioServer(asyncio.DatagramProtocol):
    """UDP microphone receiver for firmware without the API_AUDIO flag."""

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
        self._ingress.push_frame(data or b"")

    def error_received(self, exc) -> None:  # type: ignore[override]
        self._ingress.push_frame(b"")

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.remote_addr = None
