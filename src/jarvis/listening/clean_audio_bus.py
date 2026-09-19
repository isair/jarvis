"""Canonical continuous CleanAudioBus: the single post-AEC3, pre-VAD fan-out.

Every microphone source (the local USB-headset WASAPI lane and every Voice PE
satellite host AEC lane) publishes its *cleaned* PCM into one shared, bounded
bus. Consumers subscribe independently and each gets its own fixed-capacity
SPSC ring with newest-real-time semantics:

* the VAD/Whisper path consumes its 16 kHz copy directly from the listener
  queue — the frames on the bus are the very same cleaned blocks;
* the virtual microphone publisher (``jarvis.output.virtual_microphone``)
  keeps the canonical 48 kHz stream continuous for the ``Toustovač Clean
  Microphone`` endpoint;
* diagnostics may attach a third reader.

A blocked or slow consumer must never stall the native capture/AEC thread or
another consumer: each ring has a fixed capacity (the default covers ~1 s at
480 frames of 10 ms) and drops its oldest frame on overflow, counting the
drop. When no admissible source frame exists, the publisher side pushes an
explicit silence frame on the same monotonic QPC clock — the last speech
frame is never repeated.

The bus carries no DSP and no resampling decisions; it is a plain,
lock-friendly hand-off structure for frames that the native AEC engine has
already produced.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

#: Canonical bus format (matches the kernel-side PCM ring exactly).
BUS_SAMPLE_RATE = 48000
BUS_FRAME_SAMPLES = 480
BUS_FRAME_MS = 10
#: One fixed capacity per consumer ring, in 10 ms frames (~1 s).
RING_CAPACITY = 128

AEC_STATE_LABELS = (
    "disabled",
    "acquiring",
    "converged",
    "double_talk",
    "reconverging",
    "failed",
)


def qpc_timestamp_100ns() -> int:
    """Monotonic clock in 100 ns units (Windows QPC resolution matches)."""
    return time.perf_counter_ns() // 100


@dataclass(frozen=True, slots=True)
class CleanAudioFrame:
    """One canonical 10 ms cleaned block on the bus.

    ``samples_48k_f32`` is a little-endian float32 mono ``memoryview`` of
    exactly :data:`BUS_FRAME_SAMPLES` samples. ``sequence`` is monotonic per
    source; ``discontinuity`` marks the first frame of a new generation or a
    gap where frames were dropped.
    """

    source_id: str
    source_kind: Literal["local_usb", "voice_pe"]
    connection_generation: int
    session_generation: int
    sequence: int
    qpc_timestamp_100ns: int
    samples_48k_f32: memoryview
    aec_state: str
    reference_active: bool
    discontinuity: bool


class _SpscRing:
    """Fixed-capacity single-producer/single-consumer ring of frames."""

    __slots__ = ("_items", "capacity", "dropped")

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque = deque()
        self.dropped = 0

    def push(self, frame: CleanAudioFrame) -> None:
        # Newest-real-time semantics: on overflow the oldest frame loses.
        if len(self._items) >= self.capacity:
            self._items.popleft()
            self.dropped += 1
        self._items.append(frame)

    def pop(self) -> Optional[CleanAudioFrame]:
        if self._items:
            return self._items.popleft()
        return None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)


class CleanAudioBus:
    """One producer per source, N independent bounded consumers."""

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        self._capacity = max(1, int(capacity))
        self._rings: dict[str, _SpscRing] = {}
        self._sequence: dict[str, int] = {}
        self._published = 0
        self._silence_published = 0

    # -- subscription ------------------------------------------------------

    def subscribe(self, consumer_id: str) -> str:
        """Attach (or re-attach) one bounded consumer ring."""
        ring = self._rings.get(consumer_id)
        if ring is None:
            ring = _SpscRing(self._capacity)
            self._rings[consumer_id] = ring
        return consumer_id

    def unsubscribe(self, consumer_id: str) -> None:
        self._rings.pop(consumer_id, None)
        self._sequence.pop(consumer_id, None)

    # -- production --------------------------------------------------------

    def publish(self, frame: CleanAudioFrame) -> None:
        """Fan one frame out to every attached ring, never blocking."""
        self._published += 1
        for ring in self._rings.values():
            ring.push(frame)

    def publish_frame(
        self,
        samples,
        *,
        source_id: str,
        source_kind: str,
        connection_generation: int = 0,
        session_generation: int = 0,
        aec_state: str = "acquiring",
        reference_active: bool = False,
        discontinuity: bool = False,
    ) -> CleanAudioFrame:
        """Wrap one raw float array as a canonical frame and publish it."""
        view = memoryview(samples) if not isinstance(samples, memoryview) else samples
        sequence = int(self._sequence.get(source_id, 0)) + 1
        self._sequence[source_id] = sequence
        frame = CleanAudioFrame(
            source_id=str(source_id),
            source_kind=source_kind,  # type: ignore[arg-type]
            connection_generation=int(connection_generation),
            session_generation=int(session_generation),
            sequence=sequence,
            qpc_timestamp_100ns=qpc_timestamp_100ns(),
            samples_48k_f32=view,
            aec_state=str(aec_state),
            reference_active=bool(reference_active),
            discontinuity=bool(discontinuity),
        )
        self.publish(frame)
        return frame

    def publish_silence(
        self,
        source_id: str,
        *,
        source_kind: str = "local_usb",
        connection_generation: int = 0,
        session_generation: int = 0,
        aec_state: str = "failed",
    ) -> CleanAudioFrame:
        """Publish explicit digital silence on the same monotonic clock."""
        import numpy as np

        self._silence_published += 1
        return self.publish_frame(
            np.zeros(BUS_FRAME_SAMPLES, dtype=np.float32),
            source_id=source_id,
            source_kind=source_kind,
            connection_generation=connection_generation,
            session_generation=session_generation,
            aec_state=aec_state,
            reference_active=False,
            discontinuity=False,
        )

    # -- consumption -------------------------------------------------------

    def read(self, consumer_id: str) -> Optional[CleanAudioFrame]:
        ring = self._rings.get(consumer_id)
        if ring is None:
            return None
        return ring.pop()

    # -- telemetry ---------------------------------------------------------

    def status(self) -> dict:
        return {
            "published_frames": int(self._published),
            "silence_frames": int(self._silence_published),
            "consumers": {
                consumer_id: {
                    "depth_ms": int(len(ring)) * BUS_FRAME_MS,
                    "max_depth_ms": int(ring.capacity) * BUS_FRAME_MS,
                    "dropped_oldest": int(ring.dropped),
                }
                for consumer_id, ring in self._rings.items()
            },
        }


def upsample_to_48k(samples):
    """Integer ×3 linear interpolation of a 16 kHz mono float block."""
    import numpy as np

    vec = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = int(vec.size)
    if n == 0:
        return np.zeros(BUS_FRAME_SAMPLES, dtype=np.float32)
    idx = np.arange(n * 3, dtype=np.float64) / 3.0
    out = np.interp(idx, np.arange(n, dtype=np.float64), vec.astype(np.float64))
    return out.astype(np.float32)


def to_16k(frame: CleanAudioFrame):
    """The 16 kHz view of one canonical frame (exact ×3 decimation)."""
    import numpy as np

    vec = np.asarray(frame.samples_48k_f32, dtype=np.float32).reshape(-1)
    if vec.size % 3:
        vec = vec[: vec.size - (vec.size % 3)]
    return vec.reshape(-1, 3).mean(axis=1).astype(np.float32)


#: The process-wide bus instance, created once at first use.
_bus: Optional[CleanAudioBus] = None


def get_bus() -> CleanAudioBus:
    """Return the shared :class:`CleanAudioBus` instance."""
    global _bus
    if _bus is None:
        _bus = CleanAudioBus()
    return _bus


def make_silence_array():
    """One canonical 480-sample float32 zero block."""
    import numpy as np

    return np.zeros(BUS_FRAME_SAMPLES, dtype=np.float32)
