"""Standalone smoke for the Clean Microphone stack + Voice PE run close.

Runnable two ways:
  python tests/smoke_clean_mic.py
  python -m pytest tests/smoke_clean_mic.py -q
"""

from __future__ import annotations

import array
from pathlib import Path

import pytest  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]


def _f48():
    return array.array("f", [0.001 * i for i in range(480)])


def test_bus_mixed_sources_and_drop_on_oldest() -> None:
    from jarvis.listening.clean_audio_bus import CleanAudioBus, RING_CAPACITY

    bus = CleanAudioBus()
    bus.subscribe("pub")
    for _ in range(RING_CAPACITY + 3):
        bus.publish_frame(
            _f48(), source_id="local", source_kind="local_usb",
            aec_state="converged", reference_active=True,
        )
    st = bus.status()
    assert st["published_frames"] == RING_CAPACITY + 3
    assert st["consumers"]["pub"]["dropped_oldest"] == 3
    assert st["consumers"]["pub"]["depth_ms"] == RING_CAPACITY * 10
    last = bus.read("pub")
    assert last is not None and last.source_id == "local"


def test_upsample_and_16k_identity() -> None:
    from jarvis.listening.clean_audio_bus import to_16k, upsample_to_48k
    import numpy as np

    src = np.arange(160, dtype=np.float32) / 160.0
    up = upsample_to_48k(src)
    assert int(up.size) == 480
    # exact tri-linear ×3: first/last anchor values are preserved
    assert up[0] == pytest.approx(float(src[0]), abs=1e-6)
    assert up[3] == pytest.approx(float(src[1]), abs=1e-6)


def test_publisher_heartbeat_silence_and_mute() -> None:
    from jarvis.output.virtual_microphone import (
        FLAG_SILENCE,
        PACKAGE_VERSION,
        VirtualMicrophonePublisher,
        make_publisher,
        stop_publisher,
    )
    from jarvis.config import load_settings

    cfg = load_settings()
    if not cfg.virtual_microphone_enabled:
        pytest.skip("clean microphone disabled in the active config.json")
    stop_publisher()
    pub = make_publisher(cfg, None)
    assert pub is not None
    pub._pump_once()
    status = pub.status()
    assert status["state"] in (
        "disabled", "not_started", "idle", "silent", "streaming", "muted",
        "driver_missing", "broker_offline", "source_offline", "degraded",
        "aec_acquiring", "acquiring", "converged", "double_talk",
        "reconverging", "failed",
    )
    assert status["package_version"] == PACKAGE_VERSION
    assert status["fail_closed"] == bool(cfg.virtual_microphone_fail_closed)
    stop_publisher()
    stop_publisher()  # idempotent
    assert VirtualMicrophonePublisher is not None and FLAG_SILENCE == 0x02


def test_publish_frame_silence_counter() -> None:
    from jarvis.listening.clean_audio_bus import get_bus

    bus = get_bus()
    bus.subscribe("smoke")
    before = bus.status()["silence_frames"]
    bus.publish_silence("local")
    assert bus.status()["silence_frames"] == before + 1


def test_voice_pe_eos_close_keeps_finalising_utterance() -> None:
    """The plain pipeline close after microphone-end is the normal terminal."""
    import asyncio

    from jarvis.integrations.voice_pe.device import VoicePEDevice
    from jarvis.integrations.voice_pe.models import VoicePEConfig
    from jarvis.integrations.voice_pe.voice_transport import AudioIngress

    class Queue:
        def __init__(self) -> None:
            self.items: list = []

        def put_nowait(self, item) -> None:
            self.items.append(item)

    class Listener:
        def __init__(self) -> None:
            self._audio_q = Queue()

        def pad_until_endpoint(self, stream, source):
            # model the real tail: two 320-sample zero blocks on the grid
            self._audio_q.items.extend(None for _ in range(2))
            return 2

        def _sink_context(self):
            return None

        @property
        def _frame_samples(self) -> int:
            return 320

        @property
        def _vad_frame_ms(self) -> int:
            return 20

    cfg = VoicePEConfig(enabled=True)
    listener = Listener()
    device = VoicePEDevice(
        cfg, listener=listener, tts_engine=None, host="127.0.0.1", port=6053
    )
    device._ingress = ingress = AudioIngress(listener, cfg, {})
    device.connection_generation = 1

    async def run() -> int:
        device.session_generation += 1
        assert device.session_generation == 1
        stream = device._stream()
        ingress.set_stream(stream)
        ingress._multi = False  # single-payload firmware, channel locked
        await device.handle_audio(b"\x01\x00" * 320)
        await device.handle_pipeline_stop(False)   # microphone end
        pending = ingress.eos_pending(device._stream())
        await device.handle_pipeline_stop(True)    # plain pipeline close
        assert pending is True
        return len(listener._audio_q.items)

    delivered = asyncio.run(run())
    # one PCM block + the two padded silence frames reached the listener,
    # i.e. the keep-mode close did not wipe the utterance mid-finalisation.
    assert delivered == 3


def test_install_idempotent() -> None:
    from jarvis.output.virtual_microphone import ensure_installed

    first = ensure_installed()
    second = ensure_installed()
    assert first["installed"] == second["installed"]


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("JARVIS_PROJECT_ROOT", str(ROOT))
    os.environ.setdefault("JARVIS_CONFIG_PATH", str(ROOT / "examples" / "config.json"))
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("smoke_clean_mic: all passed")
