"""Windows virtual-microphone side of the Toustovač audio path.

The CleanAudioBus (``jarvis.listening.clean_audio_bus``) is the canonical
in-process producer; this module is the consumer that keeps the kernel-side
``Toustovač Clean Microphone`` ring fed through the named-pipe broker:

* one configured source at a time (``local`` or ``voice_pe:<mac>``);
* explicit fail-closed silence while the source is not ready, or on mute,
  source-offline and lane-discontinuity boundaries (never a repeated last
  frame);
* a new producer generation on source switch / AEC discontinuity / Voice PE
  connection- or session-generation change / broker reconnect;
* telemetry: frames, drops, sequence gaps, stale packets and the observed
  p50/p95/max publish-to-DPC latency in milliseconds.

The wire format is the versioned binary ``TvmicPacketV1`` header followed by
little-endian int16 PCM (or compact JSON for the two control messages), with
CRC32C over the payload on every audio packet.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from typing import Any, Optional

from ..listening.clean_audio_bus import (
    BUS_FRAME_SAMPLES,
    BUS_SAMPLE_RATE,
    get_bus,
)

#: Named-pipe identity shared by driver, broker and this worker.
PIPE_NAME = r"\\.\pipe\ToustovacCleanMic.v1"

#: Driver package version tracked against the DriverVer in
#: ``native/virtual_mic/package/ToustovacVirtualMic.inf``. A mismatch with
#: the version installed on the OS triggers an in-place package update.
PACKAGE_VERSION = "1.0.0.0"

#: Binary packet: struct_size, protocol, producer_generation, sequence,
#: QPC timestamp (100 ns), sample rate, channels, bits, sample_count,
#: flags, payload_crc32c -> 52 bytes.
_PACKET = struct.Struct("<IIqqqIHHiII")
PACKET_SIZE = _PACKET.size  # 52

FLAG_LOCAL = 0x01
FLAG_SILENCE = 0x02
FLAG_DISCONTINUITY = 0x04
FLAG_MUTED = 0x08
FLAG_FIRST_OF_GENERATION = 0x10

_STATES = (
    "disabled",
    "driver_missing",
    "broker_offline",
    "source_offline",
    "aec_acquiring",
    "streaming",
    "muted",
    "degraded",
)
#: Status priority for the single ``status.state`` field.
_PRIORITY = {
    "disabled": 0,
    "driver_missing": 1,
    "broker_offline": 2,
    "source_offline": 3,
    "aec_acquiring": 4,
    "degraded": 5,
    "muted": 6,
    "streaming": 7,
}

_CRC_TABLE: list = []


def _crc32c_table() -> list:
    if _CRC_TABLE:
        return _CRC_TABLE
    poly = 0x82F63B78  # reflected CRC-32C polynomial
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ (poly if crc & 1 else 0)
        _CRC_TABLE.append(crc)
    return _CRC_TABLE


def crc32c(data: bytes) -> int:
    """Hardware-parity CRC-32C (Castagnoli) over ``data``."""
    table = _crc32c_table()
    crc = 0xFFFFFFFF
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


class VirtualMicrophonePublisher:
    """Continuous 48 kHz publisher for the Clean-Microphone endpoint."""

    def __init__(self, cfg, listener: Any) -> None:
        self._cfg = cfg
        self._listener = listener
        self.enabled = bool(getattr(cfg, "virtual_microphone_enabled", False))
        self._source = str(
            getattr(cfg, "virtual_microphone_source", "") or ""
        ).strip().lower()
        self._fail_closed = bool(
            getattr(cfg, "virtual_microphone_fail_closed", True)
        )
        self._publish_unconverged = bool(
            getattr(cfg, "virtual_microphone_publish_unconverged", False)
        )
        self._mic_name = str(
            getattr(cfg, "virtual_microphone_name", "") or "Toustovač Clean Microphone"
        )

        self._bus = get_bus()
        self._consumer_id = "virtual_microphone"
        self._bus.subscribe(self._consumer_id)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipe: Any = None
        self._pipe_lock = threading.Lock()

        self.producer_generation = 0
        self.sequence = 0
        self.frames_produced = 0
        self.silence_frames = 0
        self.sequence_gaps = 0
        self._last_seq: dict[str, int] = {}

        self._latencies_ms: deque = deque(maxlen=200)
        self._reconnects = 0
        self._state = "disabled"
        self._muted = False
        self._last_frame_ns = 0
        self._last_source_id = ""
        self._last_aec_state = ""
        self._last_generation_pair = (0, 0)
        self._connected = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="ToustovacCleanMic", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_pipe()
        self._bus.unsubscribe(self._consumer_id)
        self._state = "disabled"

    def set_muted(self, muted: bool) -> None:
        """Soft mute from the UI/manager; hardware mute is mirrored by the
        per-device lease and takes precedence through ``_muted_now``."""
        self._muted = bool(muted)

    # -- source helpers -------------------------------------------------------

    def _target_source_id(self) -> str:
        if self._source == "local":
            return "local"
        if self._source.startswith("voice_pe:"):
            # The bus carries the bare satellite id as ``source_id``.
            return self._source.split(":", 1)[1].strip().lower()
        if self._source == "voice_pe":
            return "voice_pe"
        return ""

    def _selected_stream_of(self, source_id: str) -> Optional[tuple]:
        if not source_id or source_id == "local":
            return None
        try:
            from ..integrations.voice_pe import get_manager

            manager = get_manager()
            if manager is None:
                return None
            for device in getattr(manager, "devices", ()) or ():
                identity = getattr(device, "identity", {}) or {}
                mac = str(identity.get("mac_address", "")).upper().replace(":", "")
                if mac == source_id.upper().replace(":", ""):
                    ingress = getattr(device, "_ingress", None)
                    if ingress is not None:
                        stream = getattr(ingress, "stream", None)
                        if stream is not None:
                            return (
                                str(stream.device_id),
                                int(stream.connection_generation),
                                int(stream.session_generation),
                            )
        except Exception:
            return None
        return None

    # -- pipe I/O -------------------------------------------------------------

    def _close_pipe(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass

    def _open_pipe(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            handle = open(PIPE_NAME, "r+b", buffering=0)
        except FileNotFoundError:
            self._state = "driver_missing"
            return False
        except Exception:
            self._state = "broker_offline"
            return False
        self._pipe = handle
        self._connected = True
        return True

    def _write_raw(self, payload: bytes) -> bool:
        if self._pipe is None:
            return False
        try:
            self._pipe.write(payload)
            return True
        except Exception:
            self._close_pipe()
            self._connected = False
            self._state = "broker_offline"
            return False

    def _read_exact(self, count: int) -> Optional[bytes]:
        if self._pipe is None:
            return None
        try:
            data = self._pipe.read(count)
        except Exception:
            data = b""
        if not data:
            return None
        return data

    def _negotiate(self) -> bool:
        body = json.dumps(
            {
                "v": 1,
                "sample_rate": BUS_SAMPLE_RATE,
                "channels": 1,
                "bits": 16,
                "frame_samples": BUS_FRAME_SAMPLES,
                "mic_name": self._mic_name,
            }
        ).encode("ascii")
        packet = _PACKET.pack(PACKET_SIZE, 1, self.producer_generation, 0, 0, 0, 1, 16, 0, 0, 0)
        payload = packet + body
        header = struct.pack("<I", len(payload))
        if not self._write_raw(header + payload):
            return False
        raw = self._read_exact(4)
        if raw is None:
            return False
        (size,) = struct.unpack("<I", raw)
        raw = self._read_exact(int(size))
        if raw is None:
            return False
        try:
            ok = json.loads(raw[PACKET_SIZE:]) if len(raw) > PACKET_SIZE else {}
        except Exception:
            ok = {}
        return bool(ok.get("ok", True))

    def _heartbeat(self) -> bool:
        body = json.dumps(
            {
                "v": 1,
                "state": self._state,
                "sequence": self.sequence,
                "frames_produced": self.frames_produced,
                "silence_frames": self.silence_frames,
            }
        ).encode("ascii")
        packet = _PACKET.pack(
            PACKET_SIZE, 1, self.producer_generation,
            self.sequence, time.perf_counter_ns() // 100,
            0, 1, 16, 0, 0, 0,
        )
        payload = packet + body
        return self._write_raw(struct.pack("<I", len(payload)) + payload)

    def _query_driver(self) -> dict:
        """One QUERY_STATUS-style request/reply on the same pipe."""
        packet = _PACKET.pack(
            PACKET_SIZE, 1, self.producer_generation,
            self.sequence, time.perf_counter_ns() // 100,
            0, 1, 16, 0, 0, 0,
        )
        payload = packet + b'{"query":1}'
        if not self._write_raw(struct.pack("<I", len(payload)) + payload):
            return {}
        raw = self._read_exact(4)
        if raw is None:
            return {}
        (size,) = struct.unpack("<I", raw)
        raw = self._read_exact(int(size))
        if raw is None or len(raw) <= PACKET_SIZE:
            return {}
        try:
            return json.loads(raw[PACKET_SIZE:])
        except Exception:
            return {}

    def _send_packet(
        self, pcm16: bytes, aec_state: str, reference_active: bool, flags: int
    ) -> bool:
        samples = len(pcm16) // 2
        packet = _PACKET.pack(
            PACKET_SIZE,
            1,
            self.producer_generation,
            self.sequence,
            time.perf_counter_ns() // 100,
            BUS_SAMPLE_RATE,
            1,
            16,
            int(samples),
            flags,
            crc32c(pcm16),
        )
        payload = packet + pcm16
        return self._write_raw(struct.pack("<I", len(payload)) + payload)

    # -- state machinery ------------------------------------------------------

    def _next_generation(self) -> None:
        self.producer_generation += 1
        self.sequence = 0
        self._last_seq.clear()

    def _muted_now(self) -> bool:
        try:
            sink = getattr(self._listener, "_voice_pe_sink", None)
            if sink is not None:
                for device in getattr(sink, "devices", ()) or ():
                    lease = getattr(device, "lease", None)
                    if lease is not None and lease.muted:
                        return True
        except Exception:
            pass
        return bool(self._muted)

    def _frame_admissible(self, frame) -> bool:
        converged = str(frame.aec_state) == "converged"
        if self._publish_unconverged:
            return True
        if self._fail_closed:
            return bool(frame.reference_active) and converged
        return True

    # -- worker ---------------------------------------------------------------

    def _run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            try:
                self._step(tick)
            except Exception:
                self._state = "degraded"
            # 100 ms cadence = 10 frames per tick of nominal 10 ms frames.
            time.sleep(0.01)
            tick += 1

    def _apply_clients(self, count: int) -> None:
        """Feed one capture-client count into every desktop lease."""
        try:
            from ..integrations.voice_pe import get_manager

            manager = get_manager()
            if manager is not None:
                for device in getattr(manager, "devices", ()) or ():
                    device.notify_capture_clients(count)
        except Exception:
            pass

    def _step(self, tick: int) -> None:
        if not self._connected:
            if not self._open_pipe():
                return
            if not self._negotiate():
                self._close_pipe()
                self._reconnects += 1
                self._next_generation()
                return
            self._next_generation()  # first generation on the new pipe
        try:
            self._pump_once()
            status = self._query_driver()
            if isinstance(status, dict):
                try:
                    self._apply_clients(
                        int(status.get("active_capture_clients", 0) or 0)
                    )
                except (TypeError, ValueError):
                    pass
        except Exception:
            self._close_pipe()
            self._reconnects += 1
            self._next_generation()
            return
        if tick % 10 == 0:  # >= 2 Hz heartbeat, bounded size
            if not self._heartbeat():
                self._next_generation()

    def _pump_once(self) -> None:
        target = self._target_source_id()
        frames = 0
        while frames < 24:
            frame = self._bus.read(self._consumer_id)
            if frame is None:
                break
            frames += 1
            if str(frame.source_id).lower() != target and target != "":
                # Frames of a non-selected source keep flowing on the bus
                # but never become the published stream.
                continue
            self._last_seq[str(frame.source_id)] = int(frame.sequence)
            gap = int(frame.sequence) - (self.sequence + 1)
            if self.sequence and gap > 0:
                self.sequence_gaps += 1
            self.sequence += 1
            if self._muted_now():
                self._state = "muted"
                self._emit_silence(
                    FLAG_MUTED | (FLAG_FIRST_OF_GENERATION if self.sequence == 1 else 0)
                )
                continue
            admissible = self._frame_admissible(frame)
            if not admissible:
                self._state = (
                    "source_offline"
                    if not frame.reference_active
                    else "aec_acquiring"
                )
                self._emit_silence(FLAG_SILENCE | FLAG_DISCONTINUITY)
                continue
            self._last_aec_state = str(frame.aec_state)
            self._last_generation_pair = (
                int(frame.connection_generation),
                int(frame.session_generation),
            )
            flags = FLAG_LOCAL if frame.source_kind == "local_usb" else 0
            if frame.discontinuity:
                flags |= FLAG_DISCONTINUITY
            self._emit_packet(
                frame.samples_48k_f32,
                flags | (FLAG_FIRST_OF_GENERATION if self.sequence == 1 else 0),
                frame.qpc_timestamp_100ns,
            )
        if frames == 0 and self._connected:
            # No new source frame: keep the 48 kHz stream continuous with an
            # explicit silence on the same monotonic clock.
            if self._muted_now():
                self._state = "muted"
                self._emit_silence(FLAG_MUTED)
            elif target == "":
                self._state = "source_offline"
                self._emit_silence(FLAG_SILENCE)
            elif self._last_aec_state == "converged" and not self._fail_closed:
                self._emit_silence(0)
            else:
                self._state = (
                    self._last_aec_state or "aec_acquiring"
                ) if self._last_aec_state not in ("", "converged") else "streaming"
                if self._state == "streaming":
                    # A full 10 ms tick without new frames is a gap.
                    self._emit_silence(FLAG_SILENCE)

    def _emit_packet(self, samples: memoryview, flags: int, qpc: int) -> None:
        pcm16 = _to_pcm16(samples)
        if not self._send_packet(pcm16, self._last_aec_state, True, flags):
            return
        self.frames_produced += 1
        now = time.perf_counter_ns() // 100
        if qpc:
            self._latencies_ms.append(max(0.0, (now - int(qpc)) / 10000.0))
        self._state = "streaming"

    def _emit_silence(self, flags: int) -> None:
        pcm16 = b"\x00\x00" * BUS_FRAME_SAMPLES
        if not self._send_packet(pcm16, self._last_aec_state or "n/a", False, flags | FLAG_SILENCE):
            return
        self.silence_frames += 1

    # -- introspection --------------------------------------------------------

    #: Fields mirrored into ``status()`` for version tracking (plan §7).
    @property
    def package_version(self) -> str:
        return PACKAGE_VERSION

    @property
    def driver_version(self) -> str:
        """Version reported by the driver itself, ``""`` while offline."""
        return str(getattr(self, "_driver_version", "") or "")

    def status(self) -> dict:
        if not self._latencies_ms:
            p50 = p95 = pmax = 0.0
        else:
            ordered = sorted(self._latencies_ms)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
            pmax = ordered[-1]
        return {
            "state": self._state,
            "source": self._source,
            "mic_name": self._mic_name,
            "package_version": PACKAGE_VERSION,
            "driver_version": self.driver_version,
            "producer_generation": int(self.producer_generation),
            "sequence": int(self.sequence),
            "frames_produced": int(self.frames_produced),
            "silence_frames": int(self.silence_frames),
            "sequence_gaps": int(self.sequence_gaps),
            "stale_packets": int(self.sequence_gaps),
            "reconnects": int(self._reconnects),
            "muted": bool(self._muted),
            "fail_closed": bool(self._fail_closed),
            "publish_unconverged": bool(self._publish_unconverged),
            "latency_ms": {
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "max": round(pmax, 3),
            },
            "bus": self._bus.status(),
            "aec_state": self._last_aec_state,
            "voice_pe_generation": list(self._last_generation_pair),
        }


def _to_pcm16(samples: memoryview) -> bytes:
    """Little-endian int16 conversion of the canonical float32 block."""
    import numpy as np

    vec = np.asarray(samples, dtype=np.float32).reshape(-1)
    scaled = np.rint(np.clip(vec, -1.0, 1.0) * 32767.0)
    return scaled.astype(np.int16).tobytes()


def make_publisher(cfg, listener: Any) -> Optional[VirtualMicrophonePublisher]:
    """Create and start the publisher; ``None`` when the feature is off."""
    publisher = VirtualMicrophonePublisher(cfg, listener)
    if not publisher.enabled:
        return None
    publisher.start()
    return publisher


#: Process-wide instances.
_PUBLISHER: Optional[VirtualMicrophonePublisher] = None


def start_publisher(cfg, listener: Any) -> Optional[VirtualMicrophonePublisher]:
    global _PUBLISHER
    _PUBLISHER = make_publisher(cfg, listener)
    return _PUBLISHER


def stop_publisher() -> None:
    global _PUBLISHER
    if _PUBLISHER is not None:
        _PUBLISHER.stop()
        _PUBLISHER = None


def get_publisher() -> Optional[VirtualMicrophonePublisher]:
    return _PUBLISHER


# --- first-run / update automation (plan sections 7 + 10) ---------------------

_PACKAGE_DIR = "native/virtual_mic/package"


def _candidate_dirs() -> list:
    import os
    import sys as _sys

    dirs: list[str] = []
    for base in (
        getattr(_sys, "_MEIPASS", None),
        _sys.prefix,
        os.getcwd(),
    ):
        if base:
            dirs.append(str(base))
    root = os.environ.get("JARVIS_PROJECT_ROOT")
    if root:
        dirs.append(root)
    return dirs


def _find_file(name: str) -> Optional[str]:
    import os

    rels = (
        name,
        os.path.join("_internal", name),
        os.path.join(_PACKAGE_DIR, name),
        os.path.join("native", "virtual_mic", "package", name),
        os.path.join("native", _PACKAGE_DIR, name),
        os.path.join("dist", "Jarvis", name),
        os.path.join("dist", "Jarvis", "_internal", name),
        os.path.join("build", "virtual_mic", "Release", name),
        os.path.join("build", "virtual_mic", name),
    )
    for base in _candidate_dirs():
        for rel in rels:
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    return None


def _installed_version_from_pnputil() -> Optional[str]:
    """``pnputil /enum-drivers`` scan for our package + DriverVer.

    One block is ``Published name : ...`` + ``Version : x.y`` pairs. A
    version pair with the published-name key inside one block matches the
    original single-regex layout.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["pnputil", "/enum-drivers"], capture_output=True, text=True,
            timeout=10,
        ).stdout or ""
    except Exception:
        return None
    m = re.search(
        r"Published name\s*:.*?ToustovacVirtualMic.*?"
        r"Class\s*:.*?\n\s*Version\s*:\s*([0-9.]+)",
        out,
    )
    return m.group(1).strip() if m else None


def installed_package_version() -> str:
    """Installed package version: pnputil first, then the .inf file."""
    ver = _installed_version_from_pnputil()
    if ver:
        return ver
    inf = _find_file("ToustovacVirtualMic.inf")
    if inf:
        try:
            with open(inf, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip().startswith("DriverVer"):
                        return line.split(",")[-1].strip()
        except OSError:
            pass
    return ""


def _pipe_alive() -> bool:
    try:
        import os

        fd = os.open(PIPE_NAME, os.O_RDWR)
    except OSError:
        return False
    except Exception:
        return False
    try:
        return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def ensure_installed() -> dict:
    """Idempotent install / repair / in-place update of the package.

    * first run: ``pnputil``/devnode absent -> ``ToustovacAudioInstall.exe
      -Install`` stages the package, the devnode and the broker service;
    * every later run: version comparison (installed vs ``PACKAGE_VERSION``)
      drives the same exe - DiInstallDriver is the best-match update and the
      devnode/service steps verify instead of re-creating;
    * a live broker pipe answers before the first pump tick is considered
      healthy, otherwise the same exe re-runs (it re-starts the service or
      recreates it from the staged files).
    """
    installed = installed_package_version()
    result = {"installed": installed, "package": PACKAGE_VERSION,
              "updated": False, "returncode": None}
    needs = bool(installed) and installed != PACKAGE_VERSION
    exe = _find_file("ToustovacAudioInstall.exe")
    if (not installed) or needs or exe is None and not installed:
        if exe is None:
            result["error"] = "ToustovacAudioInstall.exe not found"
            return result
        result = _run_install(exe, result)
    elif not _pipe_alive():
        if exe is None:
            result["error"] = "ToustovacAudioInstall.exe not found"
            return result
        result = _run_install(exe, result, reason="broker_offline")
    result["after"] = installed_package_version()
    result["pipe"] = _pipe_alive()
    return result


def _run_install(exe: str, result: dict, reason: str = "first_run") -> dict:
    import subprocess

    try:
        proc = subprocess.run([exe, "-Install"], capture_output=True,
                              text=True, timeout=20)
        result["returncode"] = proc.returncode
        result["stdout"] = (proc.stdout or "").strip()[:400]
        result["reason"] = reason
        result["updated"] = True
    except Exception as exc:  # pragma: no cover - best effort
        result["error"] = repr(exc)
    return result


def resolve_device_index(cfg) -> Optional[int]:
    """sounddevice index of the endpoint, resolved by name."""
    name = str(
        getattr(cfg, "virtual_microphone_name", "")
        or "Toustovač Clean Microphone"
    ).lower()
    try:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if name in str(dev.get("name", "")).lower():
                return int(idx)
    except Exception:
        return None
    return None


__all__ = [
    "PACKAGE_VERSION",
    "PIPE_NAME",
    "VirtualMicrophonePublisher",
    "crc32c",
    "ensure_installed",
    "get_publisher",
    "installed_package_version",
    "make_publisher",
    "start_publisher",
    "stop_publisher",
]
