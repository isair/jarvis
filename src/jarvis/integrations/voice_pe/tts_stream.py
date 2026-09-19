"""TTS → PCM streaming for the Voice PE transport.

Wire contract for a device that declares ``SPEAKER`` and ``API_AUDIO``:
``VOICE_ASSISTANT_TTS_START`` (text), ``VOICE_ASSISTANT_TTS_STREAM_START``,
paced ``send_voice_assistant_audio`` payloads, ``VOICE_ASSISTANT_TTS_STREAM_END``,
``VOICE_ASSISTANT_RUN_END``.

Pacing mirrors the reference satellite: the device-side ring buffer is a fixed
512 ms and is held near 384 ms by the ``(sent - 0.384) - elapsed`` wait, so the
whole reply is never blatted out at once.

Synthesis reuses the already started Jarvis TTS engine (Piper or Chatterbox);
no second TTS stack is created.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from .models import (
    PAYLOAD_BYTES_PER_CHUNK,
    SAMPLE_CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SAMPLES_PER_CHUNK,
    SPEAKER_BUFFER_TARGET_S,
)


# ---------------------------------------------------------------------------
# WAV → PCM
# ---------------------------------------------------------------------------

class WavHeaderError(ValueError):
    """Raised for a WAV container the transport cannot stream."""


class WavHeaderParser:
    """Incremental RIFF/WAVE header parser validated against 16 k mono PCM16."""

    def __init__(
        self,
        expected_channels: int = SAMPLE_CHANNELS,
        expected_width: int = SAMPLE_WIDTH,
        expected_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.expected_channels = expected_channels
        self.expected_width = expected_width
        self.expected_sample_rate = expected_sample_rate
        self.riff_checked = False
        self.fmt_validated = False
        self.found_data = False
        self.data_bytes_remaining = 0

    def parse(self, buffer: bytearray) -> bool:
        """True once the ``data`` chunk header is consumed."""
        while True:
            if not self.riff_checked:
                if len(buffer) < 12:
                    return False
                riff, _, wave = struct.unpack("<4sI4s", buffer[:12])
                if riff != b"RIFF" or wave != b"WAVE":
                    raise WavHeaderError("Invalid WAV header: missing RIFF/WAVE")
                self.riff_checked = True
                del buffer[:12]

            if len(buffer) < 8:
                return False
            chunk_id, chunk_size = struct.unpack("<4sI", buffer[:8])

            if chunk_id == b"fmt ":
                if len(buffer) < 8 + chunk_size + (chunk_size & 1):
                    return False
                if chunk_size < 16:
                    raise WavHeaderError(f"fmt chunk too small: {chunk_size}")
                audio_format, channels, rate, _, _, bits = struct.unpack(
                    "<HHIIHH", buffer[8:24]
                )
                if audio_format != 1:
                    raise WavHeaderError(f"only PCM is streamable, got format {audio_format}")
                if channels != self.expected_channels:
                    raise WavHeaderError(
                        f"expected {self.expected_channels} channels, got {channels}"
                    )
                if rate != self.expected_sample_rate:
                    raise WavHeaderError(
                        f"expected {self.expected_sample_rate} Hz, got {rate} Hz"
                    )
                if bits // 8 != self.expected_width:
                    raise WavHeaderError(
                        f"expected {self.expected_width} bytes per sample, got {bits // 8}"
                    )
                self.fmt_validated = True
                del buffer[: 8 + chunk_size + (chunk_size & 1)]

            elif chunk_id == b"data":
                if not self.fmt_validated:
                    raise WavHeaderError("WAV data chunk before fmt chunk")
                self.data_bytes_remaining = chunk_size
                self.found_data = True
                del buffer[:8]
                return True
            else:
                padded = chunk_size + (chunk_size & 1)
                if len(buffer) < 8 + padded:
                    return False
                del buffer[: 8 + padded]


async def stream_wav(
    stream: AsyncIterator[bytes],
    *,
    expected_channels: int = SAMPLE_CHANNELS,
    expected_width: int = SAMPLE_WIDTH,
    expected_sample_rate: int = SAMPLE_RATE,
    samples_per_chunk: int = SAMPLES_PER_CHUNK,
) -> AsyncIterator[bytes]:
    """Yield PCM payloads of at most ``samples_per_chunk`` samples each."""
    parser = WavHeaderParser(expected_channels, expected_width, expected_sample_rate)
    buffer = bytearray()
    payload_bytes = samples_per_chunk * expected_width * expected_channels

    async for chunk in stream:
        buffer.extend(chunk)
        if not parser.found_data and not parser.parse(buffer):
            continue
        while (
            parser.data_bytes_remaining >= payload_bytes
            and len(buffer) >= payload_bytes
        ):
            payload = bytes(buffer[:payload_bytes])
            del buffer[:payload_bytes]
            parser.data_bytes_remaining -= payload_bytes
            yield payload
        if parser.data_bytes_remaining == 0:
            return

    remaining = min(parser.data_bytes_remaining, len(buffer))
    if remaining > 0:
        yield bytes(buffer[:remaining])


def pcm_from_wav(data: bytes) -> bytes:
    """Whole-buffer WAV → PCM bytes (used by the media-URL fallback)."""
    parser = WavHeaderParser()
    buffer = bytearray(data)
    if not parser.parse(buffer):
        raise WavHeaderError("incomplete WAV header")
    end = min(parser.data_bytes_remaining, len(buffer))
    return bytes(buffer[:end])


def wav_from_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap mono PCM16LE in a RIFF/WAVE container (form the satellite fetches)."""
    pcm = bytes(pcm or b"")
    data_size = len(pcm)
    byte_rate = sample_rate * SAMPLE_CHANNELS * SAMPLE_WIDTH
    header = (
        b"RIFF"
        + struct.pack("<I", 4 + 24 + 8 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack(
            "<HHIIHH", 1, SAMPLE_CHANNELS, sample_rate, byte_rate,
            SAMPLE_CHANNELS * SAMPLE_WIDTH, SAMPLE_WIDTH * 8,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )
    return header + pcm


# ---------------------------------------------------------------------------
# PCM resampling
# ---------------------------------------------------------------------------

def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE) -> bytes:
    """Linear resample of mono PCM16LE; identity when rates match."""
    if not pcm or src_rate <= 0 or src_rate == dst_rate:
        return pcm
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return b""
    out_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    positions = np.linspace(0, samples.size - 1, out_len)
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, samples.size - 1)
    frac = positions - low
    interp = samples[low] * (1.0 - frac) + samples[high] * frac
    return np.clip(interp, -32768, 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Synthesis on the existing engine
# ---------------------------------------------------------------------------

def synthesize_pcm(engine, text: str) -> Optional[bytes]:
    """Return resampled 16 kHz mono PCM16 for ``text`` from the shared engine."""
    if engine is None or not text or not text.strip():
        return None

    try:
        from jarvis.output.tts import _preprocess_for_speech

        prepared = _preprocess_for_speech(text)
    except Exception:
        prepared = text.strip()

    pcm: Optional[bytes] = None
    src_rate = SAMPLE_RATE

    voice = getattr(engine, "_voice", None)
    if voice is not None:
        ensure = getattr(engine, "_ensure_initialized", None)
        if callable(ensure) and not ensure():
            return None
        voice = getattr(engine, "_voice", None)
        if voice is None:
            return None
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            speaker_id=getattr(engine, "speaker", None),
            length_scale=float(getattr(engine, "length_scale", 1.0)),
            noise_scale=float(getattr(engine, "noise_scale", 0.667)),
            noise_w_scale=float(getattr(engine, "noise_w", 0.8)),
        )
        synthesis_lock = getattr(engine, "_synthesis_lock", None)
        with synthesis_lock if synthesis_lock is not None else nullcontext():
            parts = [
                chunk.audio_int16_array
                for chunk in voice.synthesize(prepared, syn_config)
                if getattr(chunk, "audio_int16_array", None) is not None
            ]
        if not parts:
            return None
        src_rate = int(getattr(engine, "_sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        try:
            import numpy as np

            pcm = np.concatenate(parts).astype(np.int16).tobytes()
        except Exception:
            return None
        return resample_pcm16(pcm, src_rate)

    model = getattr(engine, "_model", None)
    if model is not None:
        ensure = getattr(engine, "_ensure_initialized", None)
        if callable(ensure) and not ensure():
            return None
        model = getattr(engine, "_model", None)
        if model is None:
            return None
        wav = model.generate(prepared)
        src_rate = int(getattr(model, "sr", SAMPLE_RATE) or SAMPLE_RATE)
        try:
            import numpy as np

            array = wav.detach().cpu().numpy().reshape(-1)
            pcm = np.clip(array * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        except Exception:
            return None
        return resample_pcm16(pcm, src_rate)

    return None


# ---------------------------------------------------------------------------
# Paced streaming
# ---------------------------------------------------------------------------

#: One worker for the whole stack: synthesis is CPU-bound, ordered and must not
#: occupy the ESPHome event loop, which also carries keepalive, button events,
#: mute and media-state callbacks.
_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice_pe_tts")


async def synthesize_pcm_async(engine, text: str) -> Optional[bytes]:
    """``synthesize_pcm`` off the event loop, on the dedicated executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_TTS_EXECUTOR, synthesize_pcm, engine, text)


def split_sentences(text: str) -> list:
    """Sentence-sized pieces of one reply, for lower first-chunk latency."""
    if not text:
        return []
    parts: list = []
    current = ""
    for char in text:
        current += char
        if char in ".?!":
            parts.append(current.strip())
            current = ""
    if current.strip():
        parts.append(current.strip())
    return [part for part in parts if part]


async def synthesize_sentences_async(engine, text: str) -> list:
    """Per-sentence PCM fragments in order; empty ones are dropped."""
    fragments: list = []
    for sentence in split_sentences(text):
        pcm = await synthesize_pcm_async(engine, sentence)
        if pcm:
            fragments.append(pcm)
    return fragments


async def stream_pcm_paced(
    client, pcm: bytes, *, is_active=None
) -> int:
    """Send PCM in 512-sample chunks, keeping the device buffer near 384 ms."""
    if not pcm:
        return 0

    loop = asyncio.get_running_loop()
    seconds_in_chunk = SAMPLES_PER_CHUNK / SAMPLE_RATE
    audio_duration_sent = 0.0
    start_time: Optional[float] = None
    sent_chunks = 0

    for offset in range(0, len(pcm), PAYLOAD_BYTES_PER_CHUNK):
        if is_active is not None and not is_active():
            return sent_chunks
        payload = pcm[offset: offset + PAYLOAD_BYTES_PER_CHUNK]
        if not payload:
            break
        client.send_voice_assistant_audio(payload)
        sent_chunks += 1
        if start_time is None:
            start_time = loop.time()
        audio_duration_sent += seconds_in_chunk
        is_last = offset + PAYLOAD_BYTES_PER_CHUNK >= len(pcm)
        if is_last:
            break
        wait_s = (audio_duration_sent - SPEAKER_BUFFER_TARGET_S) - (
            loop.time() - start_time
        )
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    return sent_chunks


# ---------------------------------------------------------------------------
# LAN HTTP egress (firmware without the ``SPEAKER`` bit)
# ---------------------------------------------------------------------------

#: Number of WAV payloads kept for fetching; the satellite pulls one per run.
HTTP_PAYLOAD_KEEP = 4


class TtsHttpServer:
    """Minimal HTTP/1.1 server for the synthesized TTS WAVs.

    Used when ``API_AUDIO`` is set but ``SPEAKER`` is not: the reply is
    synthesized here, published under ``/<key>`` and the satellite is handed the
    LAN URL in ``VOICE_ASSISTANT_TTS_END``. One instance per device, living on
    the manager loop; it never owns a thread of its own.
    """

    def __init__(self) -> None:
        self._server: Optional[asyncio.AbstractServer] = None
        self._payloads: "dict[str, bytes]" = {}
        self.port = 0
        #: Number of answered GETs, for ``(server, device)`` health checks.
        self.requests = 0
        #: Answered GETs per payload key, and the count of missing keys.
        self.hits: "dict[str, int]" = {}
        self.missing = 0
        #: Status code seen per key, bytes handed out per key, and the one
        #: content type this server answers with. Together with ``hits`` these
        #: let one generation be traced end to end: served, fetched, complete.
        self.status_codes: "dict[str, int]" = {}
        self.served_bytes: "dict[str, int]" = {}
        #: Stored WAV size per key, so a non-empty payload can be proven.
        self.sizes: "dict[str, int]" = {}
        self.content_type = "audio/wav"

    async def start(self) -> int:
        """Bind an ephemeral IPv4 port and return it.

        ``0.0.0.0`` is spelled out because the dual-stack ``None`` form creates
        one listening socket per family with different ports, and the URL in
        ``TTS_END`` must carry the port of the socket that is actually bound.
        """
        if self._server is not None:
            return self.port
        self._server = await asyncio.start_server(self._handle, "0.0.0.0", 0)
        sockets = getattr(self._server, "sockets", None) or []
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        return self.port

    async def stop(self) -> None:
        server, self._server = self._server, None
        self.port = 0
        if server is None:
            return
        server.close()
        try:
            await server.wait_closed()
        except Exception:  # pragma: no cover - closed already
            pass

    def put(self, key: str, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        """Store one payload and return the path it is served under."""
        self._payloads[key] = wav_from_pcm(pcm, sample_rate)
        self.sizes[key] = len(self._payloads[key])
        while len(self._payloads) > HTTP_PAYLOAD_KEEP:
            oldest = next(iter(self._payloads))
            self._payloads.pop(oldest)
            self.sizes.pop(oldest, None)
        return f"/{key}"

    def hit_count(self, key: str) -> int:
        """Number of answered GETs for exactly this payload key."""
        return int(self.hits.get(key, 0))

    def payload_bytes(self, key: str) -> int:
        """Size of the stored WAV for one key, 0 when nothing is stored."""
        return int(self.sizes.get(key, 0))

    def delivery_for(self, key: str) -> dict:
        """Everything answered for one key: hits, status, bytes, content type."""
        return {
            "key": str(key),
            "stored_bytes": int(self.sizes.get(key, 0)),
            "hits": int(self.hits.get(key, 0)),
            "status": self.status_codes.get(key),
            "served_bytes": int(self.served_bytes.get(key, 0)),
            "content_type": self.content_type,
            "port": int(self.port),
        }

    async def _handle(self, reader, writer) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
        except Exception:
            request = b""
        parts = request.split()
        path = ""
        if len(parts) >= 2:
            path = parts[1].decode("latin-1", "replace").lstrip("/")
        body = self._payloads.get(path, b"")
        # Per-key accounting: an unrelated or missing path must not be able to
        # satisfy the check for a specific media key.
        self.requests += 1
        status = 200 if body else 404
        if body:
            self.hits[path] = int(self.hits.get(path, 0)) + 1
            self.served_bytes[path] = int(self.served_bytes.get(path, 0)) + len(body)
        else:
            self.missing += 1
        if path:
            self.status_codes[path] = int(status)
        head = (
            f"HTTP/1.1 {status} {'OK' if body else 'Not Found'}\r\n"
            f"Content-Type: {self.content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        try:
            writer.write(head + body)
            await writer.drain()
        except Exception:  # pragma: no cover - peer vanished
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


def lan_ip_for(host: str, port: int) -> str:
    """Local interface address facing the device, for the TTS base URL."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, int(port)))
        return str(sock.getsockname()[0])
    except Exception:
        return str(host)
    finally:
        sock.close()
