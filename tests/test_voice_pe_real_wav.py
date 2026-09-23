'''Pipeline tests for the real Piper WAV fixture (Voice PE egress).

``tests/piper_tts_real.wav`` is the byte-exact container the shared Piper engine
produces at runtime for ``"Dobrý den, Toustovač je připraven."`` — mono PCM16 at
``SAMPLE_RATE`` wrapped by the same ``wav_from_pcm`` the transport uses. Unlike
the synthetic fixture in ``test_voice_pe_wav_pipeline.py`` these bytes carry a
real 3.18 s utterance, so the payload grid, the pacing horizon and the LAN byte
accounting are checked against production-sized audio:

* paced native-API PCM (``SPEAKER | API_AUDIO``),
* LAN-WAV ``TTS_END`` egress (``API_AUDIO`` without ``SPEAKER``, flags ``61``),
* the announcement RPC carrying the same WAV for a late reply,
* regeneration from the on-disk Piper model (shape, not bytes: Piper's noise is
  unseeded, so only the container geometry is stable).

No network device is required; the model is opened read-only from the usual
per-user location when present.
'''

import asyncio
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.integrations.voice_pe import config as pe_config
from jarvis.integrations.voice_pe.capabilities import build_snapshot
from jarvis.integrations.voice_pe.device import VoicePEDevice
from jarvis.integrations.voice_pe.entities import EntityIndex
from jarvis.integrations.voice_pe.media import VoicePEMediaController
from jarvis.integrations.voice_pe.models import (
    FEATURE_ANNOUNCE,
    FEATURE_API_AUDIO,
    FEATURE_SPEAKER,
    FEATURE_START_CONVERSATION,
    FEATURE_TIMERS,
    FEATURE_VOICE_ASSISTANT,
    PAYLOAD_BYTES_PER_CHUNK,
    SAMPLES_PER_CHUNK,
    SAMPLE_WIDTH,
    SAMPLE_RATE,
    SessionState,
)
from jarvis.integrations.voice_pe.tts_stream import (
    pcm_from_wav,
    stream_pcm_paced,
    synthesize_pcm,
    wav_from_pcm,
)
from jarvis.integrations.voice_pe.voice_transport import AudioIngress

WAV_PATH = Path(__file__).resolve().parent / 'piper_tts_real.wav'

#: Geometry of the stored real utterance, derived from its own header.
FILE_BYTES = 101840
DATA_BYTES = 101796
TOTAL_SAMPLES = DATA_BYTES // SAMPLE_WIDTH          # 50898
DURATION_S = TOTAL_SAMPLES / SAMPLE_RATE            # 3.181125
FULL_CHUNKS = DATA_BYTES // PAYLOAD_BYTES_PER_CHUNK  # 99
TAIL_BYTES = DATA_BYTES % PAYLOAD_BYTES_PER_CHUNK    # 420
CHUNK_GRID = [PAYLOAD_BYTES_PER_CHUNK] * FULL_CHUNKS + [TAIL_BYTES]


def _fixture_wav() -> bytes:
    return WAV_PATH.read_bytes()


def _fixture_pcm() -> bytes:
    return pcm_from_wav(_fixture_wav())


# ---------------------------------------------------------------------------
# Stubs mirroring the aioesphomeapi surface the device uses
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.events = []
        self.audio = []
        self.media = []

    def send_voice_assistant_event(self, event_type, data):
        self.events.append((int(event_type), dict(data or {})))

    def send_voice_assistant_audio(self, payload):
        self.audio.append(payload)

    def media_player_command(self, key, **kwargs):
        self.media.append((key, kwargs))


class AnnounceClient(FakeClient):
    '''Adds the announcement RPC whose reply closes a continued conversation.'''

    def __init__(self):
        super().__init__()
        self.announcements = []

    async def send_voice_assistant_announcement_await_response(
        self, media_id, timeout, text='', preannounce_media_id='', start_conversation=False
    ):
        self.announcements.append(
            {'media_id': media_id, 'text': text, 'start_conversation': start_conversation}
        )
        return SimpleNamespace(success=True)


class FakeListener:
    def __init__(self):
        self._audio_q = queue.Queue(maxsize=64)


class Info:
    '''Stand-in for `MediaPlayerInfo` (same three positional fields).'''

    def __init__(self, id_, key, name, **extra):
        self.id = id_
        self.key = key
        self.name = name
        for k, v in extra.items():
            setattr(self, k, v)


def _config():
    return pe_config.from_settings(
        SimpleNamespace(
            voice_pe_enabled=True,
            voice_pe_prefer_api_audio=True,
            voice_pe_audio_queue_ms=300,
            voice_pe_led_rgb=[0.55, 0.0, 1.0],
        )
    )


def _caps(*, speaker: bool):
    flags = (
        FEATURE_VOICE_ASSISTANT
        | FEATURE_API_AUDIO
        | FEATURE_TIMERS
        | FEATURE_ANNOUNCE
        | FEATURE_START_CONVERSATION
    )
    if speaker:
        flags |= FEATURE_SPEAKER
    return build_snapshot(
        SimpleNamespace(voice_assistant_feature_flags=flags),
        [Info('external_media_player', 11, 'Media Player')],
    )


def _device(client, capabilities):
    device = VoicePEDevice(
        _config(),
        listener=FakeListener(),
        tts_engine=None,
        host='192.168.1.50',
        port=6053,
    )
    device._client = client
    device.entities = EntityIndex()
    device.capabilities = capabilities
    device.media = VoicePEMediaController(client, 11, 12)
    device._ingress = AudioIngress(device._listener, device.config, device.metrics)
    device.loop = None
    return device


def _run_loop(coro, device=None):
    loop = asyncio.new_event_loop()
    if device is not None:
        device.loop = loop
    try:
        return loop.run_until_complete(coro() if callable(coro) else coro)
    finally:
        pending = [t for t in asyncio.all_tasks(loop)]
        for task in pending:
            task.cancel()
        if pending:
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
        loop.close()


@pytest.mark.unit
class TestRealFixtureGeometry:
    '''The stored real WAV is a 16 kHz mono PCM16 container of 3.18 s.'''

    def test_stored_file_is_the_runtime_container(self):
        assert len(_fixture_wav()) == FILE_BYTES
        assert _fixture_wav() == wav_from_pcm(_fixture_pcm())

    def test_header_pins_mono_pcm16_at_the_wire_rate(self):
        import struct

        wav = _fixture_wav()
        riff, total, wave = struct.unpack('<4sI4s', wav[:12])
        fmt_size = struct.unpack('<I', wav[16:20])[0]
        audio_format, channels, rate, byte_rate, block, bits = struct.unpack(
            '<HHIIHH', wav[20:36]
        )
        data_id, data_size = struct.unpack('<4sI', wav[36:44])
        assert (riff, wave) == (b'RIFF', b'WAVE')
        assert fmt_size == 16 and audio_format == 1
        assert (channels, rate, bits) == (1, SAMPLE_RATE, 16)
        assert byte_rate == 32000 and block == 2
        assert (data_id, data_size) == (b'data', DATA_BYTES)
        assert total == 4 + 24 + 8 + DATA_BYTES
        assert len(wav) == total + 8

    def test_pcm_is_one_dimensional_and_fills_the_data_chunk(self):
        pcm = _fixture_pcm()
        assert len(pcm) == DATA_BYTES
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples.ndim == 1 and samples.size == TOTAL_SAMPLES

    def test_real_speech_is_not_silence(self):
        samples = np.frombuffer(_fixture_pcm(), dtype=np.int16)
        # A real utterance has both polarity and a non-trivial peak.
        assert int(samples.max()) > 0 and int(samples.min()) < 0
        assert float(np.abs(samples.astype(np.int32)).mean()) > 100.0

    def test_payload_grid_covers_the_whole_utterance(self):
        assert CHUNK_GRID[0] == SAMPLES_PER_CHUNK * SAMPLE_WIDTH
        assert sum(CHUNK_GRID) == DATA_BYTES
        assert len(CHUNK_GRID) == FULL_CHUNKS + 1
        assert 0 < TAIL_BYTES < PAYLOAD_BYTES_PER_CHUNK
        assert abs(DURATION_S - 3.181125) < 1e-6


@pytest.mark.unit
class TestRealWavOverTheWire:
    '''The transport sees 512-sample payloads, never the 44-byte container.'''

    def test_paced_sender_uses_512_sample_chunks(self):
        client = FakeClient()

        async def _run():
            started = time.monotonic()
            sent = await stream_pcm_paced(client, _fixture_pcm())
            return sent, time.monotonic() - started

        sent, elapsed = _run_loop(_run)
        assert sent == len(CHUNK_GRID)
        assert [len(p) for p in client.audio] == CHUNK_GRID
        assert len(client.audio[0]) == PAYLOAD_BYTES_PER_CHUNK
        assert len(client.audio[-1]) == TAIL_BYTES
        # Pacing keeps the device ring near 384 ms of a 512 ms ring, so the
        # wall clock lands between the audio length and one ring ahead.
        assert DURATION_S - 0.4 <= elapsed <= DURATION_S + 0.3

    def test_payload_bytes_rebuild_the_original_pcm(self):
        client = FakeClient()

        async def _run():
            return await stream_pcm_paced(client, _fixture_pcm())

        _run_loop(_run)
        assert b''.join(client.audio) == _fixture_pcm()


@pytest.mark.unit
class TestRealEgressOverTheDevice:
    '''The same real buffer through `VoicePEDevice`, per feature-flag set.'''

    def test_speaker_device_streams_the_real_pcm(self, monkeypatch):
        pcm = _fixture_pcm()
        client = FakeClient()
        device = _device(client, _caps(speaker=True))
        monkeypatch.setattr(
            'jarvis.integrations.voice_pe.tts_stream.synthesize_pcm',
            lambda engine, text: pcm,
        )

        async def _run():
            await device.handle_pipeline_start('', 0, SimpleNamespace(), None)
            await device._on_reply_async('Dobrý den, Toustovač je připraven.')
            for _ in range(240):
                await asyncio.sleep(0.01)
                if client.audio and len(client.audio) == len(CHUNK_GRID):
                    break
            if device._tts_task is not None:
                await device._tts_task
            return device.session_state

        state = _run_loop(_run, device)
        assert [len(p) for p in client.audio] == CHUNK_GRID
        sent = [event for event, _ in client.events]
        assert 98 in sent and 99 in sent and 8 not in sent
        assert sent[-1] == 2                      # RUN_END closes the run
        assert state is SessionState.IDLE

    def test_flags_61_publishes_the_real_wav_and_it_is_fetchable(self, monkeypatch):
        pcm = _fixture_pcm()
        wav = _fixture_wav()
        client = FakeClient()
        device = _device(client, _caps(speaker=False))
        monkeypatch.setattr(
            'jarvis.integrations.voice_pe.tts_stream.synthesize_pcm',
            lambda engine, text: pcm,
        )

        async def _run():
            await device.handle_pipeline_start('', 0, SimpleNamespace(), None)
            await device._on_reply_async('Dobrý den, Toustovač je připraven.')
            for _ in range(20):
                await asyncio.sleep(0.02)
            if device._tts_task is not None:
                await device._tts_task
            ordered = [event for event, _ in client.events]
            url = dict(client.events[ordered.index(8)][1])['url']
            hosts, rest = url.split('//')[1].split(':', 1)
            port, path = rest.split('/', 1)
            reader, writer = await asyncio.open_connection(hosts, int(port))
            writer.write(f'GET /{path} HTTP/1.1\r\nHost: x\r\n\r\n'.encode())
            await writer.drain()
            body = await reader.read(len(wav) + 256)
            writer.close()
            trail = device.media_delivery()
            if device._http is not None:
                await device._http.stop()
            return ordered, body, trail

        ordered, body, trail = _run_loop(_run, device)
        assert 98 not in ordered
        assert ordered[-2:] == [8, 2]                # TTS_END then RUN_END
        served = body.split(b'\r\n\r\n', 1)[1]
        assert len(served) == FILE_BYTES
        assert pcm_from_wav(served) == pcm
        assert client.audio == []
        assert trail['stored_bytes'] == FILE_BYTES
        assert trail['wav_bytes_metric'] == FILE_BYTES
        assert trail['hits'] == 1 and trail['status'] == 200
        assert trail['served_bytes'] == FILE_BYTES
        assert trail['deliveries'] == 1

    def test_pending_playback_is_closed_by_the_finished_report(self, monkeypatch):
        pcm = _fixture_pcm()
        client = FakeClient()
        device = _device(client, _caps(speaker=False))
        monkeypatch.setattr(
            'jarvis.integrations.voice_pe.tts_stream.synthesize_pcm',
            lambda engine, text: pcm,
        )

        async def _run():
            await device.handle_pipeline_start('', 0, SimpleNamespace(), None)
            await device._on_reply_async('Dobrý den, Toustovač je připraven.')
            for _ in range(20):
                await asyncio.sleep(0.02)
            if device._tts_task is not None:
                await device._tts_task
            pending = len(device._pending_playback)
            key = device._tts_media_id
            await device.handle_announcement_finished(SimpleNamespace(success=True))
            finished = device.last_finished_generation
            closed = device.metrics.get('announcements_finished')
            if device._http is not None:
                await device._http.stop()
            return pending, key, finished, closed

        pending, key, finished, closed = _run_loop(_run, device)
        assert pending == 1
        assert key.endswith(f'-{finished}')
        assert closed == 1
        assert len(device._pending_playback) == 0

    def test_late_reply_announces_the_same_real_wav_url(self, monkeypatch):
        pcm = _fixture_pcm()
        client = AnnounceClient()
        device = _device(client, _caps(speaker=False))
        monkeypatch.setattr(
            'jarvis.integrations.voice_pe.tts_stream.synthesize_pcm',
            lambda engine, text: pcm,
        )

        async def _run():
            await device.handle_pipeline_start('', 0, SimpleNamespace(), None)
            generation = device.session_generation
            await device._end_run('Dobrý den, Toustovač je připraven.', generation, stream=False)
            device.announce_reply('Dobrý den, Toustovač je připraven.', None,
                                  start_conversation=False)
            for _ in range(20):
                await asyncio.sleep(0.02)
            if device._http is not None:
                await device._http.stop()
            return list(client.announcements), dict(device.metrics)

        announcements, metrics = _run_loop(_run, device)
        assert len(announcements) == 1
        url = announcements[0]['media_id']
        assert url.startswith('http://')
        key = url.rsplit('/', 1)[1]
        assert metrics['late_reply_announcements'] == 1
        assert metrics['late_reply_announcement_success'] is True
        assert announcements[0]['start_conversation'] is False
        assert metrics.get('wav_bytes_' + key) == FILE_BYTES


@pytest.mark.unit
class TestRegenerationFromModel:
    '''The on-disk Piper model reproduces the container shape of the fixture.'''

    @staticmethod
    def _engine():
        from jarvis.output.tts import PiperTTS

        model = Path.home() / '.local/share/jarvis/models/piper/cs_CZ-jirka-medium.onnx'
        if not model.exists():
            return None
        engine = PiperTTS(enabled=True, model_path=str(model))
        if not engine._ensure_initialized():
            return None
        return engine

    def test_model_loads_when_installed(self):
        engine = self._engine()
        if engine is None:
            pytest.skip('piper model not installed')
        assert getattr(engine, '_sample_rate', 0) > 0

    def test_regenerated_pcm_matches_the_container_geometry(self):
        engine = self._engine()
        if engine is None:
            pytest.skip('piper model not installed')
        pcm = synthesize_pcm(engine, 'Dobrý den, Toustovač je připraven.')
        assert pcm is not None
        assert len(pcm) % SAMPLE_WIDTH == 0
        # Same wire rate and width; length may drift because Piper noise is
        # unseeded, so only a bounded ratio is asserted.
        assert abs(len(pcm) - DATA_BYTES) / DATA_BYTES < 0.20
        assert np.frombuffer(pcm, dtype=np.int16).ndim == 1

    def test_regenerated_pcm_reproduces_the_payload_grid(self):
        engine = self._engine()
        if engine is None:
            pytest.skip('piper model not installed')
        pcm = synthesize_pcm(engine, 'Dobrý den, Toustovač je připraven.')
        sizes = [
            len(chunk)
            for chunk in (
                pcm[i:i + PAYLOAD_BYTES_PER_CHUNK]
                for i in range(0, len(pcm), PAYLOAD_BYTES_PER_CHUNK)
            )
        ]
        assert sizes[:-1] == [PAYLOAD_BYTES_PER_CHUNK] * (len(sizes) - 1)
        assert 0 < sizes[-1] <= PAYLOAD_BYTES_PER_CHUNK
        assert sum(sizes) == len(pcm)
