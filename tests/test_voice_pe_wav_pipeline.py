'''Pipeline tests for the synthetic Piper TTS WAV fixture (Voice PE egress).

`tests/piper_tts_synthetic.wav` is the byte-exact container the shared Piper
engine produces at runtime: mono PCM16 at `SAMPLE_RATE` wrapped by the very
`wav_from_pcm` the transport uses. These tests push that buffer through the
same production code paths a live satellite sees:

* paced native-API PCM (`SPEAKER | API_AUDIO`),
* LAN-WAV `TTS_END` egress (`API_AUDIO` without `SPEAKER`, flags `61`),
* the announcement RPC carrying the same WAV for a late reply.

No network device, no real model: `synthesize_pcm` is patched with the fixture
bytes, so the numbers below are exactly what the wire would carry.
'''

import asyncio
import queue
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
    FEATURE_API_AUDIO,
    FEATURE_ANNOUNCE,
    FEATURE_SPEAKER,
    FEATURE_START_CONVERSATION,
    FEATURE_TIMERS,
    FEATURE_VOICE_ASSISTANT,
    PAYLOAD_BYTES_PER_CHUNK,
    SAMPLES_PER_CHUNK,
    SAMPLE_WIDTH,
    SessionState,
)
from jarvis.integrations.voice_pe.tts_stream import (
    pcm_from_wav,
    stream_pcm_paced,
    stream_wav,
    wav_from_pcm,
)
from jarvis.integrations.voice_pe.voice_transport import AudioIngress

WAV_PATH = Path(__file__).resolve().parent / 'piper_tts_synthetic.wav'

#: Token groups of the synthetic reply: `(token positions, per-position int16
#: streams)`. Each position holds the pointwise sum of its streams, 1-D, which
#: is the shape `InferenceSession.run` yields after `sum(-1)`.
GROUPS = ((1024, 3), (512, 7), (152, 3))
TOTAL_SAMPLES = sum(n for n, _ in GROUPS)              # 1688
DATA_BYTES = TOTAL_SAMPLES * SAMPLE_WIDTH              # 3376
CHUNK_GRID = [
    PAYLOAD_BYTES_PER_CHUNK,
    PAYLOAD_BYTES_PER_CHUNK,
    PAYLOAD_BYTES_PER_CHUNK,
    DATA_BYTES - 3 * PAYLOAD_BYTES_PER_CHUNK,
]                                                      # [1024, 1024, 1024, 304]


def _synth_pcm() -> bytes:
    '''Regenerate the fixture PCM (identical to the stored file, by design).'''
    frames = []
    for positions, streams in GROUPS:
        t = np.arange(positions, dtype=np.float64)
        acc = np.zeros(positions, dtype=np.float64)
        for j in range(streams):
            acc += (32767.0 / (j + 1)) * np.sin(2.0 * np.pi * (j + 1) * t / 64.0)
        frames.append(np.clip(acc, -32768, 32767).astype(np.int16))
    return b''.join(frame.tobytes() for frame in frames)


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
        # Cancel and drain to a terminal state: the per-device
        # ``AudioIngress.pump`` must not be destroyed while pending.
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
class TestSyntheticFixture:
    '''The stored WAV is what `wav_from_pcm` writes over the synthetic PCM.'''

    def test_stored_file_is_the_runtime_container(self):
        assert _fixture_wav() == wav_from_pcm(_synth_pcm())

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
        assert (channels, rate, bits) == (1, 16000, 16)
        assert byte_rate == 32000 and block == 2
        assert (data_id, data_size) == (b'data', DATA_BYTES)
        assert total == 4 + 24 + 8 + DATA_BYTES

    def test_pcm_round_trips_and_keeps_the_sample_grid(self):
        pcm = _fixture_pcm()
        assert len(pcm) == DATA_BYTES
        assert pcm == _synth_pcm()
        # 1-D per token after `sum(-1)`: no `(None, A, B)` shaped rows.
        assert np.frombuffer(pcm, dtype=np.int16).ndim == 1

    def test_pcm_is_a_sum_of_streams_per_position(self):
        # First group: 1024 positions, 3 summed int16 streams => deterministic.
        pcm = np.frombuffer(_fixture_pcm(), dtype=np.int16)
        t = np.arange(1024, dtype=np.float64)
        expected = np.zeros(1024)
        for j in range(3):
            expected += (32767.0 / (j + 1)) * np.sin(2.0 * np.pi * (j + 1) * t / 64.0)
        assert np.array_equal(
            pcm[:1024], np.clip(expected, -32768, 32767).astype(np.int16)
        )


@pytest.mark.unit
class TestWavOverTheWire:
    '''The transport sees 512-sample payloads, never the 44-byte container.'''

    def test_stream_wav_yields_the_payload_grid(self):
        payload = _fixture_wav()

        async def _iter():
            for i in range(0, len(payload), 7):  # awkward input slices
                yield payload[i:i + 7]

        async def _run():
            return [len(chunk) for chunk in [c async for c in stream_wav(_iter())]]

        sizes = _run_loop(_run)
        assert sizes == CHUNK_GRID
        assert sum(sizes) == DATA_BYTES

    def test_paced_sender_uses_512_sample_chunks(self):
        client = FakeClient()

        async def _run():
            return await stream_pcm_paced(client, _fixture_pcm())

        sent = _run_loop(_run)
        assert sent == len(CHUNK_GRID)
        assert [len(p) for p in client.audio] == CHUNK_GRID
        assert len(client.audio[0]) == SAMPLES_PER_CHUNK * SAMPLE_WIDTH


@pytest.mark.unit
class TestEgressOverTheDevice:
    '''The same buffer through `VoicePEDevice`, per feature-flag set.'''

    def test_speaker_device_streams_the_fixture_pcm(self, monkeypatch):
        pcm = _fixture_pcm()
        client = FakeClient()
        device = _device(client, _caps(speaker=True))
        monkeypatch.setattr(
            'jarvis.integrations.voice_pe.tts_stream.synthesize_pcm',
            lambda engine, text: pcm,
        )

        async def _run():
            await device.handle_pipeline_start('', 0, SimpleNamespace(), None)
            await device._on_reply_async('Toaster ready.')
            for _ in range(8):
                await asyncio.sleep(0.02)
            if device._tts_task is not None:
                await device._tts_task
            return device.session_state

        state = _run_loop(_run, device)
        assert [len(p) for p in client.audio] == CHUNK_GRID
        sent = [event for event, _ in client.events]
        assert 98 in sent and 99 in sent and 8 not in sent
        assert sent[-1] == 2                      # RUN_END closes the run
        assert state is SessionState.IDLE

    def test_flags_61_publishes_the_fixture_wav_and_it_is_fetchable(self, monkeypatch):
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
            await device._on_reply_async('Toaster ready.')
            for _ in range(6):
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
        assert pcm_from_wav(body.split(b'\r\n\r\n', 1)[1]) == pcm
        assert client.audio == []
        assert trail['stored_bytes'] == len(wav)
        assert trail['wav_bytes_metric'] == len(wav)
        assert trail['hits'] == 1 and trail['status'] == 200
        assert trail['served_bytes'] == len(wav)
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
            await device._on_reply_async('Toaster ready.')
            for _ in range(6):
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

    def test_late_reply_announces_the_same_wav_url(self, monkeypatch):
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
            await device._end_run('Toaster ready.', generation, stream=False)
            device.announce_reply('Toaster ready.', None, start_conversation=False)
            for _ in range(8):
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
        assert metrics.get('wav_bytes_' + key) == len(_fixture_wav())
