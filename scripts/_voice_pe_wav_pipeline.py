'''Send the synthetic Piper TTS WAV to the Voice PE satellite, as production does.

Run with the satellite powered and on the same LAN, or against the in-process
stub stack (the default, no network device needed):

    python scripts/_voice_pe_wav_pipeline.py

The fixture `tests/piper_tts_synthetic.wav` is a runtime Piper container: mono
PCM16 at 16 kHz wrapped by the production `wav_from_pcm`. One buffer travels
the three egress paths `VoicePEDevice` picks from the device feature bits, in
the same order and with the same pacing as the live pipeline:

    1  container        `wav_from_pcm` / `pcm_from_wav` round trip
    2  payload grid     `stream_wav` -> 512-sample payloads of 1024 B
    3  paced PCM        `stream_pcm_paced` on the 384 ms ring model
    4  native egress    SPEAKER | API_AUDIO: 7 -> 98 -> payloads -> 99 -> 2
    5  LAN-WAV egress   flags 61 (no SPEAKER): 7 -> 8{url} -> satellite GET -> 2
    6  announcement     late reply: announce RPC with the same WAV URL
    7  completion       `AnnounceFinished` closes the pending playback

Each line is `ok` or `FAIL`; the exit code is the number of failed
checkpoints, so CI can call it directly.
'''

from __future__ import annotations

import asyncio
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))

try:  # Windows console is cp1252; the stack prints emoji.
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
except Exception:
    pass

from jarvis.integrations.voice_pe import config as pe_config
from jarvis.integrations.voice_pe import tts_stream as pe_tts
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

WAV_PATH = _ROOT / 'tests' / 'piper_tts_synthetic.wav'
GROUPS = ((1024, 3), (512, 7), (152, 3))
DATA_BYTES = sum(n for n, _ in GROUPS) * SAMPLE_WIDTH
RESULTS: list[tuple[int, str, bool, str]] = []


def _report(index: int, name: str, ok: bool, detail: str = '') -> None:
    RESULTS.append((index, name, bool(ok), detail))
    stamp = time.strftime('%H:%M:%S', time.localtime())
    print(
        f"{stamp} {'ok  ' if ok else 'FAIL'} {index}. {name}"
        + (f' - {detail}' if detail else ''),
        flush=True,
    )


def _synth_pcm() -> bytes:
    '''The synthetic PCM behind the stored fixture.'''
    frames = []
    for positions, streams in GROUPS:
        t = np.arange(positions, dtype=np.float64)
        acc = np.zeros(positions, dtype=np.float64)
        for j in range(streams):
            acc += (32767.0 / (j + 1)) * np.sin(2.0 * np.pi * (j + 1) * t / 64.0)
        frames.append(np.clip(acc, -32768, 32767).astype(np.int16))
    return b''.join(frame.tobytes() for frame in frames)


class FakeClient:
    '''Records what the transport sends; mirrors the aioesphomeapi surface.'''

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
    def __init__(self, id_, key, name, **extra):
        self.id, self.key, self.name = id_, key, name
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
        # Cancel and then *drain* to a terminal state, so the per-device
        # ``AudioIngress.pump`` task cannot be destroyed while pending.
        pending = [t for t in asyncio.all_tasks(loop)]
        for task in pending:
            task.cancel()
        if pending:
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
        loop.close()


def main() -> int:
    wav = WAV_PATH.read_bytes()
    pcm = pcm_from_wav(wav)
    # Same substitution ``tests/test_voice_pe_wav_pipeline.py`` does with
    # ``monkeypatch``: the engine behind the wire is this deterministic PCM.
    pe_tts.synthesize_pcm = lambda engine, text: pcm

    # 1. container round trip, exactly as the transport unwraps it.
    _report(
        1,
        'container round trip',
        wav == wav_from_pcm(pcm) and pcm == _synth_pcm(),
        f'wav={len(wav)} B data={len(pcm)} B samples={len(pcm) // SAMPLE_WIDTH}',
    )

    # 2. payload grid the satellite ring expects (512 samples = 1024 B).
    expected = [PAYLOAD_BYTES_PER_CHUNK] * (DATA_BYTES // PAYLOAD_BYTES_PER_CHUNK)
    if DATA_BYTES % PAYLOAD_BYTES_PER_CHUNK:
        expected.append(DATA_BYTES % PAYLOAD_BYTES_PER_CHUNK)

    async def _grid():
        async def _iter():
            for i in range(0, len(wav), 7):
                yield wav[i:i + 7]

        return [len(p) for p in [x async for x in stream_wav(_iter())]]

    grid = _run_loop(_grid)
    _report(2, '512-sample payload grid', grid == expected, f'{grid}')

    # 3. paced sender on the 384 ms ring model.
    client = FakeClient()

    async def _paced():
        return await stream_pcm_paced(client, pcm)

    chunks = _run_loop(_paced)
    _report(
        3,
        'paced PCM sender',
        chunks == len(grid) and [len(p) for p in client.audio] == grid,
        f'{chunks} chunks x {SAMPLES_PER_CHUNK} samples = '
        f'{chunks * SAMPLES_PER_CHUNK / 16000:.4f} s',
    )

    # 4. live run on a SPEAKER device: PCM egress with its event tail.
    speaker_client = FakeClient()
    speaker = _device(speaker_client, _caps(speaker=True))

    async def _native():
        await speaker.handle_pipeline_start('', 0, SimpleNamespace(), None)
        await speaker._on_reply_async('Toaster ready.')
        for _ in range(8):
            await asyncio.sleep(0.02)
        if speaker._tts_task is not None:
            await speaker._tts_task
        return [e for e, _ in speaker_client.events], speaker.session_state

    sent, state = _run_loop(_native, speaker)
    _report(
        4,
        'native API PCM egress',
        98 in sent
        and 99 in sent
        and 8 not in sent
        and sent[-1] == 2
        and state is SessionState.IDLE
        and len(speaker_client.audio) == len(grid),
        f'7=TTS_START 98=TTS_STREAM_START 99=TTS_STREAM_END 2=RUN_END, '
        f'state={state.value}, payloads={[len(p) for p in speaker_client.audio]}',
    )

    # 5. flags 61: LAN WAV published, fetched by the satellite, trail complete.
    lan_client = FakeClient()
    lan = _device(lan_client, _caps(speaker=False))

    async def _lan():
        await lan.handle_pipeline_start('', 0, SimpleNamespace(), None)
        await lan._on_reply_async('Toaster ready.')
        for _ in range(6):
            await asyncio.sleep(0.02)
        if lan._tts_task is not None:
            await lan._tts_task
        ordered = [e for e, _ in lan_client.events]
        url = dict(lan_client.events[ordered.index(8)][1])['url']
        lan_host, rest = url.split('//')[1].split(':', 1)
        port, path = rest.split('/', 1)
        reader, writer = await asyncio.open_connection(lan_host, int(port))
        writer.write(f'GET /{path} HTTP/1.1\r\nHost: x\r\n\r\n'.encode())
        await writer.drain()
        body = await reader.read(len(wav) + 256)
        writer.close()
        trail = lan.media_delivery()
        if lan._http is not None:
            await lan._http.stop()
        return ordered, url, trail, pcm_from_wav(body.split(b'\r\n\r\n', 1)[1])

    ordered, url, trail, fetched = _run_loop(_lan, lan)
    _report(
        5,
        'LAN WAV egress fetched',
        ordered[-2:] == [8, 2]
        and fetched == pcm
        and trail['stored_bytes'] == len(wav)
        and trail['hits'] == 1
        and trail['status'] == 200
        and lan_client.audio == [],
        f'8=TTS_END 2=RUN_END hits={trail['hits']} status={trail['status']} '
        f'{trail['served_bytes']} B, {url}',
    )

    # 6. late reply: the same WAV URL rides the announce RPC.
    late_client = AnnounceClient()
    late = _device(late_client, _caps(speaker=False))

    async def _late():
        await late.handle_pipeline_start('', 0, SimpleNamespace(), None)
        generation = late.session_generation
        await late._end_run('Toaster ready.', generation, stream=False)
        late.announce_reply('Toaster ready.', None, start_conversation=False)
        for _ in range(8):
            await asyncio.sleep(0.02)
        if late._http is not None:
            await late._http.stop()
        return list(late_client.announcements), dict(late.metrics)

    announcements, metrics = _run_loop(_late, late)
    late_url = announcements[0]['media_id'] if announcements else ''
    _report(
        6,
        'announcement fallback',
        len(announcements) == 1
        and late_url.startswith('http://')
        and metrics.get('late_reply_announcements') == 1
        and metrics.get('late_reply_announcement_success') is True,
        late_url,
    )

    # 7. device-side completion closes the pending playback FIFO.
    done_client = FakeClient()
    done = _device(done_client, _caps(speaker=False))

    async def _finished():
        await done.handle_pipeline_start('', 0, SimpleNamespace(), None)
        await done._on_reply_async('Toaster ready.')
        for _ in range(6):
            await asyncio.sleep(0.02)
        if done._tts_task is not None:
            await done._tts_task
        pending = len(done._pending_playback)
        await done.handle_announcement_finished(SimpleNamespace(success=True))
        closed = done.metrics.get('announcements_finished')
        generation = done.last_finished_generation
        if done._http is not None:
            await done._http.stop()
        return pending, closed, generation

    pending, closed, generation = _run_loop(_finished, done)
    _report(
        7,
        'AnnounceFinished closes playback',
        pending == 1 and closed == 1 and len(done._pending_playback) == 0,
        f'pending 1 -> 0, announcements_finished={closed}, generation={generation}',
    )

    failed = sum(1 for _, _, ok, _ in RESULTS if not ok)
    print(
        f"{'PIPELINE_WAV_OK' if failed == 0 else 'PIPELINE_WAV_FAILED'} "
        f'{len(RESULTS) - failed}/{len(RESULTS)} checkpoints, '
        f'{len(pcm) // SAMPLE_WIDTH} samples in {len(wav)} B WAV',
        flush=True,
    )
    return failed


if __name__ == '__main__':
    sys.exit(main())
