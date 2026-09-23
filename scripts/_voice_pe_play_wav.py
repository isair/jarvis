"""Push one WAV straight to the Voice PE satellite and prove it was played.

    python scripts/_voice_pe_play_wav.py [host]

Boots the real stack (Database, Settings, the TTS engine, the VoiceListener
thread and VoicePEManager with one VoicePEDevice) and hands the satellite a WAV
through the egress its own feature bits select. The Piper engine is bypassed:
the bytes come from the stored container tests/piper_tts_real.wav (or
tests/piper_tts_synthetic.wav when the real one is absent), so the run is
reproducible offline.

Checkpoints, each ok or FAIL, exit code = number of failures:

    1  manager reached READY (handshake + capability sync)
    2  egress chosen from the decoded feature bits
    3  one run opened, RUN_START/STT_START in the per-generation ledger
    4  WAV published and TTS_END carries its URL
    5  the satellite fetched that URL itself (hits, status 200, served bytes)
    6  AnnounceFinished closed the playback of that generation

The fetch in step 5 is the audible moment: the satellite pulls the WAV over the
LAN and plays it on its own driver, so a non-zero hits with the full byte count
means the sentence was heard on the device.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

try:  # Windows console is cp1252; the stack prints emoji.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

RESULTS: list[tuple[int, str, bool, str]] = []


def _report(index: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((index, name, bool(ok), detail))
    stamp = time.strftime("%H:%M:%S", time.localtime())
    print(
        f"{stamp} {'ok  ' if ok else 'FAIL'} {index}. {name}"
        + (f" - {detail}" if detail else ""),
        flush=True,
    )


def _poll(predicate, timeout_s: float, step_s: float = 0.05):
    deadline = time.monotonic() + timeout_s
    value = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
        except Exception:
            value = None
        if value:
            return value
        time.sleep(step_s)
    return value


def _submit(manager, coro):
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, manager._loop)


def _await(manager, coro, timeout_s: float = 32.0):
    return _submit(manager, coro).result(timeout=timeout_s)


def _settings_with_host(settings, host: str):
    values = {
        n: getattr(settings, n, None) for n in dir(settings) if not n.startswith("__")
    }
    values["voice_pe_enabled"] = True
    values["voice_pe_host"] = host
    return SimpleNamespace(**values)


def _ledger(device, generation: int) -> list:
    return [name for gen, name in device.event_ledger if int(gen) == int(generation)]


def _fixture_bytes() -> bytes:
    for name in ("piper_tts_real.wav", "piper_tts_synthetic.wav"):
        path = _ROOT / "tests" / name
        if path.exists():
            return path.read_bytes()
    raise SystemExit("no WAV fixture under tests/")


def main() -> int:
    host = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    from jarvis.config import load_settings
    from jarvis.integrations import voice_pe
    from jarvis.integrations.voice_pe import tts_stream as pe_tts
    from jarvis.listening.listener import VoiceListener
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.output.tts import create_tts_engine

    wav = _fixture_bytes()
    pcm = pe_tts.pcm_from_wav(wav)
    settings = load_settings()
    if host:
        settings = _settings_with_host(settings, host)

    db = Database(settings.db_path, settings.sqlite_vss_path)
    tts = create_tts_engine(
        engine=settings.tts_engine,
        enabled=settings.tts_enabled,
        voice=settings.tts_voice,
        rate=settings.tts_rate,
    )
    memory = DialogueMemory(
        inactivity_timeout=settings.dialogue_memory_timeout, max_interactions=20
    )
    listener = VoiceListener(db, settings, tts, memory)
    listener.start()
    manager = voice_pe.start(settings, listener, tts)
    if manager is None:
        _report(1, "manager reached READY", False, "manager disabled")
        return 1
    health = _poll(
        lambda: next(
            (
                d
                for d in manager.health()["devices"]
                if d.get("device_state") in ("ready", "voice_active")
            ),
            None,
        ),
        12.0,
    )
    _report(
        1,
        "manager reached READY (handshake + capability sync)",
        bool(health),
        f"devices={len(manager.devices)} "
        f"state={health.get('device_state') if health else '-'}",
    )
    if not health:
        voice_pe.stop()
        listener.stop()
        return 1
    device = manager.devices[0]

    # 2. Which egress the decoded feature bits pick for this satellite, read
    #    through the same helper the runtime reports: ``API_AUDIO`` with a
    #    ``SPEAKER`` streams raw PCM, anything else publishes a LAN WAV.
    egress = int(device.egress_mode())
    _report(
        2,
        "egress chosen from the feature bits",
        egress in (0, 1) and egress == int(device.ui_view()["connection"]["pcm_egress"]),
        f"pcm_egress={egress} "
        f"({'Native API PCM' if egress else 'LAN WAV'}), "
        f"speaker={int(device.capabilities.speaker)} "
        f"api_audio={int('api_audio' in device.capabilities.names())}",
    )

    # Piper is not needed: the wire bytes are the stored container's PCM.
    pe_tts.synthesize_pcm = lambda engine, text: pcm

    # 3. One run on this connection generation. The ledger stamps each event
    #    with ``session_generation``, so that is the number to read back.
    _await(manager, device.handle_pipeline_start("", 0, SimpleNamespace(), None))
    generation = int(device.session_generation)
    _report(
        3,
        "one run opened with its own generation",
        generation > 0 and "RUN_START" in _ledger(device, generation),
        f"generation={generation} ledger={_ledger(device, generation)}",
    )

    # 4. Deliver the WAV: TTS_END carries the URL of the published container.
    sentence = "Dobrý den, Toustovač je připraven."
    device.on_reply(sentence, None)

    def _url():
        info = device.media_delivery()
        if int(info.get("stored_bytes", 0) or 0) and info.get("port"):
            return (
                f"http://{device._lan_ip or device._host}:{info['port']}/{info['key']}"
            )
        return ""

    url = str(_poll(_url, 20.0) or "")
    _report(
        4,
        "WAV published, TTS_END carries its URL",
        url.startswith("http://")
        and int(device.media_delivery().get("stored_bytes", 0) or 0) == len(wav)
        and "TTS_END" in _ledger(device, generation),
        f"{len(wav)} B at {url}, ledger={_ledger(device, generation)}",
    )

    # 5. The satellite's own GET is the audible moment.
    trail = _poll(
        lambda: (
            device.media_delivery()
            if int(device.media_delivery().get("hits", 0) or 0) >= 1
            else None
        ),
        20.0,
    ) or device.media_delivery()
    _report(
        5,
        "satellite fetched the WAV itself",
        int(trail.get("hits", 0) or 0) >= 1
        and trail.get("status") == 200
        and int(trail.get("served_bytes", 0) or 0) == len(wav),
        f"hits={trail.get('hits')} status={trail.get('status')} "
        f"served={trail.get('served_bytes')} B port={trail.get('port')}",
    )

    # 6. The device reports the end of that playback.
    closed = _poll(
        lambda: int(device.metrics.get("announcements_finished", 0) or 0) >= 1
        and int(device.last_finished_generation) == generation,
        20.0,
    )
    _report(
        6,
        "AnnounceFinished closed this generation",
        bool(closed),
        f"announcements_finished={device.metrics.get('announcements_finished')} "
        f"last_finished_generation={device.last_finished_generation} "
        f"(wanted {generation})",
    )

    failed = sum(1 for _, _, ok, _ in RESULTS if not ok)
    print(
        f"{'PLAY_WAV_OK' if failed == 0 else 'PLAY_WAV_FAILED'} "
        f"{len(RESULTS) - failed}/{len(RESULTS)} checkpoints, "
        f"{len(pcm) // 2} samples in {len(wav)} B WAV",
        flush=True,
    )
    if device._http is not None:
        _await(manager, device._http.stop())
    voice_pe.stop()
    listener.stop()
    return failed


if __name__ == "__main__":
    sys.exit(main())
