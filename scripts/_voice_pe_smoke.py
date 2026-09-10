"""Hardware smoke test for the Voice PE integration (stock retail firmware).

Run with the satellite powered and on the same LAN, Ollama reachable, and the
centre button within reach:

    python scripts/_voice_pe_smoke.py [host]

The script boots the real stack - ``Database``, ``Settings``, the Piper TTS
engine, ``VoiceListener`` (WebRTC VAD + Whisper) in its own thread and the
``VoicePEManager`` with one ``VoicePEDevice`` - then walks one conversation:

    button -> API audio -> VAD -> Whisper -> Jarvis reply
           -> device GET of that WAV key -> audible playback -> AnnounceFinished

Every checkpoint compares a baseline read just before the step with the value
after it, so an older counter can never satisfy a later check. The device's own
TTS HTTP server stays open for the whole run and is counted per payload key.

Each line is ``ok`` or ``FAIL``; the exit code is the number of failed
checkpoints, so CI can call it directly. No Home Assistant server, no mocks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The console on Windows is cp1252; the checkpoints print emoji from the stack.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

RESULTS: list[tuple[int, str, bool, str]] = []


def _report(index: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((index, name, bool(ok), detail))
    print(
        f"{'ok  ' if ok else 'FAIL'} {index}. {name}"
        + (f" - {detail}" if detail else ""),
        flush=True,
    )


def _poll(predicate, timeout_s: float, step_s: float = 0.05, recorder=None):
    """Poll ``predicate`` until it returns a truthy value or time is up.

    ``recorder`` is called with every snapshot, which is how the phase series of
    the ring and the avatar are collected.
    """
    deadline = time.monotonic() + timeout_s
    value = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
        except Exception:
            value = None
        if recorder is not None:
            try:
                recorder()
            except Exception:
                pass
        if value:
            return value
        time.sleep(step_s)
    return value


def _avatar_state() -> str:
    """Toaster avatar phase from the shared state file, ``?`` when absent."""
    try:
        from desktop_app.face_widget import get_jarvis_state

        return str(get_jarvis_state().state.value)
    except Exception:
        return "?"


def _metrics(device) -> dict:
    return device.metrics


def _counter(device, key: str) -> int:
    return int(_metrics(device).get(key, 0) or 0)


def _submit(manager, coro):
    """Run one coroutine on the manager loop and return its future."""
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, manager._loop)


def _await(manager, coro, timeout_s: float = 32.0):
    return _submit(manager, coro).result(timeout=timeout_s)


def _settings_with_host(settings, host: str):
    """Same settings with one manual host and the feature on."""
    values = {
        n: getattr(settings, n, None) for n in dir(settings) if not n.startswith("__")
    }
    values["voice_pe_enabled"] = True
    values["voice_pe_host"] = host
    return SimpleNamespace(**values)


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else None

    from jarvis.config import load_settings
    from jarvis.integrations import voice_pe
    from jarvis.listening.listener import VoiceListener
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.output.tts import create_tts_engine

    settings = load_settings()
    if host:
        settings = _settings_with_host(settings, host)

    # 1. Real bootstrap: database, TTS engine, listener thread, manager.
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
        _report(1, "manager boot reached READY", False, "manager disabled")
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
        "manager boot reached READY (handshake + capability sync)",
        bool(health),
        f"devices={len(manager.devices)} "
        f"state={health.get('device_state') if health else '-'}",
    )
    if not health:
        return 1
    device = manager.devices[0]

    # 2/3. Identity, decoded flags, enumeration.
    _report(
        2,
        "identity and flags decoded",
        bool(device.identity.get("node_name")) and device.capabilities.voice_assistant,
        f"{device.identity.get('project_name')} v{device.identity.get('project_version')} "
        f"[{','.join(device.capabilities.names())}] "
        f"pcm_egress={int(device.capabilities.uses_api_audio)}",
    )
    _report(
        3,
        "entity enumeration non-empty",
        device.capabilities.entity_count > 0,
        f"{device.capabilities.entity_count} entities",
    )

    # 4. Wake-word mode as the device reports it, not as the write requested.
    _report(
        4,
        "wake words off per device read-back",
        bool(device.wake_words_disabled)
        and int(_metrics(device).get("wake_words_active", -1)) == 0,
        f"disabled={device.wake_words_disabled} "
        f"active={_metrics(device).get('wake_words_active')} err={device.last_error or '-'}",
    )

    phases: list = []
    avatars: list = []

    def _record() -> None:
        led = device.ui_view()["connection"]["led_phase_id"]
        avatar = _avatar_state()
        if not phases or phases[-1] != led:
            phases.append(led)
        if not avatars or avatars[-1] != avatar:
            avatars.append(avatar)

    # 5. One run, opened by the centre button or by ``start_conversation``.
    gen_before = int(device.session_generation)
    print("   -> press the centre button now (9 s window)", flush=True)
    started = _poll(
        lambda: int(device.session_generation) > gen_before, 9.0, recorder=_record
    )
    if not started and device._client is not None and manager._loop is not None:
        try:
            _await(
                manager,
                device._client.send_voice_assistant_announcement_await_response(
                    _await(manager, device.tts_media_url("one two")),
                    30.0,
                    text="one two",
                    start_conversation=True,
                ),
            )
        except Exception as err:
            print(f"   note: announce/start_conversation: {err}", flush=True)
        started = _poll(
            lambda: int(device.session_generation) > gen_before, 20.0, recorder=_record
        )
    _report(
        5,
        "one pipeline run opened, generation moved by one",
        bool(started) and int(device.session_generation) == gen_before + 1,
        f"generation={device.session_generation} (was {gen_before})",
    )
    if not started:
        voice_pe.stop()
        listener.stop()
        return 1

    sink = getattr(listener, "_voice_pe_sink", None)
    lease = sink.lease() if sink is not None else None
    _report(
        6,
        "lease names the satellite as the audio owner",
        lease is not None
        and lease.source_id == "voice_pe"
        and lease.device_id == device.device_id
        and int(lease.session_generation) == int(device.session_generation),
        f"{None if lease is None else (lease.source_id, lease.device_id, lease.session_generation)}",
    )

    # 7/8. API audio ingress through exactly one persistent pump.
    chunks_before = _counter(device, "audio_chunks")
    _poll(
        lambda: _counter(device, "audio_chunks") > chunks_before,
        6.0,
        recorder=_record,
    )
    _report(
        7,
        "API audio reached the listener queue",
        _counter(device, "audio_chunks") > chunks_before,
        f"chunks {chunks_before} -> {_counter(device, 'audio_chunks')}",
    )
    _report(
        8,
        "one microphone pump for this connection generation",
        int(_metrics(device).get("pump_tasks", 0)) == 1,
        f"pump_tasks={_metrics(device).get('pump_tasks')}",
    )

    # 9. VAD + Whisper produced a transcript accepted without a wake word.
    stt_before = _counter(device, "stt_end")
    print("   -> speak one short sentence into the satellite (30 s window)", flush=True)
    _poll(lambda: _counter(device, "stt_end") > stt_before, 30.0, recorder=_record)
    _report(
        9,
        "VAD + Whisper transcript accepted for the open run",
        _counter(device, "stt_end") > stt_before,
        f"stt_end {stt_before} -> {_counter(device, 'stt_end')}",
    )

    # 10. Reply reached the device as a LAN WAV URL on the still-open server.
    delivered_before = _counter(device, "tts_url_deliveries")
    _poll(
        lambda: _counter(device, "tts_url_deliveries") > delivered_before,
        180.0,
        recorder=_record,
    )
    server = device._http
    key = device._tts_media_id
    _report(
        10,
        "TTS_END carried a LAN URL on a still-open server",
        _counter(device, "tts_url_deliveries") > delivered_before
        and server is not None
        and server.port > 0
        and bool(key),
        f"deliveries={_counter(device, 'tts_url_deliveries')} "
        f"port={getattr(server, 'port', 0)} key='{key}'",
    )

    # 11. The satellite fetched that same key: counted per key, misses separate.
    hits_before = server.hit_count(key) if server is not None else 0
    missing_before = server.missing if server is not None else 0
    _poll(
        lambda: server is not None and server.hit_count(key) > hits_before,
        90.0,
        recorder=_record,
    )
    _report(
        11,
        "device fetched that exact WAV key",
        server is not None and server.hit_count(key) > hits_before,
        f"key '{key}' hits {hits_before} -> {server.hit_count(key) if server else 0}, "
        f"misses {missing_before} -> {getattr(server, 'missing', 0)}",
    )

    # 12. Audible playback: AnnounceFinished of this generation, success True.
    finished_before = _counter(device, "announcements_finished")
    _poll(
        lambda: _counter(device, "announcements_finished") > finished_before,
        45.0,
        recorder=_record,
    )
    _report(
        12,
        "playback closed with a successful AnnounceFinished of this run",
        _counter(device, "announcements_finished") > finished_before
        and device.last_announce_success is True
        and int(device.last_finished_generation) == int(device.session_generation),
        f"finished={_counter(device, 'announcements_finished')} "
        f"success={device.last_announce_success} generation="
        f"{device.last_finished_generation}/{device.session_generation}",
    )

    # 13. Phase progression of the ring and the avatar over the whole walk.
    _poll(lambda: phases[-1] == 1, 5.0, recorder=_record)
    _report(
        13,
        "ring and avatar followed thinking -> replying -> idle",
        4 in phases and 5 in phases and phases[-1] == 1
        and "thinking" in avatars
        and "speaking" in avatars,
        f"led={phases} avatars={avatars}",
    )

    # 14. A second run of the same generation keeps that single pump.
    gen_second = int(device.session_generation)
    print("   -> press the centre button once more (12 s window)", flush=True)
    second = _poll(
        lambda: int(device.session_generation) > gen_second, 12.0, recorder=_record
    )
    if not second and device._client is not None and manager._loop is not None:
        try:
            _await(
                manager,
                device._client.send_voice_assistant_announcement_await_response(
                    _await(manager, device.tts_media_url("one two")),
                    30.0,
                    text="one two",
                    start_conversation=True,
                ),
            )
        except Exception:
            pass
        second = _poll(
            lambda: int(device.session_generation) > gen_second, 8.0, recorder=_record
        )
    _report(
        14,
        "second run reuses the generation's pump",
        bool(second) and int(_metrics(device).get("pump_tasks", 0)) == 1,
        f"generation={device.session_generation} "
        f"pump_tasks={_metrics(device).get('pump_tasks')}",
    )

    # 15. Media commands round-trip through the API and the device's own push.
    accepted = True
    try:
        accepted = bool(
            _await(manager, manager.pause_media(""))
            and _await(manager, manager.resume_media(""))
            and _await(manager, manager.set_volume("", 0.66))
            and _await(manager, manager.set_muted("", False))
        )
    except Exception as err:
        accepted = False
        print(f"   note: media commands: {err}", flush=True)
    _poll(
        lambda: manager.media_state("").get("volume_source") in ("device", "echo"),
        8.0,
        recorder=_record,
    )
    snapshot = manager.media_state("")
    _report(
        15,
        "pause/resume/volume/mute accepted, state echoed by the device",
        accepted and snapshot.get("volume_source") in ("device", "echo"),
        f"accepted={accepted} state={snapshot.get('state')} "
        f"volume={snapshot.get('volume')} src={snapshot.get('volume_source')}",
    )

    # 16. Metrics close the conversation.
    _poll(lambda: _counter(device, "run_end") >= 1, 120.0, recorder=_record)
    _report(
        16,
        "metrics close the conversation",
        _counter(device, "sessions") >= 2
        and _counter(device, "stt_end") >= 1
        and _counter(device, "run_end") >= 1,
        f"sessions={_counter(device, 'sessions')} stt_end={_counter(device, 'stt_end')} "
        f"run_end={_counter(device, 'run_end')} errors={_metrics(device).get('errors')}",
    )

    voice_pe.stop()
    listener.stop()
    failed = sum(1 for _i, _n, ok, _d in RESULTS if not ok)
    print(f"-- {len(RESULTS) - failed}/{len(RESULTS)} checkpoints passed", flush=True)
    if failed:
        print(
            "   note: 9 to 16 need one spoken sentence while the run is open; "
            "the rest is protocol state.",
            flush=True,
        )
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
