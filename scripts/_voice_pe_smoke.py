"""Hardware smoke test for the Voice PE integration (stock retail firmware).

Run with the satellite powered and on the same LAN, Ollama reachable, and the
centre button within reach:

    python scripts/_voice_pe_smoke.py [host]

The script boots the real stack - ``Database``, ``Settings``, the Piper TTS
engine, ``VoiceListener`` (WebRTC VAD + Whisper) in its own thread and the
``VoicePEManager`` with one ``VoicePEDevice`` - then walks the full path of one
push-to-talk conversation:

    button -> API audio -> VAD -> Whisper -> Jarvis reply
           -> device GET of the WAV -> audible playback -> AnnounceFinished

The TTS HTTP server of the device stays open for the whole run, so the device
fetches the very URL that went into ``TTS_END``. Each line is ``ok`` or
``FAIL``; the exit code is the number of failed checkpoints, so CI can call it
directly. No Home Assistant server and no mocks: plain Native API on TCP 6053.
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


def _poll(predicate, timeout_s: float, step_s: float = 0.05):
    """Poll ``predicate`` until it returns a truthy value or time is up."""
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


def _avatar_state() -> str:
    """Toaster avatar phase from the shared state file, ``?`` when absent."""
    try:
        from desktop_app.face_widget import get_jarvis_state

        return str(get_jarvis_state().state.value)
    except Exception:
        return "?"


def _submit(manager, coro):
    """Run one coroutine on the manager loop and return its future."""
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, manager._loop)


def _announce_url(manager, device) -> str:
    """Synthesized WAV URL on the device's own still-open HTTP server."""
    try:
        return _submit(manager, device.tts_media_url("one two")).result(timeout=20.0) or ""
    except Exception:
        return ""


def _settings_with_host(settings, host: str):
    """Same settings with one manual host and the feature on."""
    values = {n: getattr(settings, n, None) for n in dir(settings) if not n.startswith("__")}
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
        print("FAIL 1. manager disabled (set voice_pe_enabled)", flush=True)
        return 1
    # ``start()`` already waited for the device list, so this read is final.
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

    # 2/3/4. Identity, decoded flags, enumeration, wake-word mode.
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
    _report(
        4,
        "wake words disabled as configured",
        bool(device.wake_words_disabled),
        f"disabled={device.wake_words_disabled} err={device.last_error or '-'}",
    )

    # 5. Push-to-talk gate. The centre button is the natural trigger; the
    #    announcement RPC with ``start_conversation`` opens the same device-side
    #    run when nobody can reach the unit, so the walk always completes.
    print("   -> press the centre button now (9 s window)", flush=True)
    started = _poll(lambda: device.session is not None, 9.0)
    trigger = "button"
    if not started and device._client is not None and manager._loop is not None:
        trigger = "start_conversation"
        try:
            future = _submit(
                manager,
                device._client.send_voice_assistant_announcement_await_response(
                    _announce_url(manager, device), 30.0,
                    text="one two", start_conversation=True,
                ),
            )
            future.result(timeout=32.0)
        except Exception as err:
            print(f"   note: announce/start_conversation: {err}", flush=True)
        started = _poll(lambda: device.session is not None, 20.0)
    _report(
        5,
        "one pipeline run opened (button or start_conversation)",
        bool(started),
        f"trigger={trigger} session={device.session_generation} "
        f"led={device.led_phase} avatar={_avatar_state()}",
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
        and lease.device_id == device.device_id,
        f"{None if lease is None else (lease.source_id, lease.device_id, lease.session_generation)}",
    )

    # 7/8. API audio ingress through exactly one persistent pump.
    chunks = _poll(
        lambda: (
            device.metrics.get("audio_chunks")
            if int(device.metrics.get("audio_chunks", 0)) >= 5
            else None
        ),
        6.0,
    )
    _report(
        7,
        "API audio reached the listener queue",
        bool(chunks),
        f"chunks={device.metrics.get('audio_chunks')} "
        f"depth_ms={device.metrics.get('microphone_queue_depth_ms')}",
    )
    _report(
        8,
        "one microphone pump for this connection generation",
        int(device.metrics.get("pump_tasks", 0)) == 1,
        f"pump_tasks={device.metrics.get('pump_tasks')}",
    )

    # 9. VAD + Whisper produced a transcript accepted without a wake word. One
    #    spoken utterance inside this window is what closes the walk. The query
    #    is already consumed by the agent when the reply is fast, so the STT
    #    counter counts as accepted too.
    print("   -> speak one short sentence into the satellite (30 s window)", flush=True)
    _poll(
        lambda: int(device.metrics.get("stt_end", 0) or 0) >= 1
        or str(listener.state_manager.get_pending_query() or ""),
        30.0,
    )
    query = str(listener.state_manager.get_pending_query() or "")
    _report(
        9,
        "VAD + Whisper transcript accepted for the open run",
        bool(query),
        f"last_event={device.metrics.get('last_event')} query='{query[:40]}'",
    )

    # 10. Reply reached the device as a LAN WAV URL, server still open. The
    #     agent turn owns this window, so it is as long as a real answer.
    _poll(lambda: int(device.metrics.get("tts_url_deliveries", 0)) >= 1, 180.0)
    server = device._http
    _report(
        10,
        "TTS_END carried a LAN URL on a still-open server",
        int(device.metrics.get("tts_url_deliveries", 0)) >= 1
        and server is not None
        and server.port > 0,
        f"deliveries={device.metrics.get('tts_url_deliveries')} port={getattr(server, 'port', 0)}",
    )

    # 11. The satellite fetched that same URL (device-side GET counted).
    fetched = _poll(
        lambda: (
            getattr(server, "requests", 0)
            if getattr(server, "requests", 0) >= 1
            else None
        ),
        90.0,
    )
    _report(
        11,
        "device fetched the WAV over LAN HTTP",
        bool(fetched),
        f"GET count={getattr(server, 'requests', 0)} media_state={device.media.state}",
    )

    # 12. Audible playback: AnnounceFinished plus the avatar phase.
    finished = _poll(
        lambda: int(device.metrics.get("announcements_finished", 0)) >= 1, 20.0
    )
    _report(
        12,
        "playback finished with AnnounceFinished",
        bool(finished),
        f"finished={device.metrics.get('announcements_finished')} "
        f"last_event={device.metrics.get('last_event')} avatar={_avatar_state()}",
    )

    # 13. A second run of the same generation keeps that single pump.
    print("   -> press the centre button once more (12 s window)", flush=True)
    second = _poll(lambda: device.session_generation >= 2, 12.0)
    if not second and device._client is not None and manager._loop is not None:
        try:
            _submit(
                manager,
                device._client.send_voice_assistant_announcement_await_response(
                    _announce_url(manager, device), 30.0,
                    text="one two", start_conversation=True,
                ),
            ).result(timeout=32.0)
        except Exception:
            pass
        second = _poll(lambda: device.session_generation >= 2, 8.0)
    _report(
        13,
        "second run reuses the generation's pump",
        bool(second) and int(device.metrics.get("pump_tasks", 0)) == 1,
        f"generation={device.session_generation} "
        f"pump_tasks={device.metrics.get('pump_tasks')}",
    )

    # 14. Media state came from the device push, not from a guess.
    snapshot = manager.media_state("")
    _report(
        14,
        "media snapshot from the device push",
        bool(snapshot) and snapshot.get("volume_source") in ("device", "echo"),
        f"state={snapshot.get('state')} volume={snapshot.get('volume')} "
        f"src={snapshot.get('volume_source')}",
    )

    # 15. Metrics close the run: two runs, a transcript and a closed run.
    _poll(lambda: int(device.metrics.get("run_end", 0) or 0) >= 1, 120.0)
    metrics = manager.metrics()
    _report(
        15,
        "metrics close the conversation",
        int(metrics.get("sessions", 0)) >= 2
        and int(metrics.get("stt_end", 0) or 0) >= 1
        and int(metrics.get("run_end", 0) or 0) >= 1,
        f"sessions={metrics.get('sessions')} stt_end={metrics.get('stt_end')} "
        f"run_end={metrics.get('run_end')} errors={metrics.get('errors')}",
    )

    voice_pe.stop()
    listener.stop()
    failed = sum(1 for _i, _n, ok, _d in RESULTS if not ok)
    print(f"-- {len(RESULTS) - failed}/{len(RESULTS)} checkpoints passed", flush=True)
    if failed:
        # Speech-dependent checkpoints: the stock firmware closes the microphone
        # after its own buffered window, so 9 to 11 only fill in when a sentence
        # lands inside that window of the open run.
        print(
            "   note: 9 to 11 need one spoken sentence while the run is open; "
            "the rest is protocol state.",
            flush=True,
        )
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
