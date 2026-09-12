"""Hardware smoke test for the Voice PE integration (stock retail firmware).

Run with the satellite powered and on the same LAN, Ollama reachable, and the
centre button within reach:

    python scripts/_voice_pe_smoke.py [host]

The script boots the real stack - ``Database``, ``Settings``, the Piper TTS
engine, ``VoiceListener`` (WebRTC VAD + Whisper) in its own thread and the
``VoicePEManager`` with one ``VoicePEDevice`` - then walks one conversation:

    button -> API audio -> VAD -> Whisper -> Jarvis reply
           -> device GET of that WAV key -> audible playback -> AnnounceFinished

Every checkpoint compares a baseline read just before it with the value after,
so an older counter cannot satisfy a later step. Events come from the device's
per-generation ledger, the TTS server counts per payload key with the misses
separate, and the media step is judged by the device's own state pushes.

Each line is ``ok`` or ``FAIL``; the exit code is the number of failed
checkpoints, so CI can call it directly. No Home Assistant server, no mocks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The console on Windows is cp1252; the checkpoints print emoji from the stack.
try:
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


def _poll(predicate, timeout_s: float, step_s: float = 0.05, recorder=None):
    """Poll ``predicate`` until it returns a truthy value or time is up."""
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


def _counter(device, key: str) -> int:
    return int(device.metrics.get(key, 0) or 0)


def _ledger(device, generation: int) -> list:
    """Names of the Voice Assistant events sent for one session generation."""
    return [name for gen, name in device.event_ledger if int(gen) == int(generation)]


def _is_subsequence(needle: list, haystack: list) -> bool:
    """Order-aware check: ``needle`` appears inside ``haystack`` in order."""
    index = 0
    for item in haystack:
        if index < len(needle) and item == needle[index]:
            index += 1
    return index == len(needle)


#: The one Czech sentence the hardware checkpoints are spoken with.
SENTENCE = "Hej toustovači, jaké je počasí v Praze?"
#: Meaningful words the transcript has to carry for a turn to count.
CONTENT_WORDS = ("počasí", "Praha")
#: Stem aliases per content word, in the diacritic-free normalised form.
CONTENT_STEMS: tuple[tuple[str, ...], ...] = (
    ("pocas",),
    ("praha", "praze", "prah", "rakous"),
)
#: Order of the full per-generation chain, from microphone to playback.
CHAIN = [
    "STT_START",
    "STT_END",
    "INTENT_START",
    "TTS_START",
    "TTS_END",
    "RUN_END",
]


def _hardware_timeout(settings) -> float:
    """The one hardware budget, from config with the 180 s default."""
    try:
        return float(getattr(settings, "voice_pe_hardware_timeout_s", 180.0) or 180.0)
    except Exception:
        return 180.0


def _phase_of(device, generation: int, server=None, key: str = "") -> str:
    """Which stage the hardware chain is waiting on right now.

    Read straight from the per-generation ledger and the WAV server counters, so
    a timeout can name the blocking phase instead of a plain "timed out".
    """
    names = _ledger(device, generation)
    if "STT_END" not in names:
        return "stt_decoding" if "STT_START" in names else "waiting_for_microphone"
    if "INTENT_END" not in names:
        return "llm_generating"
    if "TTS_START" not in names:
        return "waiting_for_tts_start"
    if "TTS_END" not in names:
        return "tts_generating"
    if server is not None and key and int(server.hit_count(key)) == 0:
        return "device_fetching_wav"
    if not int(device.metrics.get("announcements_finished", 0) or 0):
        return "device_playing"
    return "complete"


def _timeline(device, generation: int) -> dict:
    try:
        return dict(device.latency_summary(generation))
    except Exception:
        return {}


def _chain_proof(device, generation: int, expected_text: str = "") -> dict:
    """Everything one generation must show, in one dict."""
    names = _ledger(device, generation)
    key = f"{device.connection_generation}-{generation}"
    server = getattr(device, "_http", None)
    delivery = {}
    try:
        delivery = device.media_delivery()
    except Exception:
        delivery = {}
    stt_record = {}
    return {
        "generation": int(generation),
        "key": key,
        "ledger": names,
        "order_ok": _is_subsequence(CHAIN, names),
        "stt_end_success": _is_subsequence(["STT_END"], names),
        "wav_stored_bytes": int(delivery.get("stored_bytes", 0) or 0),
        "wav_bytes_metric": int(delivery.get("wav_bytes_metric", 0) or 0),
        "http_status": delivery.get("status"),
        "http_hits": int(delivery.get("hits", 0) or 0),
        "served_bytes": int(delivery.get("served_bytes", 0) or 0),
        "content_type": str(delivery.get("content_type") or ""),
        "port": int(delivery.get("port", 0) or 0),
        "url_deliveries": _counter(device, "tts_url_deliveries"),
        "announcements_finished": _counter(device, "announcements_finished"),
        "last_announce_success": device.last_announce_success,
        "last_finished_generation": int(device.last_finished_generation),
        "reply_source": str(device.metrics.get("reply_source") or ""),
        "latency": _timeline(device, generation),
        "phase": _phase_of(device, generation, server, key),
        "expected_text": expected_text,
    }


def _norm_text(text: str) -> str:
    """Lowercased, diacritic-free, single-spaced form for word matching."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


def _text_matches(text: str) -> bool:
    """The transcript carries the content words, case-insensitively.

    Whisper's Czech output differs from the keyboard form in diacritics
    (``počasy`` vs ``počasí``) and in the city spelling heard by the
    decoder; every content word is therefore accepted through an alias
    stem on the normalised form.
    """
    lowered = _norm_text(text)
    return all(any(stem in lowered for stem in stems) for stems in CONTENT_STEMS)


def _button_trace(device) -> list:
    try:
        return list(device.metrics.get("button_trace") or [])
    except Exception:
        return []


def _submit(manager, coro):
    """Run one coroutine on the manager loop and return its future."""
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, manager._loop)


def _await(manager, coro, timeout_s: float = 32.0):
    return _submit(manager, coro).result(timeout=timeout_s)


def _settings_with_host(settings, host: str):
    """Same settings with one manual host and the feature on.

    The Czech sentence pins the decoder language to ``cs`` and uses the
    canonical log-domain confidence threshold.
    """
    values = {
        n: getattr(settings, n, None) for n in dir(settings) if not n.startswith("__")
    }
    values["voice_pe_enabled"] = True
    values["voice_pe_host"] = host
    values["whisper_language"] = "cs"
    values["whisper_min_avg_logprob"] = -0.7
    return SimpleNamespace(**values)


def _wav_frame_hashes(path: str, frame_ms: int, sample_rate: int) -> list:
    """SHA of each fixed-size grid frame; 0 = padding, 1 = voiced."""
    import hashlib
    import wave

    with wave.open(path, "rb") as handle:
        n = handle.getnframes()
        data = handle.readframes(n)
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    out = []
    for i, off in enumerate(range(0, len(data), frame_samples * 2)):
        chunk = data[off: off + frame_samples * 2]
        if len(chunk) < frame_samples * 2:
            break
        out.append((i, hashlib.sha1(chunk).hexdigest()[:12]))
    return out


def _adjacent_duplicates(hashes: list) -> tuple[int, int, list]:
    """N non-zero consecutive duplicates, N total, first 3 pairs for details."""
    total = len(hashes)
    if total < 2:
        return 0, 0, []
    dups: list = []
    for i in range(1, total):
        prev = hashes[i - 1]
        cur = hashes[i]
        # A frame of only zero bytes has its own SHA1; padding is a
        # legitimate continuation, not a duplicated packet.
        if cur[1] and prev[1] and cur[1] == prev[1] and len(cur[1]) == 12:
            dups.append((prev[0], cur[0]))
    return len(dups), total, dups[:3]


def _firmware_metadata(device) -> dict:
    """The device's own declared audio parameters, not a derived assumption."""
    info = {}
    try:
        raw = getattr(device, "identity", {}) or {}
        info.update(
            {
                "esphome_version": raw.get("esphome_version"),
                "project_name": raw.get("project_name"),
                "project_version": raw.get("project_version"),
                "api_version": raw.get("api_version"),
                "model": raw.get("model"),
            }
        )
    except Exception:
        pass
    return info


def _acoustic_heuristic_verdict(wav_path: str | None, rec: dict, frame_count: int) -> str:
    """Heuristic read from the same WAV bytes. Exact, diff-friendly values:
    `acoustic_silence`, `acoustic_low_energy`, `acoustic_clipped`,
    `transcript_expected_phrase`, `transcript_other:<text>`.
    `human_listen_verdict` is a separate manually-confirmed field (see
    `HUMAN_VERDICT_KEYS`) and is not filled automatically from here.
    """
    if wav_path is None:
        return "acoustic_silence"
    import wave

    try:
        with wave.open(wav_path, "rb") as handle:
            n = handle.getnframes()
            handle.readframes(n)
    except Exception:
        return "acoustic_silence"
    rms = float(rec.get("rms") or 0.0)
    peak = float(rec.get("peak") or 0.0)
    clipped = float(rec.get("clipped_ratio") or 0.0)
    rows_text = " ".join(
        str((row.get("text", "") if isinstance(row, dict) else getattr(row, "text", "")) or "")
        for row in rec.get("raw_segments") or []
    ).strip()
    text = str(
        rec.get("filtered_text") or rows_text or rec.get("raw_transcript") or ""
    ).strip()
    if rms <= 0.0 or peak == 0.0 or n < 160:
        return "acoustic_silence"
    if clipped > 0.15:
        return "acoustic_clipped"
    if rms < 1e-3:
        return "acoustic_low_energy"
    lowered = text.lower()
    if "počasí" in lowered and "praha" in lowered:
        return "transcript_expected_phrase"
    return f"transcript_other:{text}" if text else "acoustic_low_energy"


#: Only ever filled with the manual confirmation values below; the helper
#: above never writes it and the smoke test leaves the field as `None`.
HUMAN_VERDICT_KEYS = ("expected_sentence", "other_speech", "noise", "silence", "distorted")


def _human_verdict_or_none(value: object) -> str | None:
    """Accept only the five exact manual confirmations plus `None`."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in HUMAN_VERDICT_KEYS else None


def _poll_trace_count(index: int) -> int:
    """Helper for ``_poll``'s ``recorder``: returns ``index`` unchanged."""
    return index


# `def _diagnostic_two_channel(` follows next; kept as the function below.

def _diagnostic_two_channel(
    device,
    manager,
    duration_s: float = 10.0,
    prefix: str = "pe_diag",
) -> dict:
    """One fixed 10-second raw capture of both API channels, no VAD, no decoder.

    Called *just after* ``CAPTURING generation=N`` confirmed the event. Both
    channels buffer for the whole ``duration_s`` or until the real EOS/cancel;
    the first packet only records its timestamp, no early cut. The readback is
    the per-channel table, the two WAV paths and the exact reason the channel
    was locked, so the ``SpeechEvidence`` gate can be checked from that alone.

    ``human_listen_verdict`` stays ``None``; it's confirmed by hand.
    """
    import time

    out: dict = {
        "duration_s": float(duration_s),
        "pipeline_start_ts": int(device.session_generation or 0),
        "capture_start_ns": None,
        "capture_end_ns": None,
        "first_data_ts": None,
        "first_data2_ts": None,
        "selected_channel": None,
        "selection_reason": None,
        "selected_at_ns": None,
        "channel_files": {},
        "channels": {},
        "mute": None,
        "lease": None,
        "human_listen_verdict": None,
    }
    stream = device._stream() if device.turn_context() is not None else None
    # Fixed 10 s window from *now*: the caller has already confirmed the event;
    # the loop keeps sampling while both raw channels are being buffered.
    out["capture_start_ns"] = int(time.monotonic_ns())
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if device.session is None:
            break
        time.sleep(0.02)
    out["capture_end_ns"] = int(time.monotonic_ns())

    rows = []
    try:
        rows = list(device.packet_stats(stream) or [])
    except Exception:
        rows = []
    first_ns = min((int(r.timestamp_ns) for r in rows), default=None)
    for ch in (0, 1):
        ts = [
            int(r.timestamp_ns)
            for r in rows
            if int(r.channel) == ch
        ]
        key = "first_data_ts" if ch == 0 else "first_data2_ts"
        out[key] = round((min(ts) - first_ns) / 1e6, 3) if ts and first_ns is not None else None

    snapshot = None
    try:
        snapshot = device.health_snapshot()
    except Exception:
        pass
    if snapshot:
        # Device-level snapshot: identity, state, LED, generations. The
        # per-packet ``rows`` from ``packet_stats`` keep their own name.
        _mute: dict = {}
        for key in (
            "device_state",
            "led_phase_id",
            "connection_generation",
            "session_generation",
            "audio_queue_ms",
        ):
            if key in snapshot:
                _mute[key] = snapshot[key]
        if _mute:
            out["mute"] = _mute
    if device.media is not None:
        m = device.media.snapshot()
        try:
            out["lease"] = {
                "volume_source": m.get("volume_source"),
                "pushes": m.get("pushes"),
                "accepts": m.get("accepts"),
            }
            if m.get("device_state") is not None or m.get("device_muted") is not None:
                out["mute"] = {
                    "device_state": m.get("device_state"),
                    "device_volume": m.get("device_volume"),
                    "device_muted": m.get("device_muted"),
                    "volume_source": m.get("volume_source"),
                    "pushes": m.get("pushes"),
                    "accepts": m.get("accepts"),
                }
        except Exception:
            pass

    # Per-channel raw readback.
    for ch in (0, 1):
        subset = [r for r in rows if int(r.channel) == ch]
        subset.sort(key=lambda r: int(r.timestamp_ns))
        missing = 0
        for k in range(1, len(subset)):
            gap_ms = (int(subset[k].timestamp_ns) - int(subset[k - 1].timestamp_ns)) / 1e6
            if gap_ms > 50.0:
                missing += 1
        bytes_sum = sum(int(r.byte_length or 0) for r in subset)
        rms_list = [float(r.rms or 0) for r in subset if r.rms]
        rms = sum(rms_list) / len(rms_list) if rms_list else 0.0
        peak = max((float(r.peak or 0.0) for r in subset), default=0.0)
        clipped_ratio = sum(1 for r in subset if float(r.peak or 0.0) >= 0.999) / len(subset) if subset else 0.0
        row_detail = {
            "packets": len(subset),
            "bytes": bytes_sum,
            "duration_s": round(bytes_sum / 2.0 / 16000.0, 4) if bytes_sum else 0.0,
            "missing_intervals_gt_50ms": missing,
            "rms_packet_mean": round(rms, 8),
            "packet_peak": round(peak, 8),
            "clipped_ratio_packets": round(clipped_ratio, 6),
        }
        # Per-channel speech evidence rows ``(dev, conn, sess, candidate)``
        # are kept by the shared ingress; the device itself only delegates
        # ``packet_stats``. ``channel_stats`` is the ingress-side table.
        _ing = getattr(device, "_ingress", None)
        _cstats = _ing.channel_stats() if _ing is not None else []
        for r_tuple in (_cstats if isinstance(_cstats, list) else []):
            if len(r_tuple) == 4 and int(r_tuple[3].channel) == ch:
                c = r_tuple[3]
                row_detail.update(
                    {
                        "voiced_frame_count": int(c.voiced_frame_count),
                        "speech_span_ms": int(c.speech_span_ms),
                        "voiced_rms_dbfs": c.voiced_rms_dbfs,
                        "silent_rms_dbfs": c.silent_rms_dbfs,
                        "snr_db": c.snr_db,
                        "clipped_ratio": c.clipped_ratio,
                        "admissible": bool(c.admissible),
                        "nonzero_samples": int(c.nonzero_samples),
                    }
                )
                break
        out["channels"][ch] = row_detail

    # Two WAVs for the human replay; also ``offline_candidates`` so the
    # recommended channel is computed *after* the capture, offline.
    dumps = {}
    try:
        dumps = device._ingress.dump_wavs(prefix, stream) or {}
    except Exception:
        pass
    out["channel_files"] = {int(k): v for k, v in (dumps or {}).items()}

    out["selected_channel"] = device.selected_audio_channel(stream)
    try:
        out["selection_reason"] = device._ingress.selection_reason(stream)
        out["selected_at_ns"] = device._ingress.selection_at_ns(stream)
    except Exception:
        pass
    return out

def _print_two_channel_diag(diag: dict) -> None:
    print("   --two-channel raw capture:", flush=True)
    print(
        f"   window: {diag.get('capture_start_ns')} .. {diag.get('capture_end_ns')} "
        f"({(int(diag.get('capture_end_ns') or 0) - int(diag.get('capture_start_ns') or 0))/1e6:.0f} ms)",
        flush=True,
    )
    print(
        f"   first packet ms: data={diag.get('first_data_ts')} data2={diag.get('first_data2_ts')}",
        flush=True,
    )
    for ch, row in sorted((diag.get("channels") or {}).items()):
        print(
            f"   channel {ch}: packets={row.get('packets')} bytes={row.get('bytes')} "
            f"duration_s={row.get('duration_s')} missing_gt_50ms={row.get('missing_intervals_gt_50ms')} "
            f"rms={row.get('rms_packet_mean')} peak={row.get('packet_peak')} "
            f"clipped={row.get('clipped_ratio')} "
            f"voiced_frames={row.get('voiced_frame_count')} "
            f"speech_span_ms={row.get('speech_span_ms')} "
            f"voiced_rms_dbfs={row.get('voiced_rms_dbfs')} "
            f"silent_rms_dbfs={row.get('silent_rms_dbfs')} snr_db={row.get('snr_db')} "
            f"admissible={row.get('admissible')}",
            flush=True,
        )
    for ch, meta in sorted((diag.get("channel_files") or {}).items()):
        print(
            f"   saved channel {ch}: {meta['path']} bytes={meta['bytes']} samples={meta['samples']}",
            flush=True,
        )
    print(
        f"   selection: channel={diag.get('selected_channel')} reason={diag.get('selection_reason')!r} "
        f"selected_at_ns={diag.get('selected_at_ns')} "
        f"active_audio_source={diag.get('active_audio_source')} "
        f"mute={diag.get('mute')} lease={diag.get('lease')}",
        flush=True,
    )
    print(
        f"   human_listen_verdict={diag.get('human_listen_verdict')} "
        "(confirm by hand)",
        flush=True,
    )


def main() -> int:
    args = list(sys.argv[1:])
    host = None
    two_channel_diagnostic = False
    for arg in args:
        if arg == "--two-channel":
            two_channel_diagnostic = True
        elif not arg.startswith("--"):
            host = arg

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

    # ``--two-channel``: fixed 10-second raw capture of both API channels, no
    # main VAD, no decoder, no auto-selection abort. The sequence is strict so
    # a press cannot race: callbacks are wired → subscription active → both
    # generations snapshotted as a baseline → an event armed on
    # ``session_generation > baseline`` → the ``ARMED`` line is the last thing
    # before the wait → the press is taken exactly once by the event.
    if two_channel_diagnostic:
        # Event-driven, generation-scoped, non-overwriting. The press may land
        # while the loop is inside a callback, so the item is pushed with
        # ``call_soon_threadsafe`` from the loop thread and awaited as a queue
        # item; generations at or below the baseline are stale and ignored.
        import asyncio

        baseline_connection = int(device.connection_generation)
        baseline_session = int(device.session_generation or 0)

        async def _make_queue() -> "asyncio.Queue":
            # Created inside the loop thread to bind it to manager._loop.
            return asyncio.Queue()

        start_queue = _await(manager, _make_queue(), 20.0)

        async def _take_start(queue: "asyncio.Queue", limit: float):
            try:
                return await asyncio.wait_for(queue.get(), limit)
            except asyncio.TimeoutError:
                return None

        def _on_start(generation: int) -> None:
            device._timeline_stamp("physical_press_event")
            if int(generation) > baseline_session:
                manager._loop.call_soon_threadsafe(
                    start_queue.put_nowait, int(generation)
                )
                device._timeline_stamp("queue_put_scheduled")

        def _on_conn() -> None:
            # Reconnect-safe: the listener list itself survives reconnects;
            # the only thing to confirm is that the queue is still reachable.
            device._timeline_stamp("connection_refreshed")

        device._timeline_stamp("listener_registered")
        unsub = device.add_pipeline_start_listener(_on_start)
        print("ARMED \u2014 PRESS BUTTON NOW", flush=True)
        device._timeline_stamp("ARMED_printed")
        received = _await(manager, _take_start(start_queue, 180.0), 185.0)
        unsub()
        opened = received is not None and int(received) > baseline_session
        if received is not None:
            device._timeline_stamp("queue_item_received")
        if not opened:
            _report(
                99,
                "--two-channel captured after ARMED",
                False,
                "harness-FAIL: no pipeline_start after ARMED within 180 s",
            )
            return 1
        print(
            f"CAPTURING generation={int(received)}",
            flush=True,
        )
        device._timeline_stamp("CAPTURING_started")
        diag = _diagnostic_two_channel(device, manager, duration_s=10.0, prefix="pe_diag")
        _print_two_channel_diag(diag)
        # One monotonic timeline for the whole press \u2192 capture chain; the
        # first stamp (listener_registered) is the zero point.
        _tl = list(device.pipeline_timeline)
        _t0 = _tl[0][1] if _tl else 0
        for stage, ns in _tl:
            print(
                f"   t+{(ns - _t0) / 1e6:9.3f} ms  {stage}",
                flush=True,
            )
        offline = device._ingress.offline_candidates(device._stream())
        if offline:
            print(
                "   offline candidates:",
                ", ".join(
                    f"ch{c.channel}(adm={int(c.admissible)}"
                    f" snr={c.snr_db if c.snr_db is not None else '-'}"
                    f" voiced={int(c.voiced_frame_count)})"
                    for c in offline
                ),
                flush=True,
            )
        # Checkpoint 99 is conjunctive and generation-scoped: the generation the
        # event carried, the generation the capture reports, both WAVs belonging
        # to it, and a real header-plus-data size (> 44 bytes) must all agree.
        file_meta = {int(k): dict(v or {}) for k, v in (diag.get("channel_files") or {}).items()}
        capture_generation = int(diag.get("pipeline_start_ts") or 0)
        wavs_ok = (
            len(file_meta) >= 2
            and all(int(m.get("bytes") or 0) > 44 for m in file_meta.values())
        )
        _sizes = ", ".join(f"{c}:{int(m.get('bytes') or 0)}" for c, m in sorted(file_meta.items()))
        _report(
            99,
            "--two-channel diagnostic captured",
            wavs_ok
            and capture_generation == int(received)
            and diag.get("selected_channel") is not None,
            f"captured_generation={capture_generation} event_generation={int(received)} "
            f"channels={sorted(file_meta)} "
            f"sizes={{{_sizes}}} "
            f"selected={diag.get('selected_channel')} "
            f"reason={diag.get('selection_reason') or '-'}",
        )
        # Keep the harness honest: a ``None`` verdict from the heuristic is
        # still a captured diagnostic if both WAVs landed.
        return 0 if wavs_ok else 1

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
        f"{device.capabilities.entity_count} entities, "
        f"led_key={device.entities.led_key()}",
    )

    # 4. Wake-word mode as the device reports it, not as the write requested.
    _report(
        4,
        "wake words off per device read-back",
        bool(device.wake_words_disabled)
        and int(device.metrics.get("wake_words_active", -1)) == 0,
        f"disabled={device.wake_words_disabled} "
        f"active={device.metrics.get('wake_words_active')} err={device.last_error or '-'}",
    )

    # 4b. Models first: the listener loads Whisper and pages every LLM role in
    #     its own thread, and a cold load inside a spoken window would eat it.
    #     The warm-up itself is outside the 180 s per-turn budget; only the
    #     post-READY production proof runs inside a turn.
    warmed = _poll(
        lambda: getattr(listener, "model", None) is not None
        and len(getattr(listener, "_llm_warmup_results", {}) or {}) >= 4,
        90.0,
        step_s=0.2,
    )
    per_role = dict(getattr(listener, "_llm_warmup_results", {}) or {})
    metrics = dict(getattr(listener, "_llm_warmup_metrics", {}) or {})
    for role in ("chat", "judge", "router", "embed"):
        entry = per_role.get(role)
        if entry is None:
            print(f"   warmup {role}: -", flush=True)
            continue
        name, ok = entry
        m = dict(metrics.get(role) or {})
        # ``gpu_layers`` is the only number that names a CPU-only server vs
        # the GPU-resident chat model on an RTX 4090: ``None`` means the server
        # gave no hint (``accelerator_unknown``), ``0`` is a real
        # ``cpu_confirmed``, ``>= 1`` is ``gpu_confirmed``. There is no ``-1``.
        # ``first_token_s`` / ``completion_latency_s`` and ``tok/s`` come from
        # the very same streamed inference that produced them.
        _lat_key = (
            "first_token_latency"
            if "first_token_latency" in m
            else "completion_latency"
        )
        acc = m.get("accelerator", "accelerator_unknown")
        print(
            f"   warmup {role}: {'ok' if ok else 'FAIL'} model={name} "
            f"{_lat_key}={m.get(_lat_key, '?')} "
            f"tok/s={m.get('tokens_per_s', '?')} "
            f"gpu_layers={m.get('gpu_layers', '?')} "
            f"accelerator={acc} "
            f"backend={m.get('backend', '?')}",
            flush=True,
        )
    _chat_m = dict(metrics.get("chat") or {})
    _chat_gpu = _chat_m.get("gpu_layers")          # int >= 0 or None
    _chat_acc = _chat_m.get("accelerator", "accelerator_unknown")
    _chat_tps = _chat_m.get("tokens_per_s")
    _min_tps = float(settings.voice_pe_llm_min_tokens_per_s)
    _chat_27b = any(
        "27B" in str(v[0] if isinstance(v, tuple) else "") for v in per_role.values()
    )
    critical = all(bool(per_role.get(k) and per_role[k][1]) for k in ("chat", "judge"))
    # Campaign readiness for the 27B chat, resolved in this order:
    #   gpu_confirmed                                  -> ready
    #   cpu_confirmed (n_gpu_layers == 0)              -> not_campaign_ready
    #   accelerator_unknown but streamed tok/s > limit -> ready by latency
    #   accelerator_unknown and latency below limit    -> not_campaign_ready
    if _chat_acc == "gpu_confirmed":
        _llm_status = "campaign_ready"
    elif _chat_acc == "cpu_confirmed":
        _llm_status = "not_campaign_ready"
    elif _chat_tps is not None and float(_chat_tps) > _min_tps:
        _llm_status = "campaign_ready"
    elif not _chat_27b:
        _llm_status = "campaign_ready"
    else:
        _llm_status = "not_campaign_ready"
    print(
        f"   models ready={bool(warmed)} critical_ok={critical} "
        f"27b_gpu={_chat_gpu if _chat_gpu is not None else 'unknown'} "
        f"accelerator={_chat_acc} min_tok_s={_min_tps} "
        f"llm_backend={_llm_status} "
        f"roles={sorted(per_role)}",
        flush=True,
    )
    if not critical:
        print(
            "   note: chat/judge not resident; reply uses the deterministic "
            "diagnostic path. Run the separate 27B proof after READY/warmup.",
            flush=True,
        )
    if _llm_status == "not_campaign_ready" and _chat_27b and critical:
        # The warm-up ran but the model is either CPU-resident or its
        # accelerator stayed unknown while throughput stayed below the limit;
        # the production 27B turn cannot be counted as campaign-ready.
        print("   llm_backend_not_campaign_ready", flush=True)


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
    generation = int(device.session_generation)
    _report(
        5,
        "one pipeline run opened, generation moved by one",
        bool(started)
        and generation == gen_before + 1
        and "RUN_START" in _ledger(device, generation),
        f"generation={generation} (was {gen_before}) "
        f"ledger={_ledger(device, generation)}",
    )
    if not started:
        voice_pe.stop()
        listener.stop()
        return 1

    sink = getattr(listener, "_voice_pe_sink", None)
    context = sink.current_context() if sink is not None else None
    _report(
        6,
        "turn context names the satellite and this generation",
        context is not None
        and context.source == "voice_pe"
        and context.device_id == device.device_id
        and int(context.session_generation) == generation,
        f"{None if context is None else ''}"
        + (
            ""
            if context is None
            else (
                f"source={context.source} device_id={context.device_id} "
                f"connection_generation={context.connection_generation} "
                f"session_generation={context.session_generation} "
                f"stream_id={context.stream}"
            )
        ),
    )

    # 7. Audio really reached the VAD: the listener's own milestone followed.
    #    One spoken sentence covers checkpoints 7 to 9, so it is prompted here
    #    where the first of them starts waiting.
    chunks_before = _counter(device, "audio_chunks")
    print(
        "   -> speak one short sentence into the satellite (45 s)",
        flush=True,
    )
    _poll(lambda: "STT_VAD_START" in _ledger(device, generation), 45.0, recorder=_record)
    _report(
        7,
        "microphone blocks reached the VAD (not just the queue)",
        "STT_VAD_START" in _ledger(device, generation)
        and _counter(device, "audio_chunks") > chunks_before,
        f"chunks {chunks_before} -> {_counter(device, 'audio_chunks')} "
        f"ledger={_ledger(device, generation)}",
    )
    _report(
        8,
        "one microphone pump for this connection generation",
        int(device.metrics.get("pump_tasks", 0)) == 1,
        f"pump_tasks={device.metrics.get('pump_tasks')}",
    )

    # One hardware budget drives every leg: the waits are event-driven, never a
    # fixed sleep, and a timeout names the phase that blocked.
    hardware_s = _hardware_timeout(settings)

    def _sat_record() -> dict:
        values = getattr(listener, "metrics", {}) or {}
        return dict(
            values.get("last_satellite_segment") or values.get("last_segment") or {}
        )

    def _sat_text(record: dict) -> str:
        return str(record.get("filtered_text") or record.get("raw_transcript") or "")

    def _wait(name: str, predicate, budget: float = None) -> bool:
        reached = bool(
            _poll(predicate, budget or hardware_s, step_s=0.05, recorder=_record)
        )
        if not reached:
            proof = _chain_proof(device, generation)
            print(
                f"   note: {name} still waiting after {budget or hardware_s:.0f} s; "
                f"phase={proof['phase']} last_event="
                f"{(proof['ledger'] or ['-'])[-1]} ledger={proof['ledger']}",
                flush=True,
            )
        return reached

    # 9. VAD + Whisper produced a transcript whose own record says success, the
    #    decoder returned a row, that text reached dispatch, and it carries the
    #    content words of the CZECH sentence (case-insensitive).
    stt_before = _counter(device, "stt_end")
    ok9 = _wait(
        "stt_success_with_dispatch",
        lambda: str(_sat_record().get("status") or "") == "success"
        and int(_sat_record().get("row_count") or 0) >= 1
        and "STT_END" in _ledger(device, generation)
        and "INTENT_START" in _ledger(device, generation),
    )
    stt_record = _sat_record()
    stt_levels = dict(stt_record.get("audio_level") or {})
    turn_transcripts = {_counter(device, "stt_end_success"): _sat_text(stt_record)}
    #: Full-chain proof of each completed turn, printed in the final report.
    turn_proofs: list = []
    sat_text_first = _sat_text(stt_record) or _sat_text(stt_record)
    _report(
        9,
        "VAD + Whisper transcript accepted for the open run",
        ok9
        and _counter(device, "stt_end") > stt_before
        and _counter(device, "stt_end_success") > 0
        and bool(str(stt_record.get("raw_transcript") or "").strip())
        and _text_matches(sat_text_first),
        f"stt_end {stt_before} -> {_counter(device, 'stt_end')} "
        f"success={_counter(device, 'stt_end_success')} "
        f"status={stt_record.get('status')} "
        f"raw='{stt_record.get('raw_transcript')}' "
        f"rows={stt_record.get('row_count')} "
        f"avg_logprob={stt_record.get('avg_logprob')} "
        f"no_speech_prob={stt_record.get('no_speech_prob')} "
        f"dbfs_rms={stt_levels.get('dbfs_rms')} "
        f"dbfs_peak={stt_levels.get('dbfs_peak')} "
        f"gain_db={(stt_record.get('preprocessor') or {}).get('applied_gain_db')} "
        f"snr_db={stt_levels.get('snr_db')} "
        f"speech_span={stt_record.get('speech_span_s')}s "
        f"query='{stt_record.get('query')}' "
        f"filtered='{sat_text_first}' "
        f"words_ok={_text_matches(sat_text_first)}",
    )

    # 10. Reply reached the device as a LAN WAV URL on the still-open server.
    delivered_before = _counter(device, "tts_url_deliveries")
    ok10 = _wait(
        "tts_url_delivery",
        lambda: _counter(device, "tts_url_deliveries") > delivered_before
        and "TTS_END" in _ledger(device, generation),
    )
    server = device._http
    key = device._tts_media_id
    proof10 = _chain_proof(device, generation)
    _report(
        10,
        "TTS_END carried a LAN URL on a still-open server",
        ok10
        and server is not None
        and server.port > 0
        and bool(key)
        and proof10["wav_stored_bytes"] > 0
        and proof10["wav_bytes_metric"] > 0,
        f"deliveries={_counter(device, 'tts_url_deliveries')} "
        f"port={getattr(server, 'port', 0)} key='{key}' "
        f"wav_bytes={proof10['wav_stored_bytes']} "
        f"reply_source={proof10['reply_source'] or '-'}",
    )

    # 11. The satellite fetched that same key, with a status and a byte count:
    #     a URL alone is not a delivery.
    hits_before = server.hit_count(key) if server is not None else 0
    missing_before = server.missing if server is not None else 0
    ok11 = _wait(
        "device_wav_get",
        lambda: (
            server is not None
            and server.hit_count(key) >= 1
            and server.status_codes.get(key) == 200
            and int(server.served_bytes.get(key, 0)) > 0
        ),
    )
    proof11 = _chain_proof(device, generation)
    _report(
        11,
        "device fetched that exact WAV key with 200 and bytes",
        ok11
        and proof11["http_hits"] >= 1
        and proof11["http_status"] == 200
        and proof11["served_bytes"] > 0
        and proof11["content_type"] == "audio/wav"
        and proof11["served_bytes"] == proof11["wav_stored_bytes"],
        f"key '{key}' hits {hits_before} -> {proof11['http_hits']} "
        f"status={proof11['http_status']} served_bytes={proof11['served_bytes']} "
        f"stored={proof11['wav_stored_bytes']} type={proof11['content_type']} "
        f"misses {missing_before} -> {getattr(server, 'missing', 0) if server else 0}",
    )

    # 12. Audible playback, closed by the finished report of this delivery.
    finished_before = _counter(device, "announcements_finished")
    ok12 = _wait(
        "announce_finished",
        lambda: _counter(device, "announcements_finished") > finished_before
        and int(device.last_finished_generation) == int(generation),
    )
    _report(
        12,
        "playback closed by a successful AnnounceFinished of this run",
        ok12
        and device.last_announce_success is True
        and int(device.last_finished_generation) == int(generation),
        f"finished={_counter(device, 'announcements_finished')} "
        f"success={device.last_announce_success} generation="
        f"{device.last_finished_generation}/{generation} "
        f"playback_ms={_timeline(device, generation).get('playback_ms')}",
    )

    # 13. Event order for this generation, avatar order, and the ``led_ring``
    #     entity's own push as the device-side acknowledgement of the ring.
    ok13 = _wait(
        "run_end",
        lambda: _ledger(device, generation).count("RUN_END") >= 1,
    )
    ordered = _ledger(device, generation)
    # The pre-announce already wrote a ``speaking`` before the mic opened, so the
    # progression is judged from the last ``thinking`` onwards.
    try:
        thought_at = max(i for i, name in enumerate(avatars) if name == "thinking")
    except ValueError:
        thought_at = -1
    spoke_after_thinking = any(name == "speaking" for name in avatars[thought_at + 1 :])
    _report(
        13,
        "event order, avatar order and the ring's own state push",
        ok13
        and _is_subsequence(["INTENT_START", "TTS_START", "TTS_END", "RUN_END"], ordered)
        and _is_subsequence(["listening", "thinking"], avatars)
        and spoke_after_thinking
        and device.last_light_push is not None,
        f"ledger={ordered} avatars={avatars} led={phases} "
        f"latency={_timeline(device, generation)} "
        f"ring_push={device.last_light_push}",
    )
    first_proof = _chain_proof(device, generation, SENTENCE)
    first_proof["transcript"] = _sat_text(_sat_record())
    first_proof["words_ok"] = _text_matches(_sat_text(_sat_record()))
    turn_proofs.append(first_proof)
    # The turn's reply is also read back for the source of its text.
    turn_transcripts[1] = _sat_text(_sat_record())

    # 13b. The same sentence once more, this time into the PC microphone, so the
    #      two clips of one sentence can be compared like for like afterwards.
    import os as _os

    prefix = (_os.environ.get("JARVIS_VOICE_DIAG_WAV") or "").strip()
    base = Path(prefix).parent if prefix and Path(prefix).is_absolute() else Path.cwd()
    stem = Path(prefix).name if prefix else "peclip"

    def _dump_count() -> int:
        return len(list(base.glob(f"{stem}-*.json")))

    dumps_before = _dump_count()
    print("   -> say the same sentence to the PC microphone (45 s)", flush=True)
    _poll(lambda: _dump_count() > dumps_before, 45.0, step_s=0.2, recorder=_record)
    print(
        f"   clips dumped: {dumps_before} -> {_dump_count()} "
        f"(next decodes with scripts/_voice_pe_diag.py {prefix or stem})",
        flush=True,
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
        bool(second) and int(device.metrics.get("pump_tasks", 0)) == 1,
        f"generation={device.session_generation} "
        f"pump_tasks={device.metrics.get('pump_tasks')}",
    )

    # 15. Media commands with the device's own push as the acknowledgement.
    before = manager.media_state("")
    pushes_before = int(before.get("pushes", 0) or 0)
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
        lambda: int(manager.media_state("").get("pushes", 0) or 0) > pushes_before,
        12.0,
        recorder=_record,
    )
    after = manager.media_state("")
    _report(
        15,
        "pause/resume/volume/mute accepted with a later device push",
        accepted and int(after.get("pushes", 0) or 0) > pushes_before,
        f"accepted={accepted} pushes {pushes_before} -> {after.get('pushes')} "
        f"device_volume={after.get('device_volume')} "
        f"device_state={after.get('device_state')} source={after.get('volume_source')}",
    )

    # 16. Metrics close the conversation, on counts since the boot baseline. A
    #     skipped or filtered STT answer is not a success, so the success counter
    #     is the one that has to move here.
    _poll(lambda: _counter(device, "run_end") >= 2, 300.0, recorder=_record)
    _report(
        16,
        "metrics close the conversation",
        _counter(device, "sessions") >= 2
        and _counter(device, "stt_end_success") >= 1
        and _counter(device, "run_end") >= 2,
        f"sessions={_counter(device, 'sessions')} "
        f"stt_end={_counter(device, 'stt_end')} "
        f"success={_counter(device, 'stt_end_success')} "
        f"skipped={_counter(device, 'stt_end_skipped_too_short')} "
        f"filtered={_counter(device, 'stt_end_filtered')} "
        f"run_end={_counter(device, 'run_end')} errors={device.metrics.get('errors')}",
    )

    # 17/18. The real chain's two remaining shapes: one interruption closes one
    #      run, and a close of an older generation cannot touch the newer one.
    def _open_run(first_s: float, second_s: float) -> bool:
        """One run, by centre button or by the ``start_conversation`` fallback."""
        baseline = int(device.session_generation)
        opened = _poll(
            lambda: int(device.session_generation) > baseline,
            first_s,
            recorder=_record,
        )
        if opened:
            return True
        if device._client is not None and manager._loop is not None:
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
            return bool(
                _poll(
                    lambda: int(device.session_generation) > baseline,
                    second_s,
                    recorder=_record,
                )
            )
        return False

    print("   -> press the centre button for the interrupted run (12 s)", flush=True)
    interrupted_open = _open_run(12.0, 8.0)
    interrupted = device.turn_context()
    if interrupted_open:
        _await(
            manager,
            device.abort_run(
                interrupted.session_generation, "button_cancel", interrupted
            ),
        )
    closed = _ledger(device, interrupted.session_generation)
    _report(
        17,
        "an interruption closes exactly one run",
        bool(interrupted_open)
        and closed[-2:] == ["ERROR", "RUN_END"]
        and closed.count("RUN_END") == 1,
        f"generation={interrupted.session_generation} "
        f"source={interrupted.source} device_id={interrupted.device_id} "
        f"connection_generation={interrupted.connection_generation} "
        f"session_generation={interrupted.session_generation} ledger={closed} "
        f"session_released={device.session is None} "
        f"state={device.session_state.value}",
    )

    print("   -> press the centre button once more (12 s)", flush=True)
    reopened = _open_run(12.0, 8.0)
    newest = device.turn_context()
    pairs_before = _ledger(device, interrupted.session_generation).count("RUN_END")
    if reopened:
        # The older close arrives late, after the newer run is already open.
        _await(
            manager,
            device.abort_run(interrupted.session_generation, "late", interrupted),
        )
    stale_ledger = _ledger(device, interrupted.session_generation)
    _report(
        18,
        "a late close of the older generation is a no-op on the newer run",
        bool(reopened)
        and stale_ledger.count("RUN_END") == pairs_before
        and int(newest.session_generation) == int(interrupted.session_generation) + 1
        and device.session is not None,
        f"older={interrupted.session_generation} newest={newest.session_generation} "
        f"run_end_before={pairs_before} run_end_after={stale_ledger.count('RUN_END')} "
        f"older_ledger={stale_ledger} open={device.session_state.value}",
    )

    # 19-22. The shape of the assembled clip, the decoder entry, the second
    #        consecutive turn, and the rejection of older-generation frames.
    last_record = dict(getattr(listener, "metrics", {}).get("last_segment", {}) or {})
    frame_samples = int(getattr(listener, "_frame_samples", 320) or 320)
    clip_samples = 0
    non_zero = 0
    span_non_zero = 0
    if prefix:
        import json as _json_lib

        import numpy as _np
        import wave as _wave

        wav_files = sorted(base.glob(f"{stem}-*.wav"), key=lambda p: len(p.name))
        if wav_files:
            # The JSON next to this WAV carries the same clip's own bookkeeping,
            # so the grid check reads one file pair and nothing else.
            side = wav_files[-1].with_suffix(".json")
            dumped = {}
            if side.exists():
                try:
                    with open(side, encoding="utf-8") as handle:
                        dumped = _json_lib.load(handle)
                except Exception:
                    dumped = {}
            state = dict(dumped.get("frame_state") or {}) or last_record
            with _wave.open(str(wav_files[-1]), "rb") as handle:
                clip_samples = handle.getnframes()
                data = _np.frombuffer(
                    handle.readframes(handle.getnframes()), dtype="<i2"
                ).astype(_np.float64) / 32768.0
            non_zero = int(_np.count_nonzero(data))
            first = state.get("first_voiced_offset")
            last = state.get("last_voiced_offset")
            rms = float(_np.sqrt(_np.mean(_np.square(data)))) if data.size else 0.0
            peak = float(_np.max(_np.abs(data))) if data.size else 0.0
            if first is not None and last is not None and data.size:
                low = int(first) * frame_samples
                high = min(data.size, (int(last) + 1) * frame_samples)
                span = data[low:high]
                span_non_zero = int(_np.count_nonzero(span))
            grid_ok = clip_samples % frame_samples == 0
            total_frames = int(state.get("total_frame_count") or 0)
            continuous = (
                grid_ok
                and clip_samples // max(1, frame_samples) == total_frames
                and int(state.get("last_voiced_offset") or 0) + 1
                + int(state.get("post_roll_frames") or 0)
                == total_frames
            )
            dumped_levels = dict(dumped.get("audio_level") or {})
            preprocessor = dict(dumped.get("preprocessor") or {})
            _report(
                19,
                "captured waveform is one continuous frame grid",
                bool(continuous) and clip_samples > 0,
                f"clip={wav_files[-1].name} samples={clip_samples} "
                f"frame_samples={frame_samples} grid_ok={grid_ok} "
                f"total_frames={total_frames} "
                f"first_voiced={first} last_voiced={last} "
                f"post_roll_frames={state.get('post_roll_frames')} "
                f"non_zero={non_zero} voiced_span_non_zero={span_non_zero}/"
                f"{max(0, (int(last) - int(first) + 1) * frame_samples)}",
            )
            _report(
                20,
                "clip timings and levels reported",
                clip_samples > 0
                and dumped.get("speech_span_s") is not None
                and dumped.get("duration_s") is not None
                and rms > 0.0
                and peak > 0.0,
                f"clip_duration={dumped.get('duration_s')}s "
                f"speech_span={dumped.get('speech_span_s')}s "
                f"voiced_frames={state.get('voiced_frame_count')} "
                f"trailing_silence={state.get('trailing_silence_frames')} "
                f"post_roll_frames={state.get('post_roll_frames')} "
                f"rms={round(rms, 8)} peak={round(peak, 8)} "
                f"dbfs_rms={dumped_levels.get('dbfs_rms')} "
                f"dbfs_peak={dumped_levels.get('dbfs_peak')} "
                f"scale={dumped_levels.get('scale_ratio_int16_over_float32')} "
                f"raw_scale={dumped_levels.get('raw_scale_ratio_int16_over_float32')} "
                f"snr_db={dumped_levels.get('snr_db')} "
                f"applied_gain_db={preprocessor.get('applied_gain_db')} "
                f"limiter_hits={preprocessor.get('limiter_hits')}",
            )
        else:
            _report(19, "captured waveform is one continuous frame grid", False, "no WAV dumped")
            _report(20, "clip timings and levels reported", False, "no WAV dumped")
    else:
        print("   note: JARVIS_VOICE_DIAG_WAV unset, skipping 19/20", flush=True)

    listener_metrics = getattr(listener, "metrics", {}) or {}
    sat_record = dict(listener_metrics.get("last_satellite_segment") or last_record)
    _report(
        21,
        "transcribe() was entered with the resolved kwargs",
        int(listener_metrics.get("stt_end_success") or 0) >= 1
        and bool(sat_record.get("asr_backend"))
        and bool(sat_record.get("transcribe_kwargs")),
        f"backend={sat_record.get('asr_backend')} "
        f"version={sat_record.get('asr_version') or '-'} "
        f"kwargs={sat_record.get('transcribe_kwargs')} "
        f"rows={sat_record.get('row_count')} status={sat_record.get('status')} "
        f"listener_success={listener_metrics.get('stt_end_success')} "
        f"device_success={_counter(device, 'stt_end_success')}",
    )

    print("   -> speak the same phrase again for the second consecutive turn (45 s)", flush=True)
    success_before = int(
        (getattr(listener, "metrics", {}) or {}).get("stt_end_success") or 0
    )
    satellite_before = int(
        1
        if (getattr(listener, "metrics", {}) or {}).get("last_satellite_segment")
        else 0
    )
    deliveries_before_2 = _counter(device, "tts_url_deliveries")
    _poll(
        lambda: int(
            (getattr(listener, "metrics", {}) or {}).get("stt_end_success") or 0
        )
        > success_before
        and str(
            ((getattr(listener, "metrics", {}) or {}).get("last_satellite_segment") or {})
            .get("status")
            or ""
        )
        == "success"
        and _counter(device, "tts_url_deliveries") > deliveries_before_2,
        45.0,
        recorder=_record,
    )
    second_record = dict(
        (getattr(listener, "metrics", {}) or {}).get("last_satellite_segment") or {}
    )
    second_stream = second_record.get("stream") or ()
    turn_generation = (
        int(second_stream[2])
        if len(second_stream) == 3
        else int(device.session_generation)
    )
    proof22 = _chain_proof(device, turn_generation, SENTENCE)
    text22 = str(
        second_record.get("filtered_text") or second_record.get("raw_transcript") or ""
    )
    proof22["transcript"] = text22
    proof22["words_ok"] = _text_matches(text22)
    _report(
        22,
        "second consecutive turn completes the whole chain",
        int(
            (getattr(listener, "metrics", {}) or {}).get("stt_end_success") or 0
        )
        > success_before
        and str(second_record.get("status") or "") == "success"
        and _text_matches(text22)
        and proof22["order_ok"]
        and proof22["http_hits"] >= 1
        and proof22["http_status"] == 200
        and proof22["served_bytes"] == proof22["wav_stored_bytes"] > 0
        and proof22["last_finished_generation"] == int(turn_generation)
        and proof22["last_announce_success"] is True,
        f"generation={turn_generation} stream={list(second_stream)} "
        f"transcript='{text22}' words_ok={_text_matches(text22)} "
        f"avg_logprob={second_record.get('avg_logprob')} "
        f"query='{second_record.get('query')}' "
        f"reply_source={proof22['reply_source'] or '-'} "
        f"wav_bytes={proof22['wav_stored_bytes']} http={proof22['http_status']} "
        f"served={proof22['served_bytes']} type={proof22['content_type']} "
        f"finished_gen={proof22['last_finished_generation']} "
        f"latency={proof22['latency']} phase={proof22['phase']}",
    )
    turn_proofs.append(proof22)


    # 23. Late blocks of an older stream stay rejected at the frame gate, while
    #     the live stream still passes it.
    from jarvis.integrations.voice_pe.models import AudioFrame, StreamId

    stale_before = int(getattr(listener, "_stale_frames", 0) or 0)
    reference = listener._sink_context()
    if reference is None:
        print("   -> press the centre button (12 s) so a stream is open", flush=True)
        _open_run(12.0, 8.0)
        reference = listener._sink_context()
    open_stream = reference.stream if reference is not None else device._stream()
    old_streams = [
        # Same satellite, older connection.
        StreamId(open_stream.device_id, max(0, int(open_stream.connection_generation) - 1),
                 int(open_stream.session_generation)),
        # Same satellite and connection, older run.
        StreamId(open_stream.device_id, int(open_stream.connection_generation),
                 max(0, int(open_stream.session_generation) - 1)),
        # Another satellite with the same numbering.
        StreamId("other", int(open_stream.connection_generation),
                 int(open_stream.session_generation)),
    ]
    accepted = [listener._is_current_frame(s, "voice_pe") for s in old_streams]
    live_gate = listener._is_current_frame(open_stream, "voice_pe")
    # The bounded queue is drained first so the sentinel blocks fit.
    try:
        while not listener._audio_q.empty():
            listener._audio_q.get_nowait()
    except Exception:
        pass
    injected = 0
    for stream in old_streams:
        try:
            listener._audio_q.put_nowait(AudioFrame(stream, "voice_pe", b"\x01\x02"))
            injected += 1
        except Exception:
            break
    _poll(
        lambda: int(getattr(listener, "_stale_frames", 0) or 0)
        >= stale_before + injected,
        10.0,
    )
    _report(
        23,
        "frames of an older connection/session stream stay rejected",
        accepted == [False, False, False]
        and live_gate is True
        and injected == len(old_streams)
        and int(getattr(listener, "_stale_frames", 0) or 0)
        >= stale_before + injected,
        f"open={open_stream} stale_gate={accepted} "
        f"stale_frames {stale_before} -> {getattr(listener, '_stale_frames', 0)} "
        f"injected={injected} current_gate={live_gate}",
    )

    # 24/25. Barge-in: one press during an active playback closes that
    #        generation, and the next press opens a new one deterministically.
    traces_before = len(_button_trace(device))
    gen_barge = int(device.session_generation)
    # Whatever is open right now takes the barge-in; a fresh press opens one if
    # the previous checkpoint already closed its run.
    if not bool(device.holds_session()):
        print("   -> press the centre button for the barged-in run (20 s)", flush=True)
        _poll(
            lambda: int(device.session_generation) > gen_barge,
            20.0,
            recorder=_record,
        )
    barge_generation = int(device.session_generation)
    barge_open = bool(device.holds_session()) or barge_generation > 0
    finished_before_24 = _counter(device, "announcements_finished")
    if barge_open:
        print("   -> speak the sentence into the satellite (60 s)", flush=True)
        _poll(
            lambda: "TTS_START" in _ledger(device, barge_generation),
            hardware_s,
            recorder=_record,
        )
        finished_before_24 = _counter(device, "announcements_finished")
        print(
            "   -> press the centre button again while it is still speaking (40 s)",
            flush=True,
        )
        _poll(
            lambda: _counter(device, "announcements_finished") > finished_before_24
            or int(device.session_generation) > barge_generation,
            40.0,
            recorder=_record,
        )
    traces_after = _button_trace(device)
    new_after_barge = int(device.session_generation)
    _report(
        24,
        "one press during playback closes that generation",
        bool(barge_open)
        and (
            _counter(device, "announcements_finished") > finished_before_24
            or int(device.last_finished_generation) == int(barge_generation)
            or int(device.session_generation) > barge_generation
        ),
        f"barge_generation={barge_generation} "
        f"finished {_counter(device, 'announcements_finished')} "
        f"finished_gen={device.last_finished_generation} "
        f"now={int(device.session_generation)} state={device.session_state.value} "
        f"ledger={_ledger(device, barge_generation)}",
    )
    _report(
        25,
        "the press after a barge-in is traced and opens a new generation",
        len(traces_after) > traces_before
        and all(
            isinstance(row, dict) and "accepted" in row and "rejected_reason" in row
            for row in traces_after[: max(1, len(traces_after) - traces_before)]
        )
        and int(new_after_barge) >= int(barge_generation),
        f"presses traced {traces_before} -> {len(traces_after)} "
        f"newest={new_after_barge} last_button={traces_after[0] if traces_after else '-'}",
    )

    # 26. Every generation carries exactly one ERROR and one RUN_END. A second
    #     ``handle_pipeline_stop(abort=True)`` or a duplicate ``on_error`` is
    #     logged as an ``ignored_duplicate:<gen>`` and never widens the ledger.
    per_gen_terminals: dict = {}
    for gen, name in device.event_ledger:
        per_gen_terminals.setdefault(int(gen), []).append(name)
    single_terminal = all(
        int(led.count("ERROR")) <= 1 and int(led.count("RUN_END")) <= 1
        for led in per_gen_terminals.values()
    )
    _report(
        26,
        "one ERROR and one RUN_END per generation",
        single_terminal and bool(per_gen_terminals),
        " ".join(
            f"g{gen}=E{led.count('ERROR')}/R{led.count('RUN_END')}"
            for gen, led in sorted(per_gen_terminals.items())
        )
        or "empty",
    )

    # 27. Packet integrity on the very last clip: SHA over each 20 ms grid
    #     frame; adjacent equal non-zero frames are duplicated packets. The
    #     raw WAV is also read back to confirm the clip contains the same
    #     duration/rms the JSON reported, so 16 kHz mono is not an assumption
    #     baked only into a duration comparison.
    integrity_pairs: tuple = (0, 0, [])
    duration_ok = False
    rms_ok = False
    last_wav_path = None
    rec_last: dict = {}
    if prefix:
        wav_files = sorted(base.glob(f"{stem}-*.wav"), key=lambda p: len(p.name))
        if wav_files:
            last = wav_files[-1]
            last_wav_path = last.name
            hashes = _wav_frame_hashes(last.name, 20, 16000)
            integrity_pairs = _adjacent_duplicates(hashes)
            json_side = last.with_suffix(".json")
            if json_side.exists():
                import json as _json_lib

                with open(json_side, encoding="utf-8") as handle:
                    rec_last = _json_lib.load(handle)
                import wave as _wave

                with _wave.open(last.name, "rb") as handle:
                    meta = {
                        "framerate": handle.getframerate(),
                        "channels": handle.getnchannels(),
                        "sampwidth": handle.getsampwidth(),
                        "frames": handle.getnframes(),
                    }
                duration_ok = (
                    int(meta["framerate"]) == 16000
                    and int(meta["channels"]) == 1
                    and int(meta["sampwidth"]) == 2
                    and abs(int(meta["frames"]) / 16000.0 - float(rec_last.get("duration_s") or 0.0))
                    < 0.02
                )
                rms_ok = float(rec_last.get("rms") or 0.0) > 0.0 and int(rec_last.get("nonzero_samples") or rec_last.get("callback_count") or 0) > 0
    _report(
        27,
        "packet integrity on the last clip: adjacent SHAs, 16 kHz mono 2B, non-empty",
        integrity_pairs[1] > 0
        and duration_ok
        and rms_ok
        and integrity_pairs[0] <= max(0, int(integrity_pairs[1] * 0.10)),
        f"frames={integrity_pairs[1]} adjacent_duplicates={integrity_pairs[0]} "
        f"dur_ok={duration_ok} rms_ok={rms_ok} "
        f"first_pairs={integrity_pairs[2]}",
    )
    # The firmware's own audio contract, read straight from the API metadata.
    fw = _firmware_metadata(device)
    print(
        f"   audio contract from device: sr=16000 mono=1 width=16bit "
        f"esphome={fw.get('esphome_version') or '-'} "
        f"project={fw.get('project_name') or '-'} v{fw.get('project_version') or '-'} "
        f"api={fw.get('api_version') or '-'}",
        flush=True,
    )

    # 28. Heuristic verdicts read from the same WAV bytes, plus `manual` field
    #     that is set only when the CTO has listened to the WAV by hand.
    verdict = _acoustic_heuristic_verdict(last_wav_path, rec_last, integrity_pairs[1])
    human_verdict = _human_verdict_or_none(rec_last.get("human_listen_verdict"))
    print(
        f"   human_listen_verdict={"None" if human_verdict is None else human_verdict}",
        flush=True,
    )
    _report(
        28,
        f"acoustic heuristic verdict: {verdict}",
        # The heuristic field is never silently promoted to a human verdict.
        verdict.startswith(("acoustic_", "transcript_"))
        and human_verdict is None,
        f"last_clip={last_wav_path} clip_count={integrity_pairs[1]} "
        f"human={human_verdict or '-'}",
    )

    voice_pe.stop()
    listener.stop()
    failed = sum(1 for _i, _n, ok, _d in RESULTS if not ok)
    print(f"-- {len(RESULTS) - failed}/{len(RESULTS)} checkpoints passed", flush=True)
    # Two consecutive turns each have to fold the whole chain into one proof:
    # STT_END(success) -> INTENT_START -> TTS_START -> WAV HTTP 200 ->
    # served_bytes == wav_stored_bytes > 0 -> AnnounceFinished(same generation),
    # success -> TTS_END before the last RUN_END, and the content words.
    def _turn_ok(proof: dict) -> tuple[bool, str]:
        if not proof:
            return (False, "no proof")
        ordered = list(proof.get("ledger") or [])
        stt_end_at = next(
            (i for i, n in enumerate(ordered) if n == "STT_END"), -1
        )
        tts_end_at = next(
            (i for i, n in enumerate(ordered) if n == "TTS_END"), -1
        )
        run_end_at = next(
            (i for i, n in enumerate(ordered) if n == "RUN_END"), -1
        )
        checks = {
            "stt_end_present": stt_end_at >= 0 and bool(proof.get("stt_end_success")),
            "order_ok": bool(proof.get("order_ok")),
            "intent_before_tts": _is_subsequence(
                ["INTENT_START", "TTS_START"], ordered
            ),
            "tts_end_before_run_end": 0 <= tts_end_at < run_end_at,
            "http_200_served": (
                int(proof.get("http_hits", 0) or 0) >= 1
                and int(proof.get("http_status") or 0) == 200
                and int(proof.get("served_bytes", 0) or 0) > 0
                and int(proof.get("served_bytes", 0) or 0)
                == int(proof.get("wav_stored_bytes", 0) or 0)
            ),
            "announce_finished_gen": int(
                proof.get("last_finished_generation") or -1
            ) == int(proof.get("generation") or -2),
            "announce_success": proof.get("last_announce_success") is True,
            "words_ok": bool(proof.get("words_ok")),
        }
        ok = all(checks.values())
        detail = " ".join(f"{k}={'y' if v else 'N'}" for k, v in checks.items())
        return (ok, detail)

    for position, proof in enumerate(turn_proofs, start=1):
        ok_turn, detail = _turn_ok(proof)
        print(
            f"   turn {position}: generation={proof.get('generation')} "
            f"transcript='{proof.get('transcript')}' "
            f"words_ok={proof.get('words_ok')} "
            f"reply_source={proof.get('reply_source') or '-'} "
            f"order={proof.get('order_ok')} wav_bytes={proof.get('wav_stored_bytes')} "
            f"http={proof.get('http_status')} served={proof.get('served_bytes')} "
            f"type={proof.get('content_type')} "
            f"finished_gen={proof.get('last_finished_generation')} "
            f"announce_success={proof.get('last_announce_success')} "
            f"latency={proof.get('latency')} phase={proof.get('phase')} "
            f"chain={detail}",
            flush=True,
        )
        if not ok_turn:
            failed += 1
    # Announce both proofs at the same time so the CTO sees the correlated pair.
    if len(turn_proofs) >= 2:
        both_ok = all(_turn_ok(p)[0] for p in turn_proofs[:2])
        print(
            f"   consecutive-pair: both={'y' if both_ok else 'N'} "
            f"first={_turn_ok(turn_proofs[0])[1]} "
            f"second={_turn_ok(turn_proofs[1])[1]}",
            flush=True,
        )
        if not both_ok:
            failed += 1
    if failed:
        print(
            f"   note: 9 to 16 and 22 to 25 need the spoken sentence "
            f"'{SENTENCE}' while the run is open; the rest is protocol state.",
            flush=True,
        )
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
