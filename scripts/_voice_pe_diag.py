"""Offline reading of the listener's clip diagnostics.

The listener dumps, per utterance, the exact post-resample PCM it handed to
Whisper plus the raw segment values (``JARVIS_VOICE_DIAG_WAV=<prefix>``). This
script re-reads those files and re-decodes the same WAV through the same model
without the pipeline's confidence filter, so a dropped row can be told apart
from a missing one.

    python scripts/_voice_pe_diag.py <prefix>            # e.g. C:\\tmp\\peclip
    python scripts/_voice_pe_diag.py <prefix> --model large-v3-turbo

No Home Assistant server, no mocks, no writes besides stdout.
"""

from __future__ import annotations

import json
import math
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _stats(pcm: bytes, rate: int) -> dict:
    import numpy as np

    data = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if data.size == 0:
        return {"samples": 0}
    return {
        "samples": int(data.size),
        "duration_s": round(data.size / float(rate), 4),
        "rms": round(float(np.sqrt(np.mean(np.square(data)))), 8),
        "peak": round(float(np.max(np.abs(data))), 8),
        "dc_offset": round(float(np.mean(data)), 8),
        "clipped_ratio": round(
            float(np.count_nonzero(np.abs(data) >= 0.999)) / int(data.size), 6
        ),
        "nonzero_ratio": round(
            float(np.count_nonzero(data)) / int(data.size), 6
        ),
    }


def _decode(wav_path: str, model_name: str, language: str | None):
    """Same model, same clip, without the pipeline's own confidence filter.

    Two variants are reported: the kwargs the pipeline intends to pass, and the
    kwargs it effectively passes when the first call raises ``TypeError`` (the
    listener re-calls with ``language`` only, and the defaults differ).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    keyed: dict = {}
    if language:
        keyed["language"] = language
    variants = {
        "intended": dict(
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            **keyed,
        ),
        "effective": dict(keyed),
        # The literal call the listener writes, recorded to show which keyword
        # the installed library rejects and the call collapses to ``effective``.
        "listener_call": dict(
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            suppress_nospeech_text=True,
            **keyed,
        ),
    }
    out: dict = {}
    for label, kwargs in variants.items():
        try:
            segments, info = model.transcribe(wav_path, **kwargs)
        except Exception as err:
            out[label] = {"error": str(err)}
            continue
        rows = [
            {
                "text": (seg.text or "").strip(),
                "avg_logprob": round(float(seg.avg_logprob), 4),
                "no_speech_prob": round(float(seg.no_speech_prob), 4),
                "confidence": round(
                    min(1.0, max(0.0, math.exp(float(seg.avg_logprob)))), 4
                ),
            }
            for seg in segments
        ]
        out[label] = {
            "rows": rows,
            "detected": str(getattr(info, "language", "") or ""),
        }
    return out


def _matrix(wav_path: str, model_name: str, language: str | None) -> dict:
    """Same bytes decoded under each plausible sample-rate / endianness.

    The listener writes PCM16LE at 16000 Hz mono. The matrix reads what the
    decoder would see under 8k/16k/22.05k/44.1k/48k and under both endianness
    tags, so a wrong rate or a swapped byte-order becomes visible as a
    duration + text change against the intended cell, and only that cell.
    """
    with wave.open(wav_path, "rb") as handle:
        data = handle.readframes(handle.getnframes())
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    keyed: dict = {}
    if language:
        keyed["language"] = language
    rates = (8000, 16000, 22050, 44100, 48000)
    variants: dict = {}

    def _one(dtype: str, rate: int):
        import numpy as np

        raw = np.frombuffer(data, dtype=dtype).astype(np.float64)
        vals = raw / 32768.0
        with wave.open(wav_path + ".m.wav", "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(
                (np.clip(vals, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            )
        rms = float(np.sqrt(np.mean(np.square(vals)))) if vals.size else 0.0
        try:
            segs, _info = model.transcribe(
                wav_path + ".m.wav",
                vad_filter=False,
                condition_on_previous_text=False,
                **keyed,
            )
            rows = [
                {
                    "text": (s.text or "").strip(),
                    "avg_logprob": round(float(s.avg_logprob), 4),
                    "no_speech_prob": round(float(s.no_speech_prob), 4),
                    "confidence": round(
                        min(1.0, max(0.0, math.exp(float(s.avg_logprob)))), 4
                    ),
                }
                for s in segs
            ]
        except Exception as err:
            rows = [{"error": str(err)}]
        with wave.open(wav_path + ".m.wav", "rb") as handle:
            n = int(handle.getnframes())
        return {
            "rms": round(rms, 8),
            "duration_s": round(n / float(rate), 4),
            "rows": rows,
        }

    for rate in rates:
        variants[f"LE@{rate}"] = _one("<i2", rate)
    variants[">i2@16000"] = _one(">i2", 16000)
    return variants


def _listen(prefix: str, seconds: float) -> int:
    """Boot the local stack and let the microphone fill the diagnostic dumps."""
    from jarvis.config import load_settings
    from jarvis.listening.listener import VoiceListener
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.output.tts import create_tts_engine

    settings = load_settings()
    db = Database(settings.db_path, settings.sqlite_vss_path)
    tts = create_tts_engine(
        engine=settings.tts_engine,
        enabled=False,
        voice=settings.tts_voice,
        rate=settings.tts_rate,
    )
    memory = DialogueMemory(
        inactivity_timeout=settings.dialogue_memory_timeout, max_interactions=20
    )
    listener = VoiceListener(db, settings, tts, memory)
    listener.start()
    print(f"-> say the sentence into the PC microphone ({seconds:.0f} s)", flush=True)
    time.sleep(max(1.0, seconds))
    print(
        f"   callbacks={getattr(listener, '_callback_count', 0)} "
        f"queue={listener._audio_q.qsize()}",
        flush=True,
    )
    listener.stop()
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "--listen":
        prefix = args[1] if len(args) > 1 else "local"
        import os

        os.environ["JARVIS_VOICE_DIAG_WAV"] = prefix
        seconds = float(args[2]) if len(args) > 2 else 40.0
        _listen(prefix, seconds)
        return _read(prefix, None)
    if not args:
        print("usage: python scripts/_voice_pe_diag.py <prefix> [--model NAME]")
        print("       python scripts/_voice_pe_diag.py --listen <prefix> [seconds]")
        return 2
    model_override = None
    if "--model" in args:
        model_override = args[args.index("--model") + 1]
    return _read(args[0], model_override, show_matrix="--matrix" in args)


def _read(prefix: str, model_override: str | None, show_matrix: bool = False) -> int:

    base = Path(prefix).parent if Path(prefix).is_absolute() else Path.cwd()
    stem = Path(prefix).name
    indexes = sorted(
        {int(p.stem.rsplit("-", 1)[-1]) for p in base.glob(f"{stem}-*.json")}
    )
    if not indexes:
        print(f"no {stem}-N.json next to {base}")
        return 2

    for index in indexes:
        json_path = f"{prefix}-{index}.json"
        wav_path = f"{prefix}-{index}.wav"
        with open(json_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
        with wave.open(wav_path, "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            pcm = handle.readframes(handle.getnframes())
        print(f"\n=== clip {index} ({recorded.get('note', '')}) ===")
        print(f"file          : {wav_path}")
        print(
            "format        : "
            f"rate={rate} channels={channels} sample_width={width} "
            f"(recorded rate={recorded.get('sample_rate')} "
            f"channels={recorded.get('channels')} "
            f"width={recorded.get('sample_width_bytes')})"
        )
        measured = _stats(pcm, rate)
        for key in (
            "samples",
            "duration_s",
            "rms",
            "peak",
            "dc_offset",
            "clipped_ratio",
            "nonzero_ratio",
        ):
            print(
                f"{key:<14}: {measured.get(key)}"
                f"   (recorded {recorded.get(key)})"
            )
        print(
            "vad window    : "
            f"start={recorded.get('vad_start_epoch')} "
            f"end={recorded.get('vad_end_epoch')} "
            f"span={recorded.get('vad_span_s')}s"
        )
        print(
            "padding       : "
            f"{recorded.get('trailing_padding_frames')} frames = "
            f"{recorded.get('trailing_padding_ms')} ms "
            f"(endpoint_silence_ms={recorded.get('endpoint_silence_ms')})"
        )
        print(
            "identity      : "
            f"source={recorded.get('source')} stream={recorded.get('stream')} "
            f"turn={recorded.get('turn_context')}"
        )
        # Audio format source and device identity, straight from the dump.
        info_fields = recorded.get("device_info_fields") or {}
        if info_fields:
            print(
                "device_info   : "
                + json.dumps(info_fields, sort_keys=True)
            )
        src = recorded.get("audio_format_source") or "-"
        print(
            "audio_format  : "
            f"source={src} "
            + (
                f"wav={rate}Hz/{channels}ch/{width}B"
                if src == "configured_wav_header"
                else f"assumed={recorded.get('sample_rate')}Hz/"
                f"{recorded.get('channels')}ch/{recorded.get('sample_width_bytes')}B"
            )
        )
        ev = (recorded.get("speech_evidence")
              or dict(recorded.get("preprocessor") or {}).get("speech_evidence")
              or {})
        if ev:
            print(
                "speech_evidence : "
                + json.dumps(ev, sort_keys=True)
            )
        print(
            "language      : "
            f"config={recorded.get('whisper_language') or 'auto'} "
            f"last_detected={recorded.get('last_detected_language')} "
            f"model={recorded.get('whisper_model')}"
        )
        print(
            "thresholds    : "
            f"min_audio={recorded.get('min_audio_duration_s')} "
            f"min_confidence={recorded.get('min_confidence')} "
            f"no_speech={recorded.get('no_speech_threshold')}"
        )
        print(
            "device counters: "
            + json.dumps(
                recorded.get("device_metrics") or recorded.get("metrics") or {},
                sort_keys=True,
            )
        )
        print(
            "listener       : "
            f"callback_count={recorded.get('callback_count')} "
            f"queue={recorded.get('queue_size')}"
        )
        print(
            "voiced grid    : "
            f"frames={ (recorded.get('voiced_last_frame') or -1) - (recorded.get('voiced_first_frame') or 0) + 1 if recorded.get('voiced_first_frame') is not None else 0} "
            f"first={recorded.get('voiced_first_frame_ms')} ms "
            f"last={recorded.get('voiced_last_frame_ms')} ms"
        )
        print(
            "decoder opts   : "
            f"backend={recorded.get('asr_backend')!r} "
            f"version={recorded.get('asr_version') or '-'} "
            f"kwargs={recorded.get('transcribe_kwargs')}"
        )
        print(
            "frame state    : "
            + json.dumps(recorded.get("frame_state") or {}, sort_keys=True)
            + f" speech_span={recorded.get('speech_span_s')}s"
        )
        # Levels of the decoder input: both representations, dBFS, the SNR
        # against the padded silence, and the preprocessor settings applied.
        levels = recorded.get("audio_level") or {}
        if levels:
            print(
                "audio float32  : "
                f"min={levels.get('min')} max={levels.get('max')} "
                f"rms={levels.get('rms')} peak={levels.get('peak')}"
            )
            print(
                "audio int16    : "
                + json.dumps(levels.get("int16") or {}, sort_keys=True)
                + f" scale_ratio={levels.get('scale_ratio_int16_over_float32')}"
            )
            print(
                "audio raw      : "
                f"float32={json.dumps(levels.get('raw_float32') or {}, sort_keys=True)} "
                f"int16={json.dumps(levels.get('raw_int16') or {}, sort_keys=True)} "
                f"scale_ratio={levels.get('raw_scale_ratio_int16_over_float32')}"
            )
            print(
                "dBFS           : "
                f"rms={levels.get('dbfs_rms')} peak={levels.get('dbfs_peak')} "
                f"rms_raw={levels.get('dbfs_rms_raw')} "
                f"peak_raw={levels.get('dbfs_peak_raw')}"
            )
            print(
                "silence/snr    : "
                f"leading={levels.get('leading_silence_frames')} "
                f"trailing={levels.get('trailing_silence_frames')} "
                f"silent_frames={levels.get('silent_frames')} "
                f"voiced_rms={levels.get('voiced_rms')} "
                f"silent_rms={levels.get('silent_rms')} snr_db={levels.get('snr_db')}"
            )
        print(
            "preprocessor   : "
            + json.dumps(recorded.get("preprocessor") or {}, sort_keys=True)
        )
        print("in-loop raw   :")
        for row in recorded.get("raw_segments", []):
            print(
                "   "
                f"avg_logprob={row.get('avg_logprob')} "
                f"no_speech_prob={row.get('no_speech_prob')} "
                f"text={row.get('text')!r}"
            )
        print(f"in-loop kept  : {recorded.get('filtered_text')!r}")
        model_name = model_override or recorded.get("whisper_model") or "small"
        try:
            decoded = _decode(
                wav_path, model_name, recorded.get("whisper_language") or None
            )
        except Exception as err:  # pragma: no cover - reported, not fatal
            print(f"offline decode failed: {err}")
            continue
        for label in ("intended", "effective", "listener_call"):
            block = decoded.get(label, {})
            if "error" in block:
                print(f"offline {label:<9}: error {block['error']}")
                continue
            print(
                f"offline {label:<9}: detected={block.get('detected')!r} "
                f"rows={len(block.get('rows', []))}"
            )
            for row in block.get("rows", []):
                print(
                    "   "
                    f"avg_logprob={row['avg_logprob']} "
                    f"no_speech_prob={row['no_speech_prob']} "
                    f"confidence={row['confidence']} text={row['text']!r}"
                )
        # Sample-rate / endianness matrix on the very same bytes, so a wrong
        # header in the pipeline shows up as a different rms + duration cell.
        if show_matrix or model_override == "__matrix__":
            try:
                matrix = _matrix(wav_path, model_name, recorded.get("whisper_language") or None)
            except Exception as err:  # pragma: no cover
                print(f"matrix failed: {err}")
            else:
                for label, block in matrix.items():
                    print(
                        f"matrix {label:<10}: rms={block['rms']} "
                        f"dur={block['duration_s']}s rows={len(block['rows'])}"
                    )
                    for row in block["rows"]:
                        if "error" in row:
                            print(f"   1. error={row['error']}")
                            continue
                        print(
                            "   "
                            f"avg_logprob={row['avg_logprob']} "
                            f"no_speech_prob={row['no_speech_prob']} "
                            f"confidence={row['confidence']} text={row['text']!r}"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
