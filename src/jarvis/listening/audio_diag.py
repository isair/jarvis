"""One-shot local audio diagnostics (JARVIS_AUDIO_DIAG=1 only).

Writes raw / pre-OpenAI / 24 kHz WAV snapshots plus a minimal manifest.
Never logs secrets, transcripts, or API material. No-op when the env flag
is unset or after the first capture completes.
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np


def audio_diag_enabled() -> bool:
    return os.environ.get("JARVIS_AUDIO_DIAG", "").strip() == "1"


def _peak_rms(audio: np.ndarray) -> tuple[float, float]:
    if audio is None or getattr(audio, "size", 0) == 0:
        return 0.0, 0.0
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return peak, rms


def write_wav_float32_mono(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Write float32 mono [-1,1] as PCM16 WAV. Returns manifest fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    peak, rms = _peak_rms(x)
    return {
        "sample_rate": int(sample_rate),
        "samples": int(x.size),
        "channels": 1,
        "duration": float(x.size / max(int(sample_rate), 1)),
        "peak": peak,
        "rms": rms,
    }


def write_wav_pcm16_mono(path: Path, pcm16: bytes, sample_rate: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.frombuffer(pcm16, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16)
    # peak/rms in float space
    if samples.size:
        xf = samples.astype(np.float32) / 32767.0
        peak, rms = _peak_rms(xf)
    else:
        peak, rms = 0.0, 0.0
    return {
        "sample_rate": int(sample_rate),
        "samples": int(samples.size),
        "channels": 1,
        "duration": float(samples.size / max(int(sample_rate), 1)),
        "peak": peak,
        "rms": rms,
    }


def save_one_shot_capture(
    *,
    out_dir: Path,
    raw_audio: np.ndarray,
    raw_rate: int,
    pre_openai_audio: np.ndarray,
    pre_openai_rate: int,
    openai_pcm16_24k: bytes,
    dropped_frames: int,
) -> Optional[Path]:
    """Write the three WAVs + manifest.json. Returns manifest path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_m = write_wav_float32_mono(out_dir / "raw_capture.wav", raw_audio, raw_rate)
    pre_m = write_wav_float32_mono(out_dir / "pre_openai.wav", pre_openai_audio, pre_openai_rate)
    oai_m = write_wav_pcm16_mono(out_dir / "openai_24k.wav", openai_pcm16_24k, 24000)

    # Per-stage fragments only — no transcripts, keys, or extra fields.
    def _frag(m: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_rate": m["sample_rate"],
            "samples": m["samples"],
            "channels": m["channels"],
            "duration": m["duration"],
            "peak": m["peak"],
            "rms": m["rms"],
            "dropped_frames": int(dropped_frames),
        }

    manifest = {
        "raw_capture": _frag(raw_m),
        "pre_openai": _frag(pre_m),
        "openai_24k": _frag(oai_m),
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return man_path
