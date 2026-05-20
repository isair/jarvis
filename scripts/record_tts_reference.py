#!/usr/bin/env python3
"""Record Chatterbox TTS voice reference to ~/.config/jarvis/voices/reference.wav."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _default_out() -> Path:
    return Path.home() / ".config" / "jarvis" / "voices" / "reference.wav"


def _list_input_devices() -> None:
    import sounddevice as sd

    print("\n  Available microphones:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            default = " (default)" if idx == sd.default.device[0] else ""
            print(f"    [{idx}] {dev['name']}{default}")
    print()


def _record(seconds: float, samplerate: int, device: int | None) -> "object":
    import numpy as np
    import sounddevice as sd

    frames = int(seconds * samplerate)
    print(f"\n  Recording {seconds:.0f}s at {samplerate} Hz — speak now!\n")
    for i in range(3, 0, -1):
        print(f"    {i}…", flush=True)
        time.sleep(1)
    print("    GO\n", flush=True)
    audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32", device=device)
    sd.wait()
    peak = float(abs(audio).max()) if len(audio) else 0.0
    if peak < 0.01:
        print("  Warning: very quiet recording — check microphone level.", flush=True)
    return audio


def _update_config(wav_path: Path) -> None:
    cfg_path = Path.home() / ".config" / "jarvis" / "config.json"
    if not cfg_path.is_file():
        return
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        data["tts_chatterbox_audio_prompt"] = str(wav_path).replace("\\", "/")
        if data.get("tts_engine") != "chatterbox":
            data["tts_engine"] = "chatterbox"
        cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("  Updated config.json → tts_chatterbox_audio_prompt", flush=True)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Could not update config.json: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Chatterbox voice reference WAV")
    parser.add_argument("-o", "--output", type=Path, default=_default_out())
    parser.add_argument("-d", "--duration", type=float, default=8.0)
    parser.add_argument("-r", "--rate", type=int, default=24000)
    parser.add_argument("--device", type=int, default=None, help="sounddevice input index")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        _list_input_devices()
        return 0

    try:
        import soundfile as sf
    except ImportError:
        print("  soundfile not installed — run: pip install soundfile", flush=True)
        return 1

    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\n  Chatterbox voice reference recorder")
    print("  ─────────────────────────────────")
    print("  Tips:")
    print("    • Quiet room, mic 15–30 cm away")
    print("    • Speak naturally in the tone you want Jarvis to use")
    print("    • Example (English): \"Good morning. I am ready to assist you today.\"")
    print("    • Example (LV): \"Labrīt. Esmu gatavs palīdzēt.\"")
    _list_input_devices()

    try:
        input("  Press Enter when ready to record… ")
    except EOFError:
        print("  Non-interactive — starting in 2s…", flush=True)
        time.sleep(2)

    try:
        audio = _record(args.duration, args.rate, args.device)
        sf.write(str(out), audio, args.rate, subtype="PCM_16")
    except Exception as exc:
        print(f"\n  Recording failed: {exc}", flush=True)
        return 1

    print(f"\n  Saved: {out}")
    print(f"  Size: {out.stat().st_size // 1024} KB")
    _update_config(out)
    print("\n  Next: tray → Stop listening → Start listening (reload TTS).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
