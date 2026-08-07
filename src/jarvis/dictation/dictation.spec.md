# Dictation Engine Specification

## Overview

WisprFlow-like dictation: hold a hotkey to record speech, release to type the
transcription into the focused application. Completely independent from the
assistant pipeline (no wake words, intent judge, profiles, or TTS).

## Configuration

| Key                            | Type   | Default (per-platform)                         | Description                                     |
|--------------------------------|--------|------------------------------------------------|-------------------------------------------------|
| `dictation_enabled`           | bool   | `true`                                         | Master switch for the feature                   |
| `dictation_hotkey`            | string | Win: `"ctrl+cmd"`, macOS/Linux: `"ctrl+alt"`   | Hold-to-record hotkey combination               |
| `dictation_filler_removal`    | bool   | `false`                                        | LLM-based filler word removal via Ollama        |
| `dictation_custom_dictionary` | list   | `[]`                                           | Custom replacements in `"wrong -> right"` format|

Defaults are aligned with WisprFlow. Modifier-only combos are supported
(e.g. `"ctrl+cmd"` activates when both keys are held, with no extra trigger
key required).

The hotkey is configurable as a dropdown in both the setup wizard and settings
window, with four preset options: `ctrl+alt`, `ctrl+cmd`, `ctrl+shift+d`,
`ctrl+shift`.

## Core Flow

### Hold-to-Dictate (Standard Mode)

1. **Press hotkey** → start recording audio into buffer, play start beep,
   set face to `DICTATING`, pause main voice listener.
2. **Hold hotkey** → audio frames accumulate in a dedicated
   `sounddevice.InputStream`.
3. **Release hotkey** → stop recording, play stop beep, set face to
   `DICTATION_PROCESSING`, transcribe via the shared SenseVoice engine, apply
   post-processing pipeline, paste result into focused app via clipboard,
   restore face to `IDLE`, resume main voice listener.

The face therefore moves through three distinct states across a dictation
cycle: `DICTATING` while recording, `DICTATION_PROCESSING` while the captured
audio is being transcribed / post-processed / pasted, and back to `IDLE` once
the cycle completes. This gives the user visual confirmation that their voice
input has been accepted and is being processed.

### Hands-Free Mode (Double-Tap)

1. **Quick press-and-release** (hold < 0.4 s) followed by a **second tap**
   within 0.4 s → enters hands-free mode. Recording continues until
   explicitly stopped.
2. **Stop triggers** — re-press the hotkey *or* press Escape.
3. Same post-processing pipeline as standard mode.

## Post-Processing Pipeline

After transcription, text passes through these stages in order:

1. **Custom dictionary** — case-insensitive whole-word regex replacements
   from `dictation_custom_dictionary`. Each entry is `"wrong -> right"`.
2. **LLM filler removal** (optional) — when `dictation_filler_removal` is
   enabled, sends the text to the local Ollama instance (same model as the
   assistant) with a prompt to remove filler words (um, uh, like, you know,
   etc.) while preserving meaning. Uses a 5-second timeout; falls back to the
   unprocessed text on failure.

## Architecture

- **`pynput`** for global hotkey detection (cross-platform).
- **Clipboard-based paste** (`Ctrl+V` / `Cmd+V`) for text insertion — more
  reliable than character-by-character typing, handles Unicode.
- **Shared SenseVoice engine** via lazy reference (`lambda: voice_thread.engine`)
  and backend info — no double memory usage.
- **Separate `sounddevice.InputStream`** for dictation audio — avoids
  modifying the complex listener code.
- **Pause flag** on the main listener to prevent dictation speech being
  interpreted as commands.
- **No PortAudio work on the pynput hook thread.** The hotkey callbacks only
  flip state; stream open/start runs on a worker thread (`_begin_recording`)
  and stop/transcribe already ran on one. Windows silently unhooks slow hook
  callbacks, and opening streams there raced other audio threads into a
  native abort (#462). If recording is stopped before the worker finishes
  opening the stream, the worker closes it rather than leaking a live capture.
- **All PortAudio stream lifecycle calls** (open/start/stop/close, and
  `sd.play` for beeps) run under the process-wide
  `jarvis.utils.audio_lock.portaudio_lock` — PortAudio documents stream
  open/close as not thread safe, and unserialised calls abort the whole app
  on Windows.

### Optional Dependencies

- `sounddevice`, `numpy`, and `pynput` are optional at import time: the
  module must always import cleanly even when they are broken or missing
  (e.g. `import sounddevice` raises **OSError**, not ImportError, when the
  PortAudio shared library is absent; `pynput` can raise display errors on
  headless Linux). Importers such as the setup wizard rely on this — a
  missing audio backend must never crash the app.

### Audio Device Handling

- The engine accepts an optional `voice_device` parameter, passed through from
  the daemon's configured device.
- The stream first attempts the target SenseVoice sample rate (16 kHz).
- On failure (e.g. PortAudio error -50 on macOS), it falls back to the
  device's native sample rate and stores it in `_stream_sample_rate`.
- If the stream rate differs from the SenseVoice target rate, audio is resampled
  via linear interpolation before transcription.

## Edge Cases

| Case                      | Behaviour                                         |
|---------------------------|----------------------------------------------------|
| SenseVoice engine not loaded | Play "not ready" beep, skip                        |
| Max recording duration    | No cap — the user controls when to stop by releasing the hotkey. A cap would paste prematurely mid-dictation and restart recording |
| Empty transcription       | No paste occurs                                    |
| Concurrent with assistant | Dictation works independently; pauses listener     |
| macOS permissions         | `pynput` requires Accessibility permissions        |
| macOS 26+ (Tahoe)         | `pynput` disabled — TSM main-thread assertion crash |
| Linux / Wayland           | `pynput` requires X11 (limited Wayland support)    |
| Audio rate mismatch       | Resample via linear interpolation to SenseVoice rate  |
| LLM filler removal fails  | Falls back to raw transcription (5 s timeout)     |
| Custom dictionary empty   | No-op, text passes through unchanged               |

## Thread Safety

- `threading.Lock` around shared SenseVoice engine transcription calls.
- Dedicated audio stream; never touches the listener's stream.
- The `pynput` key handlers (`_on_key_press` / `_on_key_release`) must return
  quickly — Windows silently removes low-level keyboard hooks that take more
  than ~5 s to return, leaving pynput in an inconsistent state that can
  segfault on the next `Controller` call from the paste thread. `_stop_recording`
  therefore only flips state under the lock and dispatches audio-stream
  teardown, beep playback, transcription, and paste to a background thread.
  The `discard=True` path keeps the synchronous teardown so shutdown can
  reliably wait for everything to finish.

## Beeps

Two short beeps generated the same way as the existing `TunePlayer` sonar ping:
- **Start beep** — higher pitch (700 Hz), signals recording started.
- **Stop beep** — lower pitch (440 Hz), signals recording stopped.

## Setup Wizard

The setup wizard includes a dedicated Dictation page (between speech recognition and
Location steps) that allows users to:
- Enable/disable dictation
- Choose the hotkey from a dropdown of presets
- View tips about hold-to-dictate and double-tap hands-free mode

## Dependencies

- `pynput>=1.7.6` — global hotkey detection and keyboard simulation.
