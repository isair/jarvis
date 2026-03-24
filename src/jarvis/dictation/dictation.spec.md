# Dictation Engine Specification

## Overview

WisprFlow-like dictation: hold a hotkey to record speech, release to type the
transcription into the focused application. Completely independent from the
assistant pipeline (no wake words, intent judge, profiles, or TTS).

## Configuration

| Key                | Type   | Default           | Description                        |
|--------------------|--------|-------------------|------------------------------------|
| `dictation_enabled`| bool   | `true`            | Master switch for the feature      |
| `dictation_hotkey` | string | `"ctrl+shift+d"`  | Hold-to-record hotkey combination  |

## Core Flow

1. **Press hotkey** → start recording audio into buffer, play start beep,
   set face to `DICTATING`, pause main voice listener.
2. **Hold hotkey** → audio frames accumulate in a dedicated
   `sounddevice.InputStream`.
3. **Release hotkey** → stop recording, play stop beep, transcribe via shared
   Whisper model, paste result into focused app via clipboard, restore face
   to `IDLE`, resume main voice listener.

## Architecture

- **`pynput`** for global hotkey detection (cross-platform).
- **Clipboard-based paste** (`Ctrl+V` / `Cmd+V`) for text insertion — more
  reliable than character-by-character typing, handles Unicode.
- **Shared Whisper model** via lazy reference (`lambda: voice_thread.model`)
  and backend info — no double memory usage.
- **Separate `sounddevice.InputStream`** for dictation audio — avoids
  modifying the complex listener code.
- **Pause flag** on the main listener to prevent dictation speech being
  interpreted as commands.

## Edge Cases

| Case                      | Behaviour                                         |
|---------------------------|----------------------------------------------------|
| Whisper not yet loaded    | Play "not ready" beep, skip                        |
| Max recording duration    | 60 s cap to prevent memory exhaustion              |
| Empty transcription       | No paste occurs                                    |
| Concurrent with assistant | Dictation works independently; pauses listener     |
| macOS permissions         | `pynput` requires Accessibility permissions        |
| Linux / Wayland           | `pynput` requires X11 (limited Wayland support)    |

## Thread Safety

- `threading.Lock` around shared Whisper model transcription calls.
- Dedicated audio stream; never touches the listener's stream.

## Beeps

Two short beeps generated the same way as the existing `TunePlayer` sonar ping:
- **Start beep** — higher pitch (700 Hz), signals recording started.
- **Stop beep** — lower pitch (440 Hz), signals recording stopped.

## Dependencies

- `pynput>=1.7.6` — global hotkey detection and keyboard simulation.
