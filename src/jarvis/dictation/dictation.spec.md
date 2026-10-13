# Dictation Engine Specification

## Overview

WisprFlow-like dictation: hold a hotkey to record speech, release to type the
transcription into the focused application. Completely independent from the
assistant pipeline (no wake words, intent judge, profiles, or TTS).

Dictation results persist in `~/.local/share/jarvis/dictation_history.json`.
The default is resolved through the lightweight `paths.py` boundary. An explicit
history path overrides the default. The desktop watcher observes the store's
selected `path`, including explicit overrides. While visible, it refreshes on
file addition, removal or replacement, including changes with older timestamps.
History save failures preserve existing records and report an activity-log warning
even when debug logging is disabled. An unsaved result is not published through
the history result callback. Text delivery and session completion remain separate
from persistence; subsequent successful saves publish their recorded results.

## Configuration

| Key                            | Type   | Default (per-platform)                         | Description                                     |
|--------------------------------|--------|------------------------------------------------|-------------------------------------------------|
| `dictation_enabled`           | bool   | `true`                                         | Master switch for the feature                   |
| `dictation_hotkey`            | string | Win: `"ctrl+cmd"`, macOS/Linux: `"ctrl+alt"`   | Hold-to-record hotkey combination               |
| `dictation_filler_removal`    | bool   | `false`                                        | LLM-based filler word removal via the configured chat backend        |
| `dictation_custom_dictionary` | list   | `[]`                                           | Custom replacements in `"wrong -> right"` format|

Defaults are aligned with WisprFlow. Modifier-only combos are supported
(e.g. `"ctrl+cmd"` activates when both keys are held, with no extra trigger
key required).

Only the configured trigger and modifiers control capture. Unrelated character
or virtual-key events cannot activate or release a character-key shortcut.

The hotkey is configurable as a dropdown in both the setup wizard and settings
window, with four preset options: `ctrl+alt`, `ctrl+cmd`, `ctrl+shift+d`,
`ctrl+shift`.

## Core Flow

### Hold-to-Dictate (Standard Mode)

1. **Press hotkey** → start recording audio into buffer, play start beep,
   set face to `DICTATING`, pause main voice listener.
2. **Hold hotkey** → audio frames accumulate in a dedicated
   `sounddevice.InputStream`. Each stream callback retains its recording-session token. Frames from released or superseded sessions are discarded, including when release occurs during audio conversion. Frame acceptance and release snapshots share the engine lock.
3. **Release hotkey** → stop recording, play stop beep, set face to
   `DICTATION_PROCESSING`, transcribe via shared Whisper model, apply
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
   enabled, sends the text to the configured local chat backend (same model as the
   assistant) with a prompt to remove filler words (um, uh, like, you know,
   etc.) while preserving meaning. Uses a 5-second timeout; falls back to the
   unprocessed text on failure, empty output or explicitly incomplete generation.
   `dictation_thinking_enabled` independently controls reasoning for this request and defaults to disabled.

## Architecture

- **`pynput`** for global hotkey detection (cross-platform). On supported macOS releases, the listener enters the Quartz event loop directly and retains pynput character/modifier decoding. It does not open a Carbon keyboard-layout context on its worker thread: event characters come from `CGEventKeyboardGetUnicodeString`, and the unused context can trigger a fatal TSM main-queue assertion. The macOS 26+ safety guard remains in force.
- **Clipboard-based paste** (`Ctrl+V` / `Cmd+V`) for text insertion — more
  reliable than character-by-character typing, handles Unicode. macOS uses
  native CGEvent paste exclusively, with both key events prepared before input
  is posted and both references released after the attempt. Accessibility is
  checked for every paste and an unavailable permission check denies automatic
  input. A denied permission opens Settings at most once per process. Failed
  native paste retains the copied text, reports manual Cmd+V as recovery and
  never enters a pynput Controller or Carbon keyboard-layout context.
  On Windows and Linux, automatic paste attempts release the shortcut modifier
  even when pressing or pasting fails. An unavailable keyboard backend or failed
  controller/paste operation reports manual Ctrl+V recovery after a successful
  clipboard copy. A failed clipboard write reports a visible warning without
  exposing exception contents and prevents automatic paste. History storage and
  the saved-result callback remain independent of clipboard delivery.
  Partial operations use conditional recovery guidance rather
  than claiming that no text was inserted. Error diagnostics omit raw exception
  messages.
- **Shared Whisper model** via lazy reference (`lambda: voice_thread.model`)
  and backend info — no double memory usage.
- **Separate `sounddevice.InputStream`** for dictation audio — avoids
  modifying the complex listener code.
- **Independent capture pause reason** on the main listener to prevent dictation
  speech being interpreted as commands. Dictation completion releases only its
  own reason and cannot resume a user-paused assistant. Start, processing and
  completion notifications belong to their recording session and are serialised
  with session transitions. Stopping the engine invalidates lifecycle ownership,
  releases its capture pause and rejects queued hotkey presses until restart.
  Released audio can still finish transcription, paste and history storage without
  further lifecycle notifications. Superseded workers cannot change the face or
  capture pause state. Finalisation reports completion to its owning session even
  on teardown failure.
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
  the daemon's configured device. Numeric indices and input-device names select
  that device. Unicode case-insensitive exact input-name matches take precedence
  over the first partial input-name match; both skip output-only devices. A
  missing named input fails instead of recording another device.
  The default input is resolved once per dictation session. An unavailable
  default produces Settings guidance and ends the session without recording.
- The stream tries the selected device's native sample rate and mono input first.
  Unsupported formats trigger bounded channel-count and rate retries on the
  same input, including the Whisper target rate. Access and unavailable-device
  errors fail immediately.
- Multichannel samples are averaged to mono before transcription. The accepted
  sample rate is published to the active session under the engine lock before the stream starts. Superseded workers close their stream without changing the active rate. Released transcription jobs retain the sample rate captured with their audio frames, independently of subsequent recording sessions.
- If the stream rate differs from the Whisper target rate, audio is resampled
  via linear interpolation before transcription.

## Edge Cases

| Case                      | Behaviour                                         |
|---------------------------|----------------------------------------------------|
| Whisper not yet loaded    | Play "not ready" beep, skip                        |
| Max recording duration    | No cap — the user controls when to stop by releasing the hotkey. A cap would paste prematurely mid-dictation and restart recording |
| Empty transcription       | No paste occurs                                    |
| Concurrent with assistant | Dictation works independently; pauses listener     |
| macOS permissions         | `pynput` requires Accessibility permissions        |
| macOS 26+ (Tahoe)         | `pynput` disabled — TSM main-thread assertion crash |
| Linux / Wayland           | `pynput` requires X11 (limited Wayland support)    |
| Audio rate mismatch       | Resample via linear interpolation to Whisper rate  |
| LLM filler removal fails  | Falls back to raw transcription (5 s timeout)     |
| Custom dictionary empty   | No-op, text passes through unchanged               |

## Thread Safety

- `threading.Lock` around shared Whisper model transcription calls. Model references are resolved under the lock so CPU recovery in the voice listener is visible to waiting dictation jobs.
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

The setup wizard includes a dedicated Dictation page (between Whisper and
Location steps) that allows users to:
- Enable/disable dictation
- Choose the hotkey from a dropdown of presets
- View tips about hold-to-dictate and double-tap hands-free mode

## Dependencies

- `pynput>=1.7.6` — global hotkey detection and keyboard simulation.
