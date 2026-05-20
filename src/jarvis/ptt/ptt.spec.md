# Push-to-Talk (PTT) Specification

## Overview

Hold a global hotkey to record speech; on release, transcribe with the shared Whisper model and submit the text to Jarvis via `jarvis.text_input.submit_text_query`. No clipboard paste, no wake word required for that utterance.

Implementation reuses `DictationEngine` with `delivery_mode="jarvis"` (see `dictation/dictation.spec.md` for audio/hotkey mechanics).

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ptt_enabled` | bool | `true` | Master switch |
| `ptt_hotkey` | string | `ctrl+shift+j` | Hold-to-record combo (distinct from dictation) |
| `continuous_listening` | bool | `true` | When `false`, daemon skips the always-on wake-word mic loop |
| `whisper_lazy_load` | bool | `false` | When `true`, Whisper loads on first PTT/dictation/utterance |

Recommended PTT-first stack:

```json
{
  "auto_start_listening": false,
  "continuous_listening": false,
  "ptt_enabled": true,
  "ptt_hotkey": "ctrl+shift+j"
}
```

## Flow

1. User holds `ptt_hotkey` → start beep, face `LISTENING`, pause wake-word loop (`_dictation_active`).
2. User releases → stop beep, face `THINKING`, transcribe (shared Whisper + lock).
3. Non-empty transcript → `submit_text_query(text)` → same path as typed input / Command Centre.
4. Face `IDLE`, resume wake loop flag.

## Requirements

- Daemon must be running (tray **Start listening** or shell **Start listening**).
- With `whisper_lazy_load: false`, Whisper loads when the listener thread starts. With `true`, first PTT hold triggers load (may take tens of seconds).
- When `continuous_listening` is `false`, only PTT/typed/dictation use the mic; no background wake-word stream.

## Related

- `src/jarvis/dictation/dictation.spec.md` — shared hold-to-record engine
- `src/jarvis/listening/listening.spec.md` — typed input and inbox drain
- `apps/jarvis_shell/jarvis_shell.spec.md` — unified window + listener controls
