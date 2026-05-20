# Jarvis Shell Specification

Rust/Tauri 2 unified desktop window that replaces scattered PyQt windows and manual browser tabs with one side-nav shell embedding the existing Flask dashboard.

## Principles

- **Privacy-first**: UI loads only `127.0.0.1`; no remote UI assets required.
- **Boundary**: `jarvis_shell` does not link the Python `jarvis` crate. It spawns `desktop_app.memory_viewer` and embeds HTTP routes. Core logic, Whisper, LLM, and tools stay in Python.
- **Migration**: PyQt6 tray (`desktop_app`) remains until feature parity; both may run but must not double-bind port 5050.
- **Fail-open**: If the dashboard fails to start, the shell shows status on Home; user can retry.

## Layout (Cursor-inspired)

```
┌────┬────────────┬─────────────────────────────┐
│Act.│ Side nav   │ Title bar + editor (iframe) │
│bar │ per group  │                             │
│48px│ 260px      │                             │
└────┴────────────┴─────────────────────────────┘
```

Activity bar groups: **Explorer**, **Assistant**, **Operator**, **Settings** (gear).
Colours follow VS Code/Cursor dark theme (`#181818`, `#252526`, `#1e1e1e`, accent `#007fd4`).

### Routes (v0)

| Nav id | Embedded path | Notes |
|--------|---------------|-------|
| `home` | Native HTML panel | Backend status, no iframe |
| `command` | `/dashboard` | Command centre |
| `chat` | `/dashboard` | Assistant activity |
| `memory` | `/` | Diary & memories |
| `graph` | `/?tab=graph` | Knowledge graph tab |
| `meals` | `/?tab=meals` | Meals tab |
| `sulainis` | `/sulainis/` | Operator command centre |
| `pulse` | `/pulse/` | Pulse dashboard |
| `settings` | native panel | `GET/POST /api/settings/*` (metadata-driven); gear opens this |
| `logs` | `/dashboard` | Redacted logs panel |

**MCP (PyQt only):** Shell **MCP (PyQt)** button spawns `scripts/shell_settings.py` for the `mcps` category and catalogue UI.

## Backend orchestration

1. Resolve repo root: `JARVIS_ROOT` env, else walk from cwd (`src/` must exist).
2. Python: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python`, else `python`/`python3`.
3. `ensure_dashboard`: if TCP `127.0.0.1:5050` closed, spawn:

   ```text
   PYTHONPATH=<root>/src  python -m desktop_app.memory_viewer 5050
   ```

4. Poll up to 20s for port open; `GET /api/dashboard/status` for `listening` / `listener_active`.

## Voice / listener

- Shell does **not** auto-start Whisper on window open.
- **Start listening** / **Stop listening** call `scripts/shell_daemon.py` (`python -m jarvis.daemon` subprocess). A cross-process lock at `~/.local/share/jarvis/jarvis_daemon.lock` prevents duplicate daemons (shell + PyQt tray).
- Tray app respects `auto_start_listening` (default `false`).
- Dictation (hold hotkey → paste) remains separate from Jarvis PTT.

## Security

- CSP allows `frame-src` and `connect-src` to localhost only.
- iframe `sandbox` allows scripts/forms for Sulainis/Pulse static assets.

## Voice config (v0.1)

- `GET /api/dashboard/voice-config` on Flask (PTT hotkey display, `continuous_listening`, `whisper_lazy_load`, model).
- Tauri `get_voice_config` proxies that JSON for the Home metrics grid.
- `start_listener` fails fast with a clear error if port 5050 is closed.
- `ensure_cafe_orchestrator` spawns `cafe-orchestrator` on port **8787** (release binary or `cargo run`).

## Deep links (v0.2)

Embedded routes use Flask query tabs (memory viewer reads `?tab=`):

| Nav id | iframe URL |
|--------|------------|
| `memory` | `/?tab=memories` |
| `graph` | `/?tab=graph` |
| `meals` | `/?tab=meals` |

`selectRoute` forces iframe reload when re-selecting the same URL. Last route id is stored in `sessionStorage` (`jarvis_shell_route`).

## Native settings (v0.3)

- `GET /api/settings/metadata` — categories + `FIELD_METADATA` (excludes `mcps`).
- `GET /api/settings/config` — merged effective values (defaults + `config.json`).
- `POST /api/settings/config` — body `{ "values": { ... } }`; same save rules as PyQt (non-default keys only).
- Shell `#settings-panel` in `ui/settings.js`; activity-bar gear → native panel.

## Daemon lock (v0.4)

- `jarvis.daemon_lock` — `fcntl` / `msvcrt` exclusive lock; PID at byte 0 of `jarvis_daemon.lock`.
- `jarvis.daemon` acquires the lock on startup; second instance exits with code 2.
- PyQt tray `start_daemon` attaches UI if the lock is already held (e.g. shell started listening).
- `stop_daemon` without a local subprocess calls `stop_locked_daemon()` to end an external holder.

## Quick message (v0.5)

- Home **Quick message** → `POST /api/dashboard/query` → `jarvis.text_input` (`inbox` when daemon not attached, `listener` when running).
- User must **Start listening** to process inbox backlog; PTT still requires the daemon for transcription.

## Future work

- Replace iframe with `tauri::WebviewWindow` only if CSP issues arise on Windows.
- PTT hotkey without loading the full wake-word loop (lazy daemon profile).

## Related specs

- `src/desktop_app/desktop_app.spec.md` — PyQt tray, startup, daemon
- `src/jarvis/dictation/dictation.spec.md` — hold-to-dictate paste
- `src/jarvis/listening/listening.spec.md` — wake word pipeline
