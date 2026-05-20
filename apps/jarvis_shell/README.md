# Jarvis Shell (Tauri 2)

Unified windowed desktop for Jarvis: side navigation, embedded local dashboard (`127.0.0.1:5050`), Python backend orchestration.

The PyQt6 tray app remains available during migration; this shell does **not** import `jarvis` from Rust. It spawns the existing Flask dashboard (`desktop_app.memory_viewer`) and embeds routes in a WebView.

## Prerequisites

1. **Rust**: `winget install Rustlang.Rustup` then `rustup default stable`
2. **MSVC Build Tools**: `winget install Microsoft.VisualStudio.2022.BuildTools` (C++ workload)
3. **Node.js** v18+
4. **Python venv** at repo root with dependencies (incl. PyQt6 for Settings)

## Quick start

From the repository root:

```powershell
.\scripts\run_jarvis_shell.ps1
```

Or manually:

```powershell
$env:JARVIS_ROOT = (Get-Location).Path
cd apps\jarvis_shell
npm install
..\.venv\Scripts\python.exe scripts\gen_icons.py
npm run dev
```

`JARVIS_ROOT` must point at the repo root (folder containing `src/`). The shell starts the memory viewer on port **5050** if it is not already running.

## Navigation

| Item | URL |
|------|-----|
| Home | Status tiles (native panel) |
| Presence / Memory / Graph / Settings / Logs | `http://127.0.0.1:5050/` (same Flask app; section anchors TBD) |
| Sulainis | `/sulainis/` |
| Pulse | `/pulse/` |

Voice listener control stays in the Python daemon (tray or future PTT). Set `auto_start_listening: false` in config so Whisper does not load at app open.

## Build installer (release)

From `apps/jarvis_shell` with repo venv and Rust toolchain ready:

```powershell
$env:JARVIS_ROOT = (Resolve-Path ..\..).Path
cd apps\jarvis_shell
npm install
..\..\.venv\Scripts\python.exe ..\..\scripts\gen_icons.py
npm run build
```

Output: `src-tauri/target/release/bundle/` (`.msi` on Windows, `.dmg` on macOS).

**Dev vs release:** `npm run dev` uses the Vite dev server; `npm run build` embeds `ui/` into the Tauri binary. After changing `ui/app.js` or `styles.css`, rebuild for the installed app to pick up UI changes.

Recommended voice stack in `~/.config/jarvis/config.json`:

```json
"auto_start_listening": false,
"continuous_listening": false,
"whisper_lazy_load": true,
"ptt_enabled": true,
"ptt_hotkey": "ctrl+shift+j"
```

## Spec

See [jarvis_shell.spec.md](./jarvis_shell.spec.md).
