# Jarvis on this Windows machine

## Environment

```powershell
cd C:\Users\janso\.cursor\projects\empty-window\Jancuks
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe --version   # Python 3.12
```

## Start

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m desktop_app
```

Tray menu: **Web Command Centre** opens the local dashboard (memory viewer server must be running).

### Type instead of speak (Nimbus-style Pulse chat)

1. Tray → **Start Listening** (daemon must be running).
2. Open **Show Pulse** (or tray → **Type to Jarvis** focuses the chat box).
3. Write in the bottom field → **Send** or **Ctrl+Enter**. Replies appear in the same window and via TTS.
4. Optional: **Web Command Centre** also has a compose panel.

### See your screen

- Pulse window → **📸 Skatīt ekrānu**, or ask in chat: *Ko tu redzi manā ekrānā?*
- Jarvis captures the display locally (no cloud), runs OCR if Tesseract is installed, and describes the image with your Ollama model when it supports vision.
- Config (`config.json`): `screen_auto_capture_enabled`, `screen_vision_enabled`, `ollama_vision_model` (empty = use chat model).

### Drag & drop images

- Pulse logā **ievelc** `.png`, `.jpg`, `.webp` u.c. uz čatu vai logu.
- Uzraksti jautājumu (vai atstāj tukšu — Jarvis jautās pats par attēlu).
- **Send** — attēls tiek analizēts lokāli (OCR + vision) un iet kā konteksts atbildē.

## Prerequisites

- [Ollama](https://ollama.com/download) running locally
- Models: `gemma4:e2b`, `nomic-embed-text` (setup wizard installs these)
- Microphone permission for Windows
- NVIDIA GPU optional: `whisper_device: cuda` in `%USERPROFILE%\.config\jarvis\config.json`

## Config

`%USERPROFILE%\.config\jarvis\config.json`

Performance-oriented defaults on this machine:

- `ollama_chat_model`: `gemma4:e2b`
- `intent_judge_model`: `gemma4:e2b`
- `whisper_model`: `small`
- `planner_enabled`: `false`

## Build Windows `.exe`

From the project root (uses `.venv` or `.mamba_env`):

```powershell
.\scripts\build_exe.bat
```

Output: `dist\Jarvis\Jarvis.exe` — run **from that folder** (all DLLs stay alongside the exe). Do not copy only `Jarvis.exe` elsewhere.

Optional full installer (needs [Inno Setup 6](https://jrsoftware.org/isdl.php) and `.mamba_env`):

```cmd
scripts\build_installer.bat
```

Produces `dist\Jarvis-Setup-x64.exe`.

**Prerequisites on the target PC:** [Microsoft VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), [Ollama](https://ollama.com/download), microphone permission. CUDA libraries are optional (installer task or manual).

## Tests

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest
```

## Real features roadmap (no demo data)

Add one capability at a time; each must use **your real files, screen, or APIs**.

| # | Feature | Status |
|---|---------|--------|
| 1 | **Screen see** (`screenshot` + vision) — auto-capture when you ask about the screen; Pulse → **Skatīt ekrānu** | Done — optional [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) for OCR; vision uses your Ollama chat/vision model |
| 2 | **Files** (`localFiles`) — home + `data_live_roots` | Done — configure roots below |
| 3 | **Operator briefing** — real file previews in prompt | Done |
| 4 | Work queue (`manageWorkQueue`) | Done |
| 5 | Gmail / WhatsApp (`getMcpIntegrations`, real MCP discovery) | Done — pair WhatsApp / add Google OAuth |
| 6 | Ledger (`getLedgerSummary`, real JSON only) | Done |
| 7 | Scheduled background sync (weather, files, ledger) | Done — cache at `%USERPROFILE%\.config\jarvis\operator_sync.json`; interval `background_sync_interval_sec` (default 900s) |

### `data_live_roots` (real folders only)

```json
{
  "operator_name": "Mr. Johnson",
  "persona_style": "formal_majordomo",
  "data_live_roots": [
    {"path": "C:/Users/janso/Documents", "label": "Documents"},
    {"path": "C:/path/to/your/work", "label": "Darbs"}
  ]
}
```

Restart Jarvis after editing config. Ask: *“Ko redzi manā ekrānā?”* (needs Tesseract) or *“Ko ir mapē Documents?”*

### Ledger (grāmatvedība)

Point at a folder with real files, e.g. Nimbus backup ledger:

```json
{
  "ledger_enabled": true,
  "ledger_path": "C:/Users/janso/.cursor/projects/empty-window/Jancuks_backup_20260518_151347/data/live/ledger",
  "data_live_roots": [
    {"path": "C:/Users/janso/.cursor/projects/empty-window/Jancuks_backup_20260518_151347/data/live/ledger", "label": "Grāmatvedība"}
  ]
}
```

Files: `invoice_*.json` (object with `lines[]`), `sales*.json` (array of rows with `revenue_eur`). Ask: *“Kāds ir pirkumu un pārdošanas kopsavilkums?”*

### Integrations wizard (WhatsApp, Gmail, personal pages)

Tray → **Setup Wizard** → **Your integrations**:

1. **WhatsApp** — tray → **Connect WhatsApp** (or Setup Wizard → **Connect WhatsApp (show QR)**). Scan QR in the dialog. Requires [Go](https://go.dev/dl/) on PATH. Restart listening after pairing.
2. **Gmail** — Google Cloud OAuth Client ID + Secret (Desktop app), enable, restart.
3. **Personal pages** — one line per site: `Label | https://...` (used in operator briefing + Chrome MCP).

### MCP (WhatsApp, Chrome, Maps)

Status file: `%USERPROFILE%\.config\jarvis\mcp_status.json` (written when daemon starts).

Your config already includes `whatsapp`, `chrome-devtools`, `google-maps`, `everything`. To use WhatsApp:

1. Install [uv](https://docs.astral.sh/uv/) so `uvx` works in PowerShell.
2. Restart Jarvis desktop (daemon re-discovers MCP tools).
3. First time: run `uvx whatsapp-mcp-server` in a terminal and complete QR pairing.
4. Dashboard → **MCP Integrations** shows `whatsapp: ready` or the real error.

Voice checks: *“Kādi MCP ir pieejami?”* or *“Refresh MCP integrations”* (`getMcpIntegrations`).

Optional Gmail — add `google-workspace-mcp` block (see `src/jarvis/operator/mcp_integrations.spec.md`).

## Nimbus backup

Previous project: `C:\Users\janso\.cursor\projects\empty-window\Jancuks_backup_20260518_151347`
