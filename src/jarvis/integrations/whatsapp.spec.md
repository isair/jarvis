# WhatsApp integration (lharries bridge + MCP)

Local pairing via the [lharries/whatsapp-mcp](https://github.com/lharries/whatsapp-mcp) Go bridge and Python MCP server. No GreenAPI cloud account.

## Components

1. **Go bridge** (`whatsapp-bridge/`) — listens on `http://localhost:8080/api`, stores session under `whatsapp-bridge/store/`.
2. **Python MCP** (`whatsapp-mcp-server/`) — Jarvis `mcps.whatsapp` runs `uv --directory … run main.py`.

## Desktop setup flow

1. User opens **Connect WhatsApp** (tray or Setup Wizard).
2. Jarvis clones the repo to `~/.config/jarvis/whatsapp-mcp` (zip download) if missing.
3. Applies a small stdout patch so the bridge prints `JARVIS_QR_CODE:…` and `JARVIS_AUTH_OK`.
4. Runs `go get -u go.mau.fi/whatsmeow@latest` and patches `main.go` for current whatsmeow APIs (avoids WhatsApp **405 Client outdated**).
5. Starts `go run main.go` in `whatsapp-bridge/` (requires **Go** on PATH).
6. UI renders the QR with Python `qrcode`; user scans in WhatsApp → Linked devices.
7. On `JARVIS_AUTH_OK`, Jarvis writes `mcps.whatsapp` and keeps the bridge process running.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Client outdated (405)` | Automatic on next **Connect WhatsApp** (whatsmeow upgrade). If it persists, delete `whatsapp-bridge/store/` and pair again. |
| `gcc not found` | Install WinLibs gcc (see Prerequisites), restart Jarvis. |

## Prerequisites

- [Go](https://go.dev/dl/) on PATH.
- **Windows:** a C compiler (**gcc**) is required (`go-sqlite3` uses CGO). If QR never appears and the log mentions `gcc not found`, run:
  `winget install BrechtSanders.WinLibs.POSIX.UCRT`
  then restart Jarvis and try again.
- [uv](https://docs.astral.sh/uv/) for the MCP server process.
- Do **not** use `uvx whatsapp-mcp-server` in config — that is a different (GreenAPI) package. Jarvis migrates to lharries when you use **Connect WhatsApp**.

## Privacy

Messages stay in local SQLite under the bridge `store/` directory. Only tool calls you approve reach the LLM.
