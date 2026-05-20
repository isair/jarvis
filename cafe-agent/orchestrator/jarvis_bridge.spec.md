# Orchestrator Jarvis bridge (email / WhatsApp)

Rust handlers do **not** implement IMAP or WhatsApp wire protocols. They forward to the Python dashboard:

`POST /api/cafe-agent/jarvis-bridge` → `cafe_jarvis_bridge.handle_cafe_jarvis_bridge` → `queue_sulainis_action("briefing", …)`.

## Task mapping

| Cafe task | `action` (default `sync`) | Sulainis action |
|-----------|---------------------------|-----------------|
| `email` | `sync`, `inbox`, `briefing` | `briefing` (force) |
| `whatsapp` | `sync`, `threads`, `briefing` | `briefing` (force) |

Unknown actions still queue `briefing` with `requested_action` in payload.

## Requirements

- Flask `memory_viewer` on `127.0.0.1:5050` (or `JARVIS_DASHBOARD_PORT`)
- Jarvis daemon listening (tray or shell) so prompts reach MCP tools

## Sulainis UI

**Café ops** → **Email** / **WhatsApp** posts `{ type: "email"|"whatsapp", action: "sync" }` to `/api/cafe-agent/task` (same proxy as other agents).
