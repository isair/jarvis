# Startup briefing (desktop)

When the desktop app starts listening, optionally opens dashboards and speaks a short majordomo greeting.

## Config

| Key | Default | Purpose |
|-----|---------|---------|
| `startup_briefing_enabled` | `true` | Master switch for spoken greeting |
| `auto_open_pulse_on_start` | `true` | Open `/pulse` in the browser |
| `auto_open_dashboard_on_start` | `true` | Open `/dashboard` in the browser |
| `auto_open_face_on_start` | `true` | Show the Pulse presence window (chat + orb) |

Use with `operator_name` (e.g. Mr. Johnson) and `persona_style: formal_majordomo`.

## Behaviour

1. ~2s after `start_daemon`: refresh `operator_sync.json`, Gmail preview, comms log (force pulse sync).
2. Open Pulse + Web Command Centre when the memory viewer on port 5050 is up.
3. ~5s after start: speak a concise briefing. Prefer an LLM pass (`synthesize_startup_brief_llm`) on the tool-router model chain: interpret cached weather, Gmail, WhatsApp, and **work queue** (`work_queue.json` / Sulainis Task queue — same data). If there are open tasks, mention them and offer to work through them one at a time when the user asks. Fail-open to the template `build_startup_spoken_brief` (same queue lines + offer).
4. Mirror the same text in the presence window chat as an assistant line.

WhatsApp rows in `comms_log.json` use human chat labels only (`format_whatsapp_display_from` in `pulse_sync.py`); opaque `@lid` senders are dropped from the `from` field.

Runs once per desktop session. Fail-open: sync or TTS errors are logged; windows still open when configured.

## Privacy

Uses only local cache files and MCP refresh on the user's machine — no cloud briefing service.
