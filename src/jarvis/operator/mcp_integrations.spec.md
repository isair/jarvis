# MCP integrations (comms + tools)

Real status only: discovery results are saved to `~/.config/jarvis/mcp_status.json` when the daemon starts or when `getMcpIntegrations` / `refreshMCPTools` runs. No demo “connected” flags.

## Tool

- `getMcpIntegrations` — report per-server `ready` / `error` / `empty` and tool name preview.
- `refresh: true` re-runs discovery.

## WhatsApp (user config)

```json
"whatsapp": {
  "transport": "stdio",
  "command": "uvx",
  "args": ["whatsapp-mcp-server"]
}
```

Requires `uv`/`uvx` on PATH and completing the WhatsApp pairing flow the server prints on first run. Until paired, discovery may return `error` or zero tools — that is reported honestly.

## Gmail / Google Workspace

Not enabled by default. Add when OAuth credentials exist:

```json
"google_workspace": {
  "command": "npx",
  "args": ["-y", "google-workspace-mcp", "mcp"],
  "env": {
    "GOOGLE_CLIENT_ID": "...",
    "GOOGLE_CLIENT_SECRET": "..."
  }
}
```

See upstream README for OAuth setup. Jarvis only invokes tools after discovery lists them.

## Work queue

`manageWorkQueue` item types `comms_email_reply` and `comms_whatsapp_send` are for tracking; execution still goes through the relevant MCP tool when that server is `ready`.
