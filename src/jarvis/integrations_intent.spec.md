# Integrations intent (Gmail)

When `google_workspace__*Gmail*` tools are on the allow-list, small models often deny having access to the user's Gmail even though OAuth is configured locally.

## Behaviour

- `mentions_gmail_query(query)` — inbox / access / "can you see my gmail" style questions (English + Latvian fragments).
- `build_gmail_integration_prompt_block` — injected into the reply system prompt when Gmail tools are allowed (denial-template mirroring: the user has *connected* access on this PC).
- `try_resolve_gmail_tool_call` — fail-open preflight calling `google_workspace__searchGmail` on `in:inbox` (account from `~/.google-mcp/accounts.json` when present).
- Reuses `is_false_tool_refusal` from `desktop_automation_intent`; the reply engine retries with a Gmail-specific correction nudge (shared retry budget with desktop refusals).

## Not a permission gate

Access is implied by enabling `google_workspace` MCP and completing OAuth. Jarvis does not maintain a separate Gmail allow-list beyond MCP discovery.
