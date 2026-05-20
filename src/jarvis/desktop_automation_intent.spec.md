# Desktop automation intent

Detects when the user wants local Windows control and prevents small models from falsely claiming they cannot access the computer while `windows__*` MCP tools are on the allow-list.

## Behaviour

- `mentions_desktop_action(query)` — broad, language-mixed action verbs (open, launch, atver, …).
- `build_windows_automation_prompt_block` — injected into the reply system prompt when Windows tools are allowed.
- `try_resolve_desktop_tool_call` — fail-open preflight for simple app launches (`windows__App` mode `launch`).
- `is_false_tool_refusal` — detects canonical denial phrases; the reply engine retries with a correction nudge (max 2 per reply).

## Not a permission gate

Jarvis does not maintain a separate OS permission allow-list for desktop control beyond MCP configuration. If `windows` MCP is configured and discovered, tools are available. User consent is implied by enabling the integration in Settings.
