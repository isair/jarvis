# Email drafting (Sulainis)

## Purpose

Proactive Gmail reply assistance for the Sulainis dashboard: suggest short reply
angles and a full draft body from cached inbox previews, without sending mail.

## Scope

- `src/jarvis/email_drafting.py`
- `desktop_app/pulse_sync.py` (Gmail preview enrichment after MCP fetch)
- `static/sulainis/` UI (suggestion chips + pre-filled draft panel)

## Behaviour

- Runs during `sync_gmail_preview` when `sulainis_email_draft_suggestions` is true
  (default). At most `sulainis_email_draft_max` messages per sync (default 3).
- Uses a small local LLM (`intent_judge_model` chain) with temperature 0.
- Language: `reply_language` only (`en` default). `latvian_quality_enabled` does
  not switch draft text to Latvian.
- Incoming mail fields are fenced as untrusted data; output is JSON only:
  `angles` (2–3 short reply options), `subject`, `body`.
- Suggestions are stored on each message as `draft_suggestion` in
  `gmail_preview.json`.
- Optional: if a Gmail create-draft MCP tool is discoverable and `message_id`
  is present, create a Gmail draft (fail-open).
- Sulainis shows suggestion chips; selecting one fills the draft textarea.
  «Draft reply» still queues the full Jarvis agent path for refinement.

## Privacy

No cloud APIs. LLM and MCP calls stay on the operator's machine.
