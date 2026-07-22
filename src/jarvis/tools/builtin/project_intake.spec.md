## Project Intake Tool Spec

Runs a deterministic, multi-turn, one-question-at-a-time interview to
build a project brief when the user wants to kick off new work (e.g.
"vamos começar um novo projeto"). Unlike a normal builtin tool, this one
is backed by a **deterministic pre-planner gate** so that state survives
across turns without depending on the chat model's own memory of the
conversation.

### Why a gate, not just a tool

A single tool call cannot hold multi-turn state on its own — the
planner/router run fresh every turn and have no guarantee of recognising
"we are mid-interview" the way a small local model might miss it. The
fix mirrors `recall_gate.spec.md`: a cheap, no-LLM, pre-flight check that
runs **before** the planner, and when it fires, bypasses planner/router
entirely for that turn. Determinism lives in the code path, not in the
model's discipline.

### Database

```sql
project_intake_sessions (
  id INTEGER PRIMARY KEY,
  conversation_id TEXT NOT NULL DEFAULT 'default',  -- ties session to the active conversation
  project_name TEXT,                 -- filled once known, may start NULL
  project_type TEXT,                 -- template key once resolved, else NULL
  status TEXT NOT NULL,              -- 'awaiting_type' | 'in_progress' | 'completed'
  questions_json TEXT,               -- frozen question list once type resolves
  answers_json TEXT NOT NULL DEFAULT '[]',
  current_index INTEGER NOT NULL DEFAULT 0,
  abandoned INTEGER NOT NULL DEFAULT 0,  -- see "Abandoning an in-progress interview"
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

One active (`status != 'completed'`) row per `conversation_id` at a time.
Starting a new project while one is already in progress is handled as a
tool-level decision (see "Starting" below), not a DB constraint — this
keeps the failure mode a friendly reply instead of a write error.

`conversation_id` defaults to the constant `'default'` and is not
currently varied: the app runs a single global dialogue session (one
`DialogueMemory` instance, not multiplexed by conversation ID), so "one
active row per conversation" collapses to "one active row, full stop"
in practice. The column is kept for forward compatibility if the app
ever multiplexes sessions.

### Templates

Templates live in `project_templates.json` (project-config territory,
alongside `config.json`, not hardcoded in Python) so new categories or
edited questions never require a code change:

```json
{
  "<template_key>": {
    "label": "Human-readable name",
    "keywords": ["match", "terms", "lowercased"],
    "questions": ["Question 1", "Question 2", "..."]
  }
}
```

`other` is required and MUST have an empty `keywords` list — it is the
fallback template, never matched on purpose, always available as a
catch-all.

### Public schema

Single optional string input, following the same direct-exec-friendly
shape as `log_meal`:

```json
{
  "type": "object",
  "properties": {
    "input": {
      "type": "string",
      "description": "What the user said — either a request to start a new project, an answer to the pending intake question, or the project type when asked"
    }
  }
}
```

The tool infers what to do from **database state**, not from the
argument's shape. This is what makes it safe for the planner to call
blindly with `input='<redacted user text>'` every relevant turn.

### The gate (engine-level, runs before the planner)

In `run_reply_engine`, immediately after redaction and before
`plan_query()`:

```
session = get_active_intake_session(conversation_id)
if session is not None:
    force tool_call = {"name": "projectIntake", "arguments": {"input": redacted_text}}
    skip planner, tool router, and memory enrichment for this turn
```

This is a hard override, same tier as the recall gate — cheap, pure
lookup, fail-open (`get_active_intake_session` returning `None` on any
DB error just lets the turn proceed normally). Because it runs before
the planner, an in-progress interview can never be derailed by the
planner deciding to do something else with the turn.

### Trigger detection (starting a new session)

When the gate finds **no** active session, normal routing applies. The
tool's one-line description in the catalogue is what lets the router/
planner select it for a fresh start:

> "Call when the user wants to start a new project, kick off a new
> piece of work, or explicitly says something like 'let's start a new
> project' / 'vamos começar um novo projeto'."

This is the only point in the flow where tool selection depends on
normal (LLM) routing — acceptable because starting is a single,
low-stakes classification, not a multi-turn state to lose track of.

### Flow

1. **No active session, trigger phrase recognised** → `run()` creates a
   row with `status='awaiting_type'`, `questions_json=NULL`. Returns:
   > "Que tipo de projeto é este? Site, App, Campanha de Marketing,
   > Marca/Identidade Visual, ou Outro?"

2. **`status='awaiting_type'`** → deterministic keyword match: NFKD-normalise
   `input` (decompose accented characters, strip the combining marks) then
   casefold, check substring overlap against each
   template's `keywords` list (first match wins; longer keyword lists
   checked before shorter ones to prefer specific over generic).
   - Match found → freeze that template's `questions` into
     `questions_json`, set `project_type`, `status='in_progress'`,
     `current_index=0`. Return question 1.
   - No match → fall back to `other`, same transition. Never re-asks
     the type question — an unrecognised answer must not stall the
     flow.

3. **`status='in_progress'`** → append `input` to `answers_json` at
   `current_index`, increment `current_index`.
   - More questions remain → return the next question verbatim from
     `questions_json`.
   - Last question just answered → set `status='completed'`, compile
     the brief (template label + every question/answer pair, ordered),
     persist it (see "On completion"), return the compiled summary.

4. **No active session, no trigger phrase** → tool is simply not
   selected this turn; irrelevant to normal conversation.

### One question per turn, by construction

The tool returns exactly one question's text per call — never the full
list. The chat model's only job is to relay that string. This makes
"ask one thing and stop" a property of the data returned, not a rule the
model has to remember to follow.

### On completion — write to Obsidian, do NOT auto-delegate yet

Intake ending and development starting are two separate, user-gated
moments — completion only **saves the plan**; it does not hand anything
to Antigravity. That happens later, on its own explicit trigger (see
"Starting development" below), so the user reviews/edits the plan in
the vault before any agent work is dispatched.

- The compiled brief is written as a note via the Obsidian MCP
  (`Jarvis Brain`, respecting the vault's existing PARA structure and
  note-format conventions):
  - Path: the relevant project folder if one already exists for this
    business/client, else a new folder created under the vault's
    project-numbering convention (same pass that creates the folder
    also creates/updates its `<Folder Name>.md` index, per the vault's
    own rules).
  - Filename: the project name (slugified) if given during intake, else
    `<project_type> - <YYYY-MM-DD>`.
  - Frontmatter: `status: active`, `project: <slug>`, `type: plan`.
  - Body: template label as a heading, then every question/answer pair
    in order, then a `## Status` line: `Plano criado, desenvolvimento
    ainda não iniciado.`
- `project_intake_sessions.answers_json` also keeps a local copy
  (already written during the flow) purely as a fail-open cache — if
  the Obsidian MCP write fails, the tool reports the failure honestly
  ("brief guardado localmente, mas falhou a gravação no Obsidian —
  tenta 'grava o plano' outra vez mais tarde") rather than claiming
  success it can't back up. This is the same rule as everywhere else in
  the system: never confirm an action the tool result didn't confirm.
- The `project_intake_sessions` row stays `status='completed'` for
  history; it is not deleted. A later "vamos começar um novo projeto"
  always opens a fresh row.

**Implementation note**: no question in the current templates explicitly
captures a project name, and `project_name` is never populated, so today
every write takes the `<project_type> - <YYYY-MM-DD>` filename branch.
The "relevant project folder if one already exists" lookup is also
simplified to a fixed `Projects/<slug>/` path rather than a live vault
search for a matching business/client folder — full PARA-aware
folder-detection needs to be validated against the user's actual vault
schema and Obsidian MCP server before being tightened further. The exact
Obsidian/Antigravity MCP tool names used (`create_note`, `simple_search`,
`get_file_contents`, `patch_content`, `run_task`) are best-effort
defaults in `project_intake.py` — confirm they match the user's actual
configured servers.

### Starting development (separate trigger, any later session)

A distinct, stateless tool/directive — no gate needed here, because
this is a single-shot classification ("the user wants to kick off
development on an existing plan"), not multi-turn state to protect.

Trigger phrasing: "vamos começar o desenvolvimento", "avança com o
projeto X", "manda isto para os agentes".

1. Resolve **which** project: if the user named one, match it against
   note filenames/`project` slugs in the vault via the Obsidian MCP
   search; if none named and exactly one `status: active`, `type: plan`
   note exists, use that; if several match, ask the user which one
   (single question, per the "close the loop" rule) instead of
   guessing.
2. Read the resolved note's full content via the Obsidian MCP.
3. Send that content as the opening task to Antigravity via MCP, with
   an instruction wrapper making clear this is a production plan to be
   broken down and assigned to its own sub-agents — the plan text
   itself is treated as the brief, not as instructions to Jarvis.
4. On a successful hand-off, update the note's frontmatter to
   `status: active` stays, but flip the `## Status` line to
   `Desenvolvimento iniciado em <data>, delegado ao Antigravity.` —
   this is an explicit vault write, not an inferred one, so it follows
   the same confirm-only-after-real-result rule.
5. If the Obsidian search finds no plan notes at all, say so plainly
   and offer to start a fresh intake instead of guessing at content.

This keeps a clean separation: intake is Jarvis's job (structured,
gated, deterministic), planning is the human's review pass in
Obsidian, and delegation to Antigravity is a distinct, explicit action
that only fires when asked — never automatically the moment intake
finishes.

### Abandoning an in-progress interview

If the user's `input` during `in_progress` or `awaiting_type` is clearly
an unrelated request (not an answer, not a type) rather than trying to
detect this via another LLM call, expose an explicit escape hatch: a
recognised phrase like "esquece o projeto" / "cancela isto" sets
`status='completed'` with `answers_json` unchanged (incomplete) and a
`abandoned=true` flag, so the gate stops force-routing future turns to
this tool. Without an explicit exit, a user who changes their mind mid-
interview would otherwise be stuck being asked questions forever.

### Reply shape

```
question turn  → "<verbatim question text>"
completion turn → "Brief do projeto '<project_name or project_type>' concluído:\n<Q1>: <A1>\n<Q2>: <A2>\n..."
abandon turn   → "Ok, cancelei o intake do projeto. Diz 'vamos começar um novo projeto' quando quiseres recomeçar."
```

### Fail-open behaviour

- `get_active_intake_session` DB error → treated as "no session", gate
  does not fire, normal routing proceeds. An interview can stall but
  never corrupts a turn.
- Keyword match against a malformed/missing `project_templates.json` →
  falls back to `other` with a minimal built-in question set
  (name, target audience, success criteria, deadline) so the tool never
  hard-fails even with a broken config file.
- Any write failure while advancing `current_index` → the turn returns
  a friendly "não consegui guardar essa resposta, podes repetir?" and
  does **not** advance the index, so the answer isn't silently dropped.

### Config keys

- `project_templates_path` — path to `project_templates.json`, default
  co-located with `config.json`.
- `project_intake_enabled` — default `true`; when `false`, the gate
  never fires and the tool is excluded from the catalogue entirely.

### Testing

- Gate unit tests: active session forces the tool call; no session
  leaves the planner untouched; DB error fails open.
- Template matching: keyword overlap resolves ties toward more specific
  templates; unmatched input falls back to `other`; NFKC/casefold
  normalisation covers accented input ("PÁGINA" matches "pagina").
- Full-flow integration test: awaiting_type → in_progress → completed,
  asserting exactly one question is returned per turn and the compiled
  brief contains every Q/A pair in order.
- Abandon-phrase test: mid-interview cancellation stops the gate from
  firing on the next turn.
