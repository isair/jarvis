# Task-list planner

## Purpose

Small chat models (gemma4:e2b class) don't reliably decompose multi-step
queries turn-by-turn. They stop after one tool call when a second is
needed, echo the raw user utterance into tool arguments, or skip tools
entirely and confabulate from training. The planner fixes this by
running a single cheap classification-shaped LLM pass **at the very
front of the reply flow** that emits a short ordered list of sub-tasks.

The planner runs **after the tool router** and **before memory search**.
The router narrows the catalogue first so the planner's tool steps reference
concrete chosen names; the planner then **gates memory enrichment** and
**drives direct execution** for small models.

The engine uses the plan for three things:
1. **Gate memory enrichment** — the planner emits an explicit
   `searchMemory topic='<topic>'` directive on queries that need past
   user context; we skip the keyword-extraction LLM call, the diary
   / graph lookup, and the memory-digest LLM call otherwise.
2. **Confirm the tool allow-list** — the router's picks are
   authoritative; the tool names the planner references are unioned
   in as a safety net. Feeding the planner the narrowed catalogue
   (instead of the full 30+ list) stops small planners from
   paraphrasing ("get the weather") and from defaulting to
   `webSearch` when a more specific tool exists.
3. **Drive direct execution** for small models, as before — each
   planned step is resolved to a concrete tool call without
   round-tripping the chat model for intermediate turns.

## Scope

This spec covers `src/jarvis/reply/planner.py` and the engine
integration in `src/jarvis/reply/engine.py`.

## Behaviour

### When the planner runs

- After the dialogue context is assembled, MCP tools are loaded, and
  the tool router has produced a narrowed catalogue. Memory search
  runs *after* the planner so it can be gated on its output.
- The planner sees the **router-narrowed** tool catalogue (name +
  one-line description), not the full 30+ list. It does not see memory
  content — it decides whether memory is needed, via the
  `searchMemory` directive.
- Only when the query is at least `MIN_QUERY_CHARS` long (default 4).
  Pure noise like "hi" / "ok" still short-circuits.
- Only when `cfg.planner_enabled` is True (default).
- Only when an `ollama_base_url` and a resolvable model are available.

### Model resolution

1. `cfg.planner_model` (explicit override, for benchmarking)
2. `cfg.ollama_chat_model`

The planner must track the chat model. The plan is the scaffolding the
chat model follows; a weaker planner on top of a stronger chat model
produces bad scaffolding the chat model then fights against. The chat
model is also the one the user picked during setup as their quality
target, so upgrading it (through the setup wizard or config) must
automatically upgrade plan quality without requiring a second choice.

Note: the planner pays a cache miss relative to the tool router, which
*does* ride the warm small model. This is the intended trade-off —
plan quality drives everything downstream, router quality only narrows
one turn's allow-list.

### Prompt contract (plan_query)

The planner prompt instructs the model to emit:

- Short imperative sub-tasks, one per line.
- At most `MAX_STEPS` (default 5) steps.
- As the FIRST step, a `searchMemory topic='<topic>'` directive **only
  when** answering requires information the user shared in prior
  conversations. Omit otherwise — every extra directive is an
  avoidable LLM call downstream.
- Tool names from the provided catalog only (exact match), for any
  concrete tool step.
- Concrete arguments composed against dialogue context, not the raw
  utterance. Optional arguments that the user did not supply must be
  omitted, not fabricated from unrelated words.
- Angle-bracket placeholders (e.g. `<director name from step 1>`) for
  entities the lookup will reveal at runtime.
- A final synthesis/reply step when any `searchMemory` or tool step
  was planned.
- Steps in the same language the user wrote the query in.

### Parsing and hygiene

- Numbering (`1.`, `1)`), bullets (`-`, `*`, `•`), wrapping quotes,
  and markdown fences are stripped.
- Overlong steps (>200 chars) are truncated with an ellipsis.
- The list is capped at `MAX_STEPS`.
- The planner no longer filters out 1-step plans. A single
  `["Reply to the user."]` plan is the planner's *positive* decision
  that no memory or tools are needed — the engine uses that to skip
  the memory extractor, the tool router, and the direct-exec path
  entirely. Only an **empty** list means "planner failed / disabled;
  fall open to legacy safe defaults" (run memory enrichment + tool
  router). The two states must stay distinguishable.

### Engine integration

The engine consumes the plan in two phases.

**Phase 1 — preparation gating (before the turn loop starts):**

- `plan_requires_memory(plan)` — true iff any step is a `searchMemory`
  directive. The engine uses it to gate the entire memory-enrichment
  block (keyword extractor LLM call, diary / graph lookups, digest
  LLM call). Optional `memory_topic_of(step)` extracts the directive's
  `topic='...'` hint, threaded into the keyword extractor so it
  anchors on what the planner wanted to look up rather than
  re-deriving from the raw utterance.
- `tool_names_in_plan(plan, known_names)` — ordered de-duped list of
  tool names the planner referenced. The engine unions this into the
  router-selected allow-list (never replaces it). `stop` and
  `toolSearchTool` are always added regardless.
- `plan_has_unresolved_tool_steps(plan, known_names)` — true when the
  plan has non-synthesis steps but names no known tool (e.g. the
  model wrote `get the weather` instead of `getWeather ...`). In
  this state the direct-exec path is skipped — vague step text
  would otherwise force the resolver LLM to guess arguments (e.g.
  emitting `location='Nowhere'` for a bare weather request). The
  chat model takes the turn instead, using the router-selected
  allow-list.
- `strip_memory_directives(plan)` — the engine strips the
  `searchMemory` step from the plan once memory has been fetched, so
  downstream consumers (system-message injection, direct-exec,
  progress nudge) see a plan of pure tool + synthesis steps.

**Phase 2 — loop integration (existing behaviour):**

- `format_plan_block(steps)` renders an `ACTION PLAN:` block that is
  appended to the initial system message. Empty plan renders nothing.
  Single-step reply-only plans are not rendered either — they are
  noise to the chat model since the plan just says "reply".
- `progress_nudge(steps, tool_results_so_far)` produces a remainder
  hint injected after each tool result, naming the next planned step
  and reminding the model to substitute discovered entities and avoid
  duplicate arguments.
- When `use_text_tools` is active and the plan still has unexecuted
  tool steps, the engine runs `resolve_next_tool_call` to convert the
  next step into a concrete `{name, arguments}` JSON and dispatches
  the tool directly, bypassing the chat model for that turn. This
  keeps small models on-rails without relying on their native
  tool-call reliability.
- The chat model still runs the final synthesis turn so the reply is
  phrased in the daemon's voice using its own profile and persona.

### resolve_next_tool_call

- **Fast path**: if the step text is fully concrete (tool name in the
  allow-list + `key='value'` / `key="value"` pairs matching the tool's
  declared property keys, and no `<placeholder>`), parse it
  deterministically and return without any LLM call. This removes the
  resolver LLM as a failure surface for the common case — small models
  occasionally flake (timeout, empty, spurious `null`) even on
  trivially-concrete steps like `webSearch query='foo'`, which used to
  fall back to the chat model and produce a refusal instead of the
  search. The fast path is purely regex-driven, language-agnostic, and
  never calls the model.
- **LLM path**: when the step contains a `<placeholder>`, uses unknown
  argument keys, or doesn't fit the `key=value` shape, the step is
  passed to the LLM resolver which can substitute entities from prior
  results and remap names.
- Returns `None` for synthesis steps (the LLM emits the literal
  `null`), unknown tools, or invalid JSON. All `None` paths fall back
  to the normal chat-model turn.
- Validates the tool name against the provided schema's allow-list.
- Filters the returned `arguments` against the tool's declared
  JSON-schema property keys; unknown keys are dropped before dispatch.
  Tools that declare no properties keep the args as-is (they are
  free-form by design).
- Tolerates markdown fences the model may add despite instructions.
- Both planner LLM calls (`plan_query` and `resolve_next_tool_call`)
  request `num_ctx=8192` from Ollama so enriched memory and tool
  catalogue don't silently truncate in the 4096-token default window.

## Fail-open invariants

- Timeout, empty response, or exception in the planner LLM call →
  return `[]`.
- Invalid JSON in the step resolver → return `None` and let the chat
  model handle the turn normally.
- No plan never worsens the baseline; the engine behaves exactly as it
  did pre-planner.

## Configuration

| Key | Default | Purpose |
|-----|---------|---------|
| `planner_enabled` | `True` | Feature gate. |
| `planner_model` | `""` | Explicit planner model override. |
| `planner_timeout_sec` | `6.0` | Timeout for plan and step-resolver LLM calls. |

## Non-goals

- The planner does not re-plan mid-turn. If the emitted plan is wrong,
  the engine still progresses via the chat model's native tool calls.
  When the chat model produces natural-language content the loop
  terminates immediately.
- The planner does not validate semantic correctness of the plan; it
  trusts the model to produce sensible steps and relies on the
  resolver's schema-level guard to reject unknown tools.
- Plans are not cached across turns. Each user utterance gets its own
  plan because the dialogue state and entity references change.
