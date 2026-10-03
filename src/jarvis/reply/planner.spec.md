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
  `searchMemory` directive. An empty external catalogue still allows
  private-history preparation. The same personal-history rules apply with
  or without external tools.
- Only when the query is at least `MIN_QUERY_CHARS` long (default 4).
  Pure noise like "hi" / "ok" still short-circuits.
- Only when `cfg.planner_enabled` is True (default).
- Only when a resolvable chat model is available; calls use the configured LLM backend.

### Fast-path skip (engine-level)

The engine skips the planner entirely when **all** of these hold:

- The tool router returned no real tools (only system tools like `stop` — the router's positive "none" decision, not its fall-open-to-all-tools path).
- The query is short (≤ 8 words, split on whitespace — language-agnostic).
- `planner_enabled` is True (the skip is an optimisation, not a feature-gate bypass).

When skipped the engine injects `["Reply to the user."]` as the plan — a positive signal that no tools and no memory enrichment are needed. The warm-profile block is still injected, so the chat model sees user identity and preferences. Longer tool-free queries ("what do you know about my dietary preferences") still reach the planner so it can emit a `searchMemory` directive when the warm profile alone is insufficient.

### Generation budget

Planning and LLM step resolution share a bounded 1,024-token generation budget.
This includes any reasoning emitted by the backend before its plan or JSON
answer. The configured planner deadline still bounds each request; empty or
invalid output keeps the existing fail-open behaviour. The five-step plan
limit, tool allow-list and deterministic concrete-step fast path apply
independently of the generation budget.

### Model resolution

The planner runs on the chat tier (`resolve_model(cfg, Tier.CHAT)`).

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
  Personalised recommendations retrieve the user's tastes, interests,
  diet or history first. General facts about named people or places,
  utility requests and definitions use available tools or a direct reply
  without searching private conversation history. Explicit requests for
  what the user said about a named person still require memory.
- Tool names from the provided catalog only (exact match), for any
  concrete tool step.
- Concrete arguments composed against dialogue context, not the raw
  utterance. Optional arguments that the user did not supply must be
  omitted, not fabricated from unrelated words.
- Angle-bracket placeholders (e.g. `<director name from step 1>`) for
  entities the lookup will reveal at runtime.
- Pronouns and demonstratives in the user query ("he", "his", "her",
  "their", "it", "that film") must be resolved against the dialogue
  context before emitting the step. Tools never see prior turns, so
  the named entity has to appear literally inside the tool argument
  string — `webSearch query='Harry Styles most famous songs'`, not
  `webSearch query='his most famous songs'`.
- A final synthesis/reply step when any `searchMemory` or tool step
  was planned.
- Steps in the same language the user wrote the query in.
- Zero-temperature sampling for the plan classification; execution and
  synthesis keep their own sampling settings.
- Never emit `stop` as a plan step. The main assistant decides
  when to stop at runtime; a pre-planned stop directive would
  produce a silent dismissal for many non-trivial queries.
  Instead, emit `Reply to the user.` for dismissive queries so
  the assistant handles tone and termination naturally.
  A deterministic post-plan guard in `plan_query` rejects any
  plan where every step is `stop`, returning `[]` so the engine
  falls through to the tool router and chat model.
- Trust the tool router: when the available-tools catalogue contains
  a tool relevant to the query, plan to use it even for seemingly
  trivial requests (jokes, opinions, creative content). The tool
  router already judged the query needs external information — a
  reply-only plan overrides that judgment and produces stale replies.

### Parsing and hygiene

- Numbering (`1.`, `1)`), bullets (`-`, `*`, `•`), wrapping quotes,
  and markdown fences are stripped.
- Overlong steps (>200 chars) are truncated with an ellipsis.
- The list is capped at `MAX_STEPS`.
- The planner no longer filters out 1-step plans. A single
  `["Reply to the user."]` plan is the planner's *positive* decision
  that no memory or tools are needed — the engine uses that to skip
  the memory extractor, the tool router, and the direct-exec path
  entirely. A single-step tool plan like `["getWeather query='tomorrow'"]`
  is also preserved: `tool_steps_of` returns it as a tool step, the
  engine injects the ACTION PLAN block, and direct-exec runs the tool
  without waiting for the chat model. Only an **empty** list means
  "planner failed / disabled; fall open to legacy safe defaults"
  (run memory enrichment + tool router). The two states must stay
  distinguishable.

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
  Single-step tool plans ARE rendered so the model sees the planned
  tool call in its context.
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

- **Fast path**: the step names an allowed tool, uses `key='value'` or
  `key="value"` pairs matching its declared property keys, supplies every
  required field and has no `<placeholder>`. The resolver parses and
  returns this concrete call without model inference.
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
  free-form by design). Required fields must remain present after filtering.
- Incomplete concrete calls use the LLM resolver. If its argument object
  still lacks required fields, return `None` for a normal chat-model turn.
- Missing or null `arguments` represent an empty object for optional calls.
  Arrays, strings, numbers and booleans are rejected rather than converted
  into empty calls. Argument value types remain the tool implementation's
  responsibility.
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
| `planner_timeout_sec` | `3.0` | Timeout for plan and step-resolver LLM calls. Planner fails open on timeout — an empty list is returned and the engine behaves as if the planner never ran. |

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
