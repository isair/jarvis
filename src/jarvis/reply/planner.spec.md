# Task-list planner

## Purpose

Small chat models (gemma4:e2b class) don't reliably decompose multi-step
queries turn-by-turn. They stop after one tool call when a second is
needed, echo the raw user utterance into tool arguments, or skip tools
entirely and confabulate from training. The planner fixes this by
running a single cheap classification-shaped LLM pass at the top of the
reply flow that emits a short ordered list of sub-tasks. The engine
then uses that plan both as context for the chat model and as the
driver of a direct-execution path that resolves each step to a concrete
tool call without needing the chat model to plan turn-by-turn.

## Scope

This spec covers `src/jarvis/reply/planner.py` and the engine
integration in `src/jarvis/reply/engine.py`.

## Behaviour

### When the planner runs

- After tool selection (the router has produced a tools allow-list).
- Only when the query is at least `MIN_QUERY_CHARS` long (default 20).
  Shorter utterances are either trivial or not multi-step.
- Only when `cfg.planner_enabled` is True (default).
- Only when an `ollama_base_url` and a resolvable model are available.

### Model resolution

Same chain as the tool router:

1. `cfg.planner_model` (explicit override)
2. `cfg.tool_router_model`
3. `cfg.intent_judge_model`
4. `cfg.ollama_chat_model`

Planning is classification-shaped so it rides the warm small model
instead of paging in a separate planner model. This preserves KV-cache
warmth across planner, router, and intent judge.

### Prompt contract (plan_query)

The planner prompt instructs the model to emit:

- Short imperative sub-tasks, one per line.
- At most `MAX_STEPS` (default 5) steps.
- Tool names from the provided allow-list only.
- Concrete arguments composed against dialogue + memory context, not
  the raw utterance.
- Angle-bracket placeholders (e.g. `<director name from step 1>`) for
  entities the lookup will reveal at runtime.
- A final synthesis/reply step when any tools are planned.
- Steps in the same language the user wrote the query in.

### Parsing and hygiene

- Numbering (`1.`, `1)`), bullets (`-`, `*`, `•`), wrapping quotes,
  and markdown fences are stripped.
- Overlong steps (>200 chars) are truncated with an ellipsis.
- The list is capped at `MAX_STEPS`.
- Trivial plans (length ≤ 1) return an empty list so the engine falls
  through to existing behaviour. The planner prompt (rule 8) mandates
  a final synthesis step whenever any tool is planned, so a 1-step
  plan is by contract a pure reply and adds no value. This check is
  purely structural — no language-specific verb matching — so the
  filter works for any language the planner emits.

### Engine integration

- `format_plan_block(steps)` renders an `ACTION PLAN:` block that is
  appended to the initial system message. Empty plan renders nothing.
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
