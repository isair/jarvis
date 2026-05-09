## Log Meal Tool Spec

Logs a single meal (or drink) to the nutrition database when the user
mentions eating or drinking something specific. Estimates approximate macros
and notable micronutrients via the chat model, then asks the same model for
short, pragmatic follow-ups for the rest of the day.

### Public schema

The tool exposes exactly one optional property:

```json
{
  "type": "object",
  "properties": {
    "meal": {
      "type": "string",
      "description": "Natural language description of what was eaten or drunk"
    }
  }
}
```

Nutrition fields (`description`, `calories_kcal`, `protein_g`, `carbs_g`,
`fat_g`, `fiber_g`, `sugar_g`, `sodium_mg`, `potassium_mg`, `micros`,
`confidence`) are **implementation details** resolved internally by
`extract_and_log_meal`. They MUST NOT appear in the public schema:

- They bloat the planner's tool catalogue, wasting context on a small model.
- They cannot be filled deterministically by the planner's fast-path
  parser (`logMeal meal='Big Mac'` is what the planner emits), so listing
  them as required would force the LLM resolver to hallucinate values.
- They are best estimated by the dedicated nutrition extractor system
  prompt (`NUTRITION_SYS`), not the planner.

The single `meal` key is what enables direct-exec for small models: the
planner emits `logMeal meal='Big Mac'`, the fast-path parser
(`_parse_plan_step_concrete`) accepts it because `meal` is a declared
property, and dispatch happens with no LLM resolver call.

### Extraction-input precedence

Inside `run()` the extractor input is chosen as:

1. `args["meal"]` — when the planner emits `logMeal meal='…'` via fast-path.
   Stripped; whitespace-only is treated as missing.
2. `context.redacted_text` — the full redacted utterance. Used when no
   `meal` arg is provided or it was empty.

If BOTH are empty (e.g. a pure voice trigger with no recognised speech),
the tool returns a graceful failure (`success=False`) with a friendly
"I didn't catch what you ate" prompt rather than calling the LLM with an
empty body.

### Untrusted-data fence

`original_text` (whether sourced from `meal` arg or `redacted_text`) is
treated as untrusted data inside the prompt to `NUTRITION_SYS`. It is
truncated to 1200 characters and wrapped in explicit delimiters:

```
<<<BEGIN UNTRUSTED USER TEXT>>>
…meal description…
<<<END UNTRUSTED USER TEXT>>>
```

The instruction above the fence tells the model to treat the contents as
data and ignore any embedded instructions. This is defence-in-depth: small
models still occasionally honour in-fence instructions, but the fence is a
detectable boundary for evals and reviewers, and reduces the surface for
trivial "ignore previous instructions" injections in meal descriptions.

### LLM passes

Two passes against the chat model (`cfg.llm_chat_model`):

1. **Extraction** (`extract_and_log_meal` → `NUTRITION_SYS`): returns either
   a JSON object with the nutrition fields above OR the literal string
   `NONE` if no meal is described. Fences (` ```json … ``` `) added by
   small models are stripped before parsing. Failure to parse returns
   `None` and the tool retries up to `context.max_retries`.
2. **Follow-ups** (`generate_followups_for_meal`): a short coach prompt
   asking for 2-3 healthy, realistic follow-ups (hydration, protein,
   veggies, sodium/potassium balance, light activity).

Both passes share `cfg.llm_chat_timeout_sec` and the `llm_thinking_enabled`
flag.

### Database

Logged via `Database.insert_meal(...)`, which uses parameterised SQL.
`source_app` is `"stdin"` when `cfg.use_stdin` is true, otherwise
`"unknown"`. Optional fields (potassium, micros, confidence) are stored as
NULL when missing.

### Reply shape

On success the tool returns:

```
Logged meal #<id>: <description> — <macro summary>[ (confidence X%)].
Follow-ups: <coach text>
```

The macro summary is a comma-joined list of present-only fields (kcal,
protein, carbs, fat, fiber). On failure: `"Failed to log meal"` (extractor
returned NONE or all retries raised) or `"No meal description provided"`
(extract-text guard).
