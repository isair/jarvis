# Tool schema presentation

The tool registry supplies the selected built-in and MCP tool definitions to both
native function calling and the text tool-call protocol. Both representations
use the same canonical name, description and complete input schema. Unselected
tools are excluded from the presented catalogue.

Text help includes the tool name and description followed by `Input schema:` and
a compact JSON object. It retains parameter descriptions, enum and constant
values, required fields, nested structures, conditional constraints and additional
property rules. Empty argument contracts remain explicit. JSON encoding preserves
Unicode and escapes embedded newlines and quotes without creating extra catalogue
lines. Unicode newline separators use JSON escapes so each schema remains one
catalogue record. Rendering does not mutate the tool's schema.

Tools without a supplied input schema use the registry's empty object contract in
both representations. Text generation reuses native definition selection and
schema construction, so provider format does not define a different input contract.

Built-in tools declare `allows_repeated_action(args)` for matching requests
with distinct IDs in one model batch. Timer creation opts in; other built-ins
retain coalescing. The engine snapshots outcomes before the batch, preserving
whole-request retry protection across later model turns.

`schema.missing_required_arguments` is the shared presence contract for planner
fast paths, resolver results and engine dispatch. It includes root `required`
fields and `dependentRequired` fields triggered by supplied arguments. Duration
fields in the timer schema require `expiry_announcement`; list/cancel calls
without duration fields do not. Incomplete calls cannot be directly executed.

`retry_arguments(args)` identifies a repeated effect across model turns; its default retains every argument. Complete valid timer creation uses its canonical rounded duration and sanitised label as this identity, excluding expiry wording; incomplete or invalid requests retain their full arguments. Dispatch and recorded history retain the complete arguments. Schema `writeOnly` properties are omitted from model-facing completed requests, without changing execution or request/result identity.

`bind_retry_identity(current, previous)` can bind non-identical identities to one recorded effect. The default declines binding. Timer binding retains named ownership while distinguishing different non-empty labels. This memo is reply-local, bounded and independent of committed record metadata. Text-mode tool descriptions use JSON objects containing a complete `tool_calls` array; normal answers, questions and blockers use their separate JSON fields.

Constrained reply, step and coverage generation use an owned model-facing schema copy prepared by executable tools. Coverage admits a separate completion object without changing tool-reference scopes. Its single inference retains the existing context, output and deadline limits; unsupported scopes use the ordinary unconstrained path. Coverage proposals enforce the shared required-field and dependency contract before dispatch. The public tool declarations retain their canonical input contract. Metadata-only external tool descriptions retain their supplied schema.

## Optional declared record relationships

A tool can supply `x-jarvis-records` inside its canonical `inputSchema`:

```json
{
  "discovery": 0,
  "collection": {"source": "result", "path": "/remaining_entries", "complete": true},
  "actions": [
    {"purpose": "List all saved entries", "arguments": {"operation": "list"}, "selector": null},
    {"purpose": "Delete selected entries", "arguments": {"operation": "delete"}, "selector": "labels"}
  ]
}
```

`discovery` indexes an action of the same tool whose fixed arguments perform a
read-only namespace lookup. `collection.source` is `result` (raw JSON) or
`result_metadata`; `path` is a local JSON pointer to its record array.
`complete` truthfully declares whether such arrays describe the full namespace.
Scalar string/integer records are their own identifiers; object records use an
action's `identity` pointer and the collection's `display` pointer. Both pointers
default to the whole value. Optional `reference_kind` admits matching raw
resource references through the same identity/display pointers.

Every action has a non-empty purpose, fixed public argument variant and a
nullable selector property. Selectors declare an inline `string`, `integer` or `array` type; item constraints
and local definition references still apply to the copied identifiers.
Actions with free generated parameters are unsupported. Declarations and
compiled calls must satisfy the canonical public schema; unsupported reference
scopes decline this path. The owner declares the complete applicable capability
set and tests discovery and mutation effects. Code cannot infer these
relationships safely from operation names or prose.

Built-ins and discovered MCP tools preserve the same extension through native
and text presentation. The record compiler derives its choices and operations
from that definition. See `../reply/record_binding.spec.md` for interpretation,
execution, budgets and evidence rules. A metadata-free or mixed catalogue uses
the ordinary tool path, without a claimed declared-record guarantee.
