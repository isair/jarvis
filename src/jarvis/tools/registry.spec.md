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
