# Tool execution results

`ToolExecutionResult` carries success, reply data, an optional error, an optional
missing-context field and structured resource references. References are tuples
of records with `kind`, `id` and `label`. Tools supply IDs from completed local
operations, never from model-generated summaries or inferred database ordering.

The reply engine retains references only for successful results. Each result
keeps at most eight references. IDs remain complete: integers are bounded to
128 bits; non-empty string IDs to 160 characters. Kinds are non-empty strings
of at most 80 characters; labels are bounded to 200 characters. Malformed
records are omitted. References are rendered as data beside the effective
result prose, independently of whether prose digestion succeeds, fails or is
disabled. They are also retained as internal tool-message metadata.

Dialogue tool carryover copies and scrubs reference strings with the same
secret scrubber as tool arguments. It follows the conversation's existing tool
turn retention and reset rules and is excluded from diary text. Metadata
survives prose truncation. Backend message sanitisation removes internal fields
from the wire; models receive references through the rendered data block.

Planning receives at most the latest eight successful references in chronological
order, separately from its bounded user/assistant dialogue selection. Unknown or
failed tool outcomes cannot supply reference identity. The planner resolves the
requested resource using this evidence and uses its ID for follow-up operations.
Unresolved identity requires retrieval or clarification. No deletion tool infers
"latest" or relaxes its own ambiguity checks.
