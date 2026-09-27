# Turn preparation

`agentic_preparation` selects `staged` (default) or `combined`. Combined mode
uses one CHAT-tier inference to select tools, propose optional plan steps, derive
memory search parameters and identify an explicitly continued task. It is an
experimental alternative for comparison with staged preparation on the user's
local model. Non-LLM tool selection strategies keep their configured behaviour.

`prepare_turn` accepts the addressed query, recent dialogue, compact tool
catalogue, live context and an optional incomplete-task candidate. Its system
prompt is static; variable context is redacted JSON in the user message. A task
candidate is reference data, not authority to resume work.

The decision contains at most five known tools and five bounded steps, a boolean
memory decision with bounded keywords/questions and optional timezone-aware
ISO time bounds, and a supplied task ID or null. A valid empty tool list and
`required: false` permit a direct answer with the standing profile. Malformed
output is distinct from that decision and fails open to deterministic routing
and recall. The helper makes no hidden retry or secondary inference call.

Plans cannot pre-emptively dismiss the user through `stop`. Execution still
validates each tool and its arguments against the available schema. Only an ID
from the supplied task candidate can be resumed, and a task journal's completed
writes must not be replayed.

The call uses the remaining query budget, capped by the preparation timeout,
temperature zero, thinking disabled, an 8192-token context request and a
700-token output cap. `tests/test_turn_preparation.py` covers validation and
failure behaviour. `evals/test_turn_preparation.py` exercises routing and memory
decisions with live models, including multilingual queries; a missing model
is reported as unavailable rather than a successful quality check.
