# Declared record binding

Optional `x-jarvis-records` metadata belongs to the tool's canonical input schema.
It lets the shared voice/chat assistant bind existing records without guessing
stored identifiers or adding an interpretation context. Tools return raw data.
The tool registry spec defines the declaration; owners must verify its truth
against their backend effects. This contract does not prove arbitrary natural
language meaning or the safety of a dishonest tool declaration.

When every routed non-control tool declares a valid supported contract, the
existing first reply generation interprets an ordered request plan instead of
raw API arguments. The ordinary planner is skipped. The same generation can
answer, ask a question, report a blocker, dismiss the conversation or request
capability discovery through the canonical control tools. Other routed
catalogues use their ordinary generation path. Metadata-free tools have no
claimed deterministic identity guarantee.

Interpretation uses CHAT with thinking disabled, zero temperature, 8,192 context
tokens, at most 512 output tokens and the lesser of 20 seconds and the configured
chat deadline. The complete request, catalogue, reference context and current
results share the existing 6,000-byte input budget with whole prior observations.
Overflow cannot admit a raw replacement record call. The five-intent plan cap
and existing reply turn budget bound execution. Each actual tool call consumes
an ordinary reply slot. Cancellation gates inference and every dispatch.

The model selects declared action purposes and describes targets in the source
language. It cannot generate operation variants or choose a stored identifier.
Code copies fixed arguments, validates source attribution for literal names,
and resolves descriptive topics using Unicode word sets against observed
identifiers or display names in the tool's own namespace. A unique match copies
the complete original identifier. Literal identifiers require quotation, path
notation or an adjacent source-anchored naming marker. Semantic interpretation
and the completeness of unquoted names remain model responsibilities.

Unknown or ambiguous identity requires the declared read-only discovery action.
A complete successful snapshot replaces earlier names, including names removed
by a mutation. Failed, truncated and feedback observations supply no identity.
An attempted discovery supersedes prior names; a missing or ambiguous result
blocks mutation and repeated discovery. Partial snapshots can supply positive
identity but cannot establish absence. Prior observations cannot complete a new
request. Completion consumes successful current actions in order, using the
identity evidence available before each effect. A prerequisite lookup cannot
complete a later requested read.

A reply-local plan is bound to its request, reference context and exact catalogue.
Code recompiles it after each result without another interpretation call.
The normal dispatcher retains permissions, retries, cancellation and paired
request/result history. Raw model record calls cannot bypass the compiler in
this mode. Catalogue expansion retains ordinary access to newly discovered
capabilities; the original bound record tools still require admitted calls.

Coverage recovery uses the same interpretation and compiler as a replacement
for its existing inference, under its existing deadline and caps. Invalid
contracts or unsupported schema scopes use the ordinary generic path. The
metadata-free diagnostic retains its original effect assertions and is distinct
from declared-contract qualification.
