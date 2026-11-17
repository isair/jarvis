# Declared record binding

Optional `x-jarvis-records` metadata belongs to the tool's canonical input schema.
It lets the shared voice/chat assistant bind existing records without guessing
stored identifiers or adding an interpretation context. Tools return raw data.
The tool registry spec defines the declaration; owners must verify its truth
against their backend effects. This contract does not prove arbitrary natural
language meaning or the safety of a dishonest tool declaration.

When every routed non-control tool declares record metadata, the engine reserves
the declared path. Valid supported contracts use the existing first reply
generation to interpret an ordered request plan instead of
raw API arguments. The ordinary planner and query-driven memory enrichment are skipped; bounded
dialogue/resource references and the warm profile remain available. The same generation can
answer, ask a question, report a blocker, dismiss the conversation or request
capability discovery through the canonical control tools. Mixed and metadata-free routed
catalogues use their ordinary generation path. Invalid, unsupported or oversized
fully declared inputs block unadmitted record calls instead of falling back to
raw argument generation. Metadata-free tools have no
claimed deterministic identity guarantee.

Interpretation uses CHAT with thinking disabled, zero temperature, 8,192 context
tokens, at most 512 output tokens and the lesser of 20 seconds and the configured
chat deadline. The complete request, catalogue, reference context and current
results share the existing 6,000-byte input budget with whole prior observations.
Overflow cannot admit a raw replacement record call. The five-intent plan cap
and existing reply turn budget bound execution. Each actual tool call consumes
an ordinary reply slot. Cancellation gates inference and every dispatch.

The model selects declared action purposes and describes targets in the source
language. Target classification and source fields precede the English paraphrase
in the same bounded structured output. It cannot generate operation variants or choose a stored identifier.
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
complete a later requested read. A separately requested discovery read executes
afresh after preceding effects, without bypassing mutation retry protection.

A reply-local plan is bound to its request, reference context and exact catalogue.
Code recompiles it after each result without another interpretation call.
The normal dispatcher retains permissions, retries, cancellation and paired
request/result history. Final delivery checks the complete declared plan against
actual current effects, including the last available execution slot. An unresolved
plan accepts an explicit question/blocker or uses the existing execution blocker;
it cannot deliver a completion claim or invoke a recovery/digest inference. Raw model record calls cannot bypass the compiler in
this mode. Catalogue expansion retains ordinary access to newly discovered
capabilities; the original bound record tools still require admitted calls.

Coverage recovery uses the same interpretation and compiler as a replacement
for its existing inference, under its existing deadline and caps. Invalid
contracts, unsupported schema scopes and exceeded input budgets admit no
coverage call from a fully declared catalogue. The
metadata-free diagnostic retains its original effect assertions and is distinct
from declared-contract qualification.

Successful declared results retain a derived identity projection beside rendered
conversation history. It contains only per-action identifier/display pairs,
completeness and a canonical schema digest, not unrelated result fields or request
arguments. Its complete JSON is capped by the existing 2,000-byte identity budget.
Both planned and ordinary dispatch produce it from the same declaration. History
scrubs it; any altered projection is discarded and cannot supply substitute
identifiers. The configured per-entry carryover budget drops oversized
projections atomically. A complete projection can survive prose-only truncation;
partial prose cannot supply additional names. Qualified request/result matching,
source names, schema identity and the shared aggregate packet limits still apply.

Projection producers enforce the existing 6,000-byte raw JSON input bound before
decoding. Metadata collections and resource-reference counts are rejected before
iteration when they cannot fit the 2,000-byte projection budget. Individual
identifier/display strings and aggregate encoded pairs are checked incrementally;
all action projections share the complete receipt cap. Overflow discards the
whole projection. Deduplication is linear in the admitted record count.
Successful metadata-only results retain their success, factual presentation and
identity projection through both ordinary and planned result producers.

An untargeted collection classification has null literal, naming-marker and topic
fields. Literal classifications require a non-empty literal target. Descriptive
classifications require a non-empty topic and null literal/naming-marker fields.
Contradictory or missing target data makes the plan invalid before dispatch. The
same existing interpretation retains each explicitly requested operation in order
and separates distinctive topic spans from generic record-category nouns.

Voice interpretation receives the same captured, redacted and fenced
`SpeechContext` as ordinary reply generation, alongside dialogue/resource
references in its existing reference packet. Relevant speech can resolve a
subject or pronoun; only the current request authorises operations. Topics may
copy an exact span from that reference, while literal naming claims remain
anchored to the current request. The complete packet retains its 6,000-byte cap
and request-local fingerprint. Raw ambient speech is not conversation history;
only actual tool request/result observations enter carryover.
