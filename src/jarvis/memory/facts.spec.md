# Source-grounded fact memory

`FactStore` keeps durable facts as individual SQLite rows in the same local
database as diary summaries. Each fact has a source reference, an exact redacted
evidence excerpt, source role/channel, observed time, valid interval, owner,
subject, predicate key and status. Facts never depend on a proprietary cloud
service. The graph remains a separate editable collection of unverified legacy
knowledge; new facts are not copied into graph nodes.

All instants are normalised to UTC before persistence and comparison, including
offset and `Z` timestamps. Fact text, source text, evidence and subject fields
pass through structural redaction at the storage boundary.

## Writes

The diary flush captures formatted chunks and structured source messages in
one locked snapshot. It persists redacted, bounded batches in
`fact_pending_batches` before the diary summary call and commit. If the diary
fails, the source queue remains durable and the dialogue saved marker does not
advance; retrying the same snapshot does not duplicate pending batches. Every
nonempty message is covered: long redacted text is segmented,
and segments are packed into durable batches within the extractor's context
budget. The batches contain addressed dialogue text, role, channel and timestamp;
it contains no audio or tool payload. A failed extractor leaves the batch for
retry. Successful extraction deletes the batch and retains only source hashes
and cited evidence excerpts. Duplicate batches and duplicate facts are
idempotent. `process_pending_fact_batches()` processes batches in order within
one total timeout and stops at a failed or unfinished batch. An extractor result
above the per-batch candidate cap is treated as failed before any candidate is
written, leaving the complete source batch for retry rather than dropping its
tail.

The extractor sees the numbered source messages and a bounded list of current
facts. It returns independent facts with an exact evidence quote and source
index. Invalid indices, absent quotes, empty text and invalid categories are
rejected. User ownership and directives require a direct addressed user
utterance. A visibly paired quotation is rejected deterministically, whether
its delimiters sit inside or just outside the cited evidence span. Reported
speech without quotation marks relies on the extractor's direct-versus-reported
classification. Ambient or assistant assertions cannot create a user-owned
fact or directive. Ambiguous ownership stays `unknown`. The model must
classify a statement as direct; attribution is not guessed from a topic or an
assistant claim. The source quote is verified against the source text before
SQLite accepts the fact. Content is redacted again at the storage boundary.

`add_fact()` stores a new assertion. It never infers supersession from recency.
An explicit `supersedes_id` is accepted only for an active fact with matching
kind and owner. The extractor further requires the same subject and predicate
key. `correct_fact()` creates a successor and closes the old fact's valid
interval atomically. `retract_fact()` marks an active fact retracted and
records the reason. Both preserve source history; Retract is not erasure.
Conditional status updates prevent competing clients from creating two active
successors for the same predecessor.
`get_fact_history()` follows the supersession chain, while `get_fact()` and
`list_facts()` expose source evidence to the viewer.

Fact mutations advance a SQLite revision and call registered listeners with
`action`, `fact_id`, `kind` and `ownership`. Listener failures are logged and
cannot prevent a committed write. The reply layer invalidates its warm profile
cache when relevant facts are added, corrected or retracted.

## Retrieval

`search_facts()` combines Unicode FTS5 keyword candidates with optional local
embedding candidates using reciprocal rank fusion. It never adds raw BM25
scores to cosine similarities. Search can filter by observed time, valid time
and source type. Normal recall includes only active facts; historical valid-time
queries may include superseded assertions valid at the requested instant.

`recall_evidence()` retrieves fact and diary candidates and compares their
positions with bounded rank weights. It labels each fact with ownership,
observed date, source type and evidence; diary entries are labelled summaries
and reference-only. It also searches editable legacy graph nodes, including
World, and labels every match unverified. `memory_enrichment_source` selects
`none`, `diary`, `graph` (facts plus legacy graph), or `all`. The optional
embedding timeout is bounded by the caller. It chooses complete evidence lines
within one `max_tokens` context budget. Legacy node text contributes only lines
with Unicode lexical overlap with the query; matching lines are ordered by
overlap and capped per node so broad blobs cannot consume the budget with
unrelated content. Older superseded or retracted facts do
not enter ordinary recall. Historical `as_of` lookup can return assertions
valid at the requested time. If local embeddings are unavailable, FTS retrieval
continues.

`build_fact_warm_profile()` prioritises active user-owned facts and direct user
instructions. It adds whole lines that fit each budget and never cuts a rule
mid-string. Existing User, Directives and Legacy graph node text is surfaced
separately as unverified reference material, read live from the editable graph.
`format_fact_warm_profile_block()` keeps verified facts, standing instructions
and unverified legacy material under distinct headings.

## Viewer and legacy graph

The Facts panel reads `list_facts()`, `get_fact()` and `get_fact_history()` and
uses `correct_fact()` and `retract_fact()` for writes. It labels retraction as
history-preserving. Graph edits and deletes affect only legacy graph nodes; no
hidden fact projection of their text exists. The graph's legacy rehoming keeps
the original text in editable nodes without a duplicate archive. Deleting a
legacy node therefore removes that current graph text.
