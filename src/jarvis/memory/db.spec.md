# Diary retrieval

Meal description deletion uses one parameterised SQL statement. It deletes a
record only when exactly one stored description matches literally, including
case. Duplicate descriptions preserve all records and require a meal ID.
Description deletion commits on success and rolls back a failed write.

`Database.search_hybrid` searches conversation summaries with FTS5 and optional
vector candidates. Both Python/FAISS and sqlite-vss use the same fusion routine.

- Candidate pools contain up to `max(top_k * 3, 50)` hits per source.
- Lower vector distance and more negative FTS5 BM25 rank first in their own lists.
- Weighted reciprocal rank fusion uses `0.6 / (60 + vector_rank)` plus
  `0.4 / (60 + keyword_rank)`. Ranks start at one; ties share a competition rank.
- A missing source contributes zero. Raw BM25 values and vector distances are
  never added or compared across sources.
- Only live summaries are returned, ordered by descending fused score then ID.
- Keyword normalisation retains Unicode word characters across scripts and
  strips punctuation. Generated FTS operators, including keyword OR queries,
  retain their meaning. Matching follows the database FTS tokenizer.
- Without a query vector, keyword results retain ascending BM25 order.
- Without a usable keyword query, the most recent summaries are returned.

`evals/test_hybrid_retrieval.py` measures lexical and semantic recall@3 on a
controlled corpus. It exercises ranking, not a particular embedding model.

Each `(date_utc, source_app)` summary retains its row ID when its text, topics or
timestamp are updated. Existing vector references remain attached when an
optional refresh is unavailable. Full-text update triggers replace the indexed
terms, and a failed text write rolls back and releases the writer lock while
preserving the committed summary and its references.

Opening a diary rebuilds its full-text index from live summaries once, tracked
in `diary_index_migrations`. The rebuild and its marker commit together under a
SQLite writer transaction shared across database owners. A failed rebuild rolls
back without a marker, allowing a later open to retry. Diary text and vector
rows are unchanged by the rebuild; orphaned vectors are not reassociated.

The database maintains indexes on `meals.ts_utc` and
`conversation_summaries.date_utc` for the meal time-range and summary date
retrieval paths. They are created idempotently whenever a database is opened,
including databases created by earlier versions.

`upsert_summary_embedding` writes the sqlite-vss vector and its summary mapping
in one transaction under the database lock. A failed write rolls back the index
transaction and releases the writer lock; previously committed diary text and
vector mappings remain readable. Callers commit diary text before refreshing
the optional index.

`get_summary_embedding_text` captures the current summary and topics, joined by
a space, before embedding inference. `upsert_summary_embedding` requires that
exact source text. It accepts a refresh only while the live row has the same
embedding text and returns `None` for a superseded or deleted row. The content
comparison and persistence are atomic across database owners: sqlite-vss uses
a writer transaction, while file-backed Python/FAISS indices use a conditional
insert from the live summary. In-memory diaries compare and publish under their
owner's lock. No lock spans embedding inference. Rejected refreshes preserve
the current vector and are not reported as successfully refreshed by maintenance.

Generic vector-store `add_vector` writes are unconditional. Diary-specific
`add_summary_vector` writes require the captured source text and publish an
in-memory candidate only when the conditional persistent write succeeds.

Python and FAISS indices are shared only by active owners of the same resolved
database file; FAISS dimensions also belong to the in-memory index identity.
Persistence holds one current vector per summary. Re-embedding a summary with
a different dimension replaces its persisted vector; an index with another
dimension cannot load that vector. Distinct
database files and independent `:memory:` databases never share candidates or
vector writes. Weak ownership releases an unused index, so a later owner loads
its persisted file afresh. File-backed stores retain the resolved absolute path
for every write, independent of later working-directory changes. Factory construction is serialised to prevent two
active owners of one file from receiving divergent indices.

Local Python and FAISS indices accept non-empty, finite, one-dimensional
embeddings with a non-zero direction. Validation precedes replacement and
persistence, so an invalid refresh preserves the current memory vector.
Normalisation retains the direction of large and small finite values without
overflow or underflow. FAISS writes must match the index dimension. Python
search considers only stored vectors matching the query dimension. Unusable
queries contribute no semantic candidates, preserving keyword retrieval.
Loading skips individual malformed persisted vectors and retains valid rows.

File-backed Python and FAISS vector refreshes persist successfully before
publishing an in-memory replacement or candidate. A failed SQLite write raises
to the caller, preserves the previous index and releases the writer connection
for a later retry. Committed diary text remains available to keyword search.
Python `:memory:` indices retain vectors in memory without a persistent file.

`has_vector_store` reports availability of sqlite-vss or a local FAISS/Python
index. Diary saves, summary rewrites and topic optimisation use this capability
alongside the configured embedding model to refresh semantic candidates. An
unavailable model or index skips embedding inference and preserves keyword
retrieval.
