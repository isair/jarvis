# Diary retrieval

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

`upsert_summary_embedding` writes the sqlite-vss vector and its summary mapping
in one transaction under the database lock. A failed write rolls back the index
transaction and releases the writer lock; previously committed diary text and
vector mappings remain readable. Callers commit diary text before refreshing
the optional index.

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
