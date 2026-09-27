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
- Without a query vector, keyword results retain ascending BM25 order.
- Without a usable keyword query, the most recent summaries are returned.

`evals/test_hybrid_retrieval.py` measures lexical and semantic recall@3 on a
controlled corpus. It exercises ranking, not a particular embedding model.
