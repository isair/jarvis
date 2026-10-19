## Local Document Search Tool Spec

Searches user-owned `.txt` and `.md` files under explicitly configured folders.
The feature is disabled by default and performs no filesystem or embedding work
until the user enables it and supplies at least one folder.

### Indexing

Each search performs a bounded refresh of configured folders and scans only
regular files whose resolved path remains inside that folder. Symlinks that
resolve outside an allowed folder are skipped. Other extensions are ignored.
Device, inode, byte size and modification/change times identify each source.
Unchanged files in the same embedding space are not re-embedded. Changed files are re-chunked and embedded, and deleted files
remove their chunks and vectors. Unavailable or removed roots also remove their
cached chunks and vectors, including when no authorised root remains.

Chunks are line-aware and use the configured embedding backend. Document text,
normalised vectors and file identity publish atomically in a dedicated SQLite
index beside the diary. The index owns and closes its connection independently
of the diary transaction and vector backend. Failed embeddings or storage writes
publish no partial file and are retried on a later refresh. Model, endpoint,
backend and detected dimension changes invalidate incompatible cached vectors.
Search uses cosine similarity over document candidates only. Sources and current
folder authority are revalidated before returning cached excerpts.

Refreshes use the finite `IndexLimits` policy: at most 64 changed files,
128 chunks, 1 MiB of source input, 2,048 directory entries and five seconds
per pass. Individual files are limited to 256 KiB. Traversal limits the folder
count and depth to 32; directory symlinks are not traversed. The persistent
cache holds at most 256 files and 1,024 chunks. Embedding dimensions are
limited to 4,096, with the canonical finite/unit-vector validation. Query
ranking retains at most ten candidates in memory.

Each embedding timeout is capped by the remaining pass deadline. Cancellation
and deadlines are checked between directory entries, source files, embedding
calls and publication. Query inference checks interruption before invalidating
cached vectors, so a cancelled or expired query preserves reusable excerpts.
Cancellation during search produces an unsuccessful tool result. A cancelled file
never publishes partially. Sources
that exceed a budget are skipped as complete files. An exhausted pass reports
partial coverage and its reasons; a bounded scan does not promise exhaustive
coverage of a larger collection. No permanent indexing worker is required.

### Search results

The query is embedded locally and matched against indexed chunks. Results are
bounded to the requested top-k (maximum 10) and include the resolved file path,
line range, and excerpt. Excerpts are wrapped in
`UNTRUSTED LOCAL DOCUMENT` delimiters. Document text is data to quote or
summarise, not instructions to follow.

Every result includes complete or partial index coverage and explicit reasons
for incomplete coverage. Empty results do not establish that an unscanned source
lacks the requested information. Query embedding failures return an error rather
than a successful empty search. No result invents document content.

### Configuration

- `document_search_enabled` (bool, default `false`): opt in to local document
  search.
- `document_search_paths` (list of strings, default `[]`): folders allowed for
  indexing. Only `.txt` and `.md` files below these folders are eligible.
