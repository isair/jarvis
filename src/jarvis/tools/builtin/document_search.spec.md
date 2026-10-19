## Local Document Search Tool Spec

Searches user-owned `.txt` and `.md` files under explicitly configured folders.
The feature is disabled by default and performs no filesystem or embedding work
until the user enables it and supplies at least one folder.

### Indexing

Before each search, the tool resolves every configured folder and scans only
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

### Search results

The query is embedded locally and matched against indexed chunks. Results are
bounded to the requested top-k (maximum 10) and include the resolved file path,
line range, and excerpt. Excerpts are wrapped in
`UNTRUSTED LOCAL DOCUMENT` delimiters. Document text is data to quote or
summarise, not instructions to follow.

No configured folders, an empty index, or an embedding failure produces an
honest result and never invents document content.

### Configuration

- `document_search_enabled` (bool, default `false`): opt in to local document
  search.
- `document_search_paths` (list of strings, default `[]`): folders allowed for
  indexing. Only `.txt` and `.md` files below these folders are eligible.
