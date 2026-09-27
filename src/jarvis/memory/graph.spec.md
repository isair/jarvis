# Editable knowledge graph

## Purpose and ownership

`GraphMemoryStore` holds editable topic nodes in the local SQLite database.
It supports the Knowledge tab in the memory viewer and explicit graph import
and consolidation actions. Automatic dialogue extraction writes individual
source-grounded facts through `FactStore`; graph node text has unverified
provenance and is labelled as such when included in a warm profile. Graph
editing does not silently modify, duplicate or supersede individual facts.

## Shape and preservation

The root node has stable ID `root`. Three fixed non-deletable children have
stable IDs `user`, `directives` and `world`. They support manual organisation
and the graph import tools. The optional reserved `legacy` child holds
uncategorised material and is non-deletable. Its ordinary descendants remain
editable and deletable.

`migrate_legacy_shape()` runs at daemon startup. When the root holds text or
has children outside the fixed branches, it creates the `legacy` child,
moves the root's text there verbatim, and reparents those children. Existing
child IDs, contents, timestamps and descendants are preserved. It does not
delete rows or create a hidden archive. The operation is idempotent. Editing
or deleting a legacy descendant changes the only current copy of its text.
The reserved `legacy` branch itself cannot be deleted, so root knowledge
remains reachable from the viewer.

## Data model

`MemoryNode` contains `id`, `name`, `description`, `data`, `parent_id`,
`access_count`, `last_accessed`, `created_at`, `updated_at`, and a cached
`data_token_count`. Nodes form an adjacency-list tree in `memory_nodes`.
Names and descriptions guide manual organisation; `data` holds free-form
lines. `data_token_count` is estimated as roughly four characters per token.
Foreign-key deletion of an ordinary node orphans its children rather than
deleting them.

## Operations

`create_node`, `update_node`, `delete_node` and `append_to_node` mutate nodes.
Root, the three fixed branches and the reserved legacy branch cannot be
deleted. Description length is capped by `SUMMARY_MAX_LENGTH`. `touch_node`
increments access count and updates the last-accessed time.

`get_children`, `get_recent_nodes`, `get_top_nodes`, `get_all_nodes` and
`search_nodes` use a time-decayed access score, calculated from count and
age. The stored access count remains intact. `search_nodes` uses
case-insensitive keyword matches over name, description and data; it
excludes root. `get_subtree`, `get_ancestors` and `get_graph_data` provide
the viewer's tree, breadcrumb and canvas representations.

`register_graph_mutation_listener` and `unregister_graph_mutation_listener`
manage callbacks receiving `action`, `node_id`, and fixed-branch ancestry.
They run after successful node mutations. Listener errors are logged and do
not roll back a write. Touches do not trigger mutation callbacks.

## Explicit graph tools

`graph_ops.py` supports the memory viewer's user-triggered diary import and
whole-graph consolidation. Import extracts candidate knowledge from a diary
summary, classifies it into User, Directives or World, traverses inside that
branch, deduplicates, merges and may split oversized nodes. It is separate
from the automatic source-grounded fact path. The imported content retains
its graph provenance, and recall labels it unverified. Consolidation
rewrites populated node text using the configured local model; it does not
rewrite fact-ledger rows.

The picker can use a fast model override; when absent it uses the chat
model. `find_best_node(..., branch_root_id=...)` descends only inside the
chosen branch. Exact deduplication normalises Unicode with NFKC, casefolding
and whitespace folding. Merge and split failures leave the original node
text available. `SPLIT_THRESHOLD` is 1500 estimated tokens. Sparse-node
merge and housekeeping are not invoked automatically.

## Viewer

The Knowledge tab has a tree navigator, canvas and node detail panel. Its
HTTP API includes graph tree/nodes, node CRUD, recent/top/stats, presets,
diary import and consolidate-all. Import and consolidation stream progress
events. The Facts panel uses the separate `FactStore` API described in
`facts.spec.md`.

## Privacy

The graph is stored locally. No graph store operation requires a remote
service. The optional LLM operations use the configured backend, which can
run wholly on device. No duplicate archive keeps deleted graph text hidden.
