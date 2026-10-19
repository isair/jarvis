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

`has_unclassified_root_content()` reports non-empty root text or children outside
fixed branches. It reads root text in incremental SQLite chunks of at most 4,096
bytes and recognises Unicode whitespace. Inspection preserves every saved row,
including root data, node identities, parent relationships and timestamps.
Daemon startup reports these notes for explicit organisation in the viewer.
Inspection failures are non-fatal and close the inspection connection.

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

`read_node_text` returns at most the requested number of text characters.
It uses incremental SQLite BLOB reads of at most 4,096 bytes, checks activity
between reads, and decodes the database encoding incrementally. Embedded NULs
and complete Unicode characters are preserved without loading the whole node.
`get_child_ids` returns at most the requested number of child identifiers in
access-score order, without loading their text. These bounded read methods
check caller activity before and after reading and every 1,000 SQLite
virtual-machine operations. A caller exception interrupts the read and
propagates unchanged; the progress handler is removed on every exit. Each
bounded read owns a savepoint spanning identity lookup through BLOB closure,
so concurrent replacement cannot substitute another node with a reused row ID.
Cleanup releases only that savepoint and preserves any caller-owned transaction.

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
