# Graph Memory System (v2) Specification

## Overview

A self-organising node graph that dynamically structures memories by topic relevance. Replaces the flat daily-summary model with a hierarchical tree where nodes auto-split when they grow too large and summaries cascade upward.

Three fast-access entry points — **recent nodes**, **top nodes**, and **root node** — ensure the most relevant memories are always reachable without exhaustive search.

## Data Model

### MemoryNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID string | Unique identifier (root node has id `"root"`) |
| `name` | string | Human-readable label |
| `description` | string | 1-2 sentences used by traversal to decide which branch to explore |
| `data` | string | The actual memories held at this node |
| `parent_id` | UUID or null | Back-reference (null for root) |
| `access_count` | int | Total accesses (for top-nodes ranking) |
| `last_accessed` | ISO 8601 | For recent-nodes ranking |
| `created_at` | ISO 8601 | When the node was created |
| `updated_at` | ISO 8601 | Last modification time |
| `data_token_count` | int | Cached token estimate (len/4 heuristic) |

### Storage

SQLite table `memory_nodes` in the same database as the diary system. Schema is initialised automatically on first access. The root node is created if absent.

### Entry Points

| Entry Point | Query | Purpose |
|-------------|-------|---------|
| Recent nodes | Last N accessed (excl. root) | Fast path for ongoing conversations |
| Top nodes | Highest decayed access score (excl. root) | Core knowledge domains |
| Root node | Single root | Full graph traversal for novel queries |

## Core Operations

### Create

New nodes are created with a name, description, optional data, and a parent_id (defaults to root). Token count is computed on creation.

### Read

Nodes can be fetched individually, as children of a parent, as a subtree (nested dict), or as graph data (flat nodes + edges for visualisation).

### Update

Any combination of name, description, and data can be updated. Token count is recomputed when data changes. `updated_at` is always refreshed.

### Delete

Any node except root can be deleted. Children are orphaned (parent_id set to NULL via FK). The UI should warn before deleting nodes with children.

### Touch

Increments `access_count` and updates `last_accessed`. Called automatically when a node is viewed in the UI or retrieved during query traversal.

### Access Decay

All ordering by access frequency uses a **time-decayed score** computed at query time: `access_count / (1 + age_days / half_life)`. This is hyperbolic decay — a node's effective score halves every `DECAY_HALF_LIFE_DAYS` (default 14) since its last access. The raw `access_count` is never modified, so changing the half-life retroactively reweights all nodes. This applies to `get_top_nodes`, `get_children`, `get_all_nodes`, and `search_nodes` tie-breaking.

### Search

- **search_nodes(query, limit)** — Keyword search across name, description, and data fields. Case-insensitive LIKE matching; nodes matching more keywords rank higher. Excludes root. Touches matched nodes for access tracking.
- **find_node_by_name(name, parent_id)** — Exact name match (case-insensitive), optionally scoped to a parent node. Excludes root when no parent specified.

## Tree & Graph Queries

- **get_subtree(node_id, max_depth)** — Nested dict for tree sidebar
- **get_ancestors(node_id)** — Path from root to node (breadcrumb)
- **get_graph_data(root_id, max_depth)** — Flat {nodes, edges} for canvas rendering. Each node includes depth and has_children flags.

## Auto-Split (Natural Reduction)

Triggered automatically when `data_token_count > SPLIT_THRESHOLD` (1500 tokens) after a write. Auto-split is the system's primary consolidation mechanism — it's where temporal events get distilled into patterns and the tree structure deepens organically.

1. LLM analyses the node's data and proposes 2-5 child categories
2. Each fact is assigned to exactly one child
3. **Consolidation during split**: duplicate facts are merged, and repeated similar activities across different dates are consolidated into patterns (e.g. "ate sushi on Mon, ate sushi on Thu" → "regularly eats sushi"). Single occurrences are kept as-is. Date context is preserved only for significant life events.
4. Child nodes are created under the split node
5. Parent data is cleared; parent description updated to a summary

This means the tree depth itself encodes a temporal→enduring spectrum: surface-level nodes hold recent raw facts, deeper nodes hold patterns that survived multiple split cycles.

Split quality safeguards:
- Minimum 2 categories required (abort if LLM proposes fewer)
- Each category must have at least one fact
- If the split fails (LLM error, bad JSON), the node retains its data and the next write retries

## Auto-Merge (Future — requires LLM integration)

When all children collectively hold < MERGE_THRESHOLD (200 tokens):

1. Collapse children's data back into parent
2. Delete child nodes
3. Update parent description
4. Cascade summaries upward

## Housekeeping (Future)

Periodic process that:
- Promotes buried-but-hot nodes (high access, depth > 3)
- Compresses cold branches (no access in > Y days)
- Merges sparse subtrees
- Validates parent summaries

## LLM Integration

The graph memory system is fully automatic — no tool calls required. It integrates at two points in the existing pipeline.

### Automatic Writes (via `graph_ops.py`)

Piggybacks on the existing diary update flow in `conversation.py`:

1. After a successful diary update, the conversation summary is passed to `update_graph_from_dialogue()`
2. **Extract**: LLM extracts facts that reveal something about the user as a person (third-person statements). Activities, meals, preferences, and situations are kept. Assistant interactions that reveal nothing about the user are filtered (e.g. "asked for the time", "requested news"). Requests that imply an interest are reframed (e.g. "asked about boxing venues" → "interested in boxing"). The diary entry date is included for temporal context. Patterns and consolidation emerge through auto-split.
3. **Traverse**: Each fact is placed in the best-fitting node using the three entry points:
   - **Recent nodes** — checked first; follows conversational momentum
   - **Top nodes** — checked second; matches frequently accessed knowledge domains
   - **Root traversal** — greedy top-down descent; LLM picks the best child at each level, or stops at the current node if none fit
4. **Append**: The fact is appended to the chosen node's data
5. **Split**: If the node now exceeds `SPLIT_THRESHOLD`, auto-split is triggered

Cold start: everything goes to root until enough data accumulates for the first auto-split. The tree structure emerges organically.

LLM failure at any step is non-fatal — the diary update still succeeds, and the graph simply misses that cycle.

### Automatic Reads (via enrichment in `engine.py`)

At the start of each reply cycle, the reply engine enriches the system prompt with graph context:

1. **Keyword search**: Uses the same keywords already extracted for diary search to find matching graph nodes (up to 5 results with data previews)
2. **Recent nodes**: Includes 2-3 recently accessed nodes for conversational continuity
3. Results are injected as "Stored knowledge about the user" — separate from diary history to preserve provenance

No tool calls needed. The LLM sees relevant graph memories as part of its system context.

Controlled by `memory_enrichment_source` config:
- `"all"` — both diary and graph enrich replies
- `"diary"` — only diary (conversation summaries) used for enrichment
- `"graph"` — only graph (structured knowledge) used for enrichment

Default is `"diary"` — graph enrichment can be enabled once tested. Both systems always receive writes regardless of this setting.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SPLIT_THRESHOLD` | 1500 | Tokens before auto-split |
| `MERGE_THRESHOLD` | 200 | Tokens below which children collapse |
| `RECENT_NODES_COUNT` | 10 | Recent nodes to surface |
| `TOP_NODES_COUNT` | 15 | Top nodes to surface |
| `TOP_NODES_WINDOW_DAYS` | 30 | Legacy — kept for API compat, no longer used for filtering |
| `DECAY_HALF_LIFE_DAYS` | 14 | Days until a node's access score halves |
| `MAX_TRAVERSAL_DEPTH` | 8 | Safety limit on graph traversal |
| `SUMMARY_MAX_LENGTH` | 300 | Max chars for node description |
| `memory_enrichment_source` | `"diary"` | Which system enriches replies: `"all"`, `"diary"`, or `"graph"` |

## UI: Memory Viewer Integration

The graph explorer appears as the **Memories (v2)** tab in the memory viewer, positioned between the Diary and Meals tabs.

### Three-Panel Layout

1. **Left sidebar — Tree navigator**: Collapsible tree showing the full hierarchy. Clicking a node selects it in both the tree and the graph canvas. Shows child count badges.

2. **Centre — Graph canvas**: Interactive HTML5 Canvas with radial tree layout. Supports pan (drag), zoom (scroll wheel), and click-to-select. Toolbar provides zoom in/out, fit-to-view, add-node, and import-from-diary actions. Node size reflects access count. Selected node is highlighted with accent glow.

3. **Right sidebar — Node detail**: Shows breadcrumb path, name, description, metadata (accesses, tokens, last seen, children count), stored data, children list, and action buttons (edit, add child, delete).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/nodes` | Graph data (nodes + edges) for canvas |
| GET | `/api/graph/tree` | Nested tree structure for sidebar |
| GET | `/api/graph/node/<id>` | Single node + children + ancestors |
| POST | `/api/graph/node` | Create new node |
| PUT | `/api/graph/node/<id>` | Update node fields |
| DELETE | `/api/graph/node/<id>` | Delete node (not root) |
| GET | `/api/graph/recent` | Recently accessed nodes |
| GET | `/api/graph/top` | Most frequently accessed nodes |
| GET | `/api/graph/stats` | Node count |
| POST | `/api/graph/import-diary` | Import all diary summaries into graph (streaming NDJSON) |

### Import from Diary

The graph toolbar includes an "Import from Diary" button (📥) that bootstraps the graph with existing diary data. This is a one-time migration path so users don't lose their accumulated memories when switching from diary-only to graph enrichment.

The endpoint streams NDJSON progress events (`start`, `progress`, `complete`, `error`) so the UI shows real-time feedback. Each diary summary is processed through the standard `update_graph_from_dialogue()` pipeline (extract → traverse → append → split). Failures on individual summaries are non-fatal — the import continues with the remaining entries.

## Relationship to Existing Systems

The graph memory system lives alongside the existing diary system (conversation_summaries + FTS + vector search). It shares the same SQLite database but uses its own table. The diary system remains the primary memory system for now; the graph is a v2 system being built in parallel.

Users can import existing diary data into the graph via the "Import from Diary" button in the Memory Viewer. This processes all historical summaries through the extract-and-place pipeline, building the graph structure organically.

## Privacy

All data is stored locally in the user's SQLite database. No data leaves the device. The graph store has no network dependencies.
