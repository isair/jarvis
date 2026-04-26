# Knowledge Graph Specification

## Overview

A self-organising node graph that stores the assistant's accumulated world knowledge — anything learned during conversations that it wouldn't already know from training data. This includes user-specific facts, real-world discoveries (opening hours, local businesses), practical knowledge (recipes, solutions), and current events. The diary records *what happened*; the knowledge graph records *what was learned*.

The graph dynamically structures knowledge by topic relevance using a hierarchical tree where nodes auto-split when they grow too large. Three fast-access entry points — **recent nodes**, **top nodes**, and **root node** — ensure the most relevant knowledge is always reachable without exhaustive search.

## Fixed Top-Level Branches

On first bootstrap the graph seeds three non-deletable branches under root, defined in `FIXED_BRANCHES` in `graph.py`:

| Branch ID | Name | Purpose |
|-----------|------|---------|
| `user` | User | Everything about the user: identity, location, tastes, habits, history |
| `directives` | Directives | Imperatives the user issued at the assistant: reply style, tone rules, standing instructions |
| `world` | World | External facts the assistant has learned: discoveries, practical knowledge, current events |

These branches are created idempotently via `INSERT OR IGNORE` on stable IDs. The structure is intentionally shallow and purpose-driven — splits deepen each subtree over time, but the top layer stays fixed so the **warm profile** (see below) has a stable shape.

No Other branch: the extractor defaults unknown classifications to `user`. A fact that genuinely belongs nowhere should not be stored.

### Legacy-Shape Migration (destructive)

`GraphMemoryStore.migrate_legacy_shape()` checks the on-disk graph against the expected shape at daemon start-up. The graph is considered non-conforming if root has any direct child that isn't one of the fixed branches, or if root's own `data` column is non-empty (cold-start writes that landed on root before the taxonomy existed). In either case the entire `memory_nodes` table is wiped and root + the three fixed branches are re-seeded.

Why destructive: pre-taxonomy nodes sitting under root would remain invisible to the warm profile forever. Carrying them as dead weight is worse than a clean slate. The diary is untouched, so users can re-populate via "Import from Diary" in the memory viewer once the wipe completes. Knowledge nodes are in beta — the structure and classification are now stable but the extractor quality is still being tuned.

Called **only** from the daemon start-up path in `daemon.main()`. The memory viewer and reply engine instantiate `GraphMemoryStore` without triggering the migration, so a mid-session open never wipes anything.

### Branch-Pinned Traversal

`find_best_node(..., branch_root_id=...)` skips the recent/top entry points and descends from the given branch root only. This prevents cross-branch contamination when routing extracted facts: a User fact cannot land in the World subtree just because a World node was recently touched.

## Warm Profile

`build_warm_profile(store, *, user_max_chars, directives_max_chars)` returns a `{"user": "...", "directives": "..."}` dict by walking the User and Directives subtrees breadth-first (ordered by each sibling's decayed access score) and concatenating node data up to the char caps. `format_warm_profile_block(profile)` renders it as a labelled system-prompt section using denial-template mirroring (see CLAUDE.md): the headings literally occupy the semantic slot that small-model canonical denials refer to ("INFORMATION THE USER HAS SHARED IN PRIOR CONVERSATIONS", "STANDING INSTRUCTIONS FROM THE USER").

The warm profile is injected into every reply's initial system message (see `reply/engine.py` Step 3.5) unconditionally and query-agnostically — personalisation is the default, not something gated behind a question-detection heuristic. No LLM call is involved in composition; it's a pure SQLite read.

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

Triggered automatically when `data_token_count > SPLIT_THRESHOLD` (1500 tokens) after a write. Auto-split is the system's primary consolidation and pruning mechanism — it's where temporal events get distilled into patterns, common knowledge gets dropped, and the tree structure deepens organically.

1. LLM analyses the node's data and proposes 2-5 child categories
2. Each fact is assigned to exactly one child
3. **Consolidation**: duplicate facts are merged, and repeated similar activities across different dates are consolidated into patterns (e.g. "ate sushi on Mon, ate sushi on Thu" → "regularly eats sushi"). Date context is preserved only for significant events.
4. **Pruning**: facts that the LLM already knows from its training data are dropped. This keeps the graph as a delta from the model's baseline knowledge. When migrating to a newer model with broader training data, subsequent splits will naturally prune more — reducing the graph's memory footprint over time.
5. Child nodes are created under the split node
6. Parent data is cleared; parent description updated to a summary

This means the tree depth itself encodes a raw→refined spectrum: surface-level nodes hold recently ingested knowledge, deeper nodes hold distilled novel knowledge that survived multiple split cycles. Model upgrades naturally shrink the graph as previously-novel facts become common knowledge.

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
2. **Extract + classify**: LLM extracts novel knowledge from the summary and classifies each fact into one of the three fixed branches (`USER` / `DIRECTIVES` / `WORLD`). Output is a JSON list of `{"branch": "...", "fact": "..."}` objects. Rough routing heuristic baked into the prompt: if the user is *telling the assistant how to behave* → DIRECTIVES; if the user is *telling the assistant about themselves* → USER; if the assistant *discovered a fact about the world* → WORLD. Unknown branches default to USER. Requests are reframed as knowledge ("user asked about CEX hours" → "CEX Kensington closes at 6pm on Sundays"). Patterns and consolidation emerge through auto-split.
3. **Traverse**: Each fact is placed in the best-fitting node using branch-pinned descent from its tagged branch root (recent/top shortcuts are skipped so cross-branch contamination is impossible):
   - **Recent nodes** — checked first; follows conversational momentum
   - **Top nodes** — checked second; matches frequently accessed knowledge domains
   - **Root traversal** — greedy top-down descent; LLM picks the best child at each level, or stops at the current node if none fit
   - **Picker model**: `update_graph_from_dialogue` / `find_best_node` / `_llm_pick_best_child` accept an optional `picker_model` override. Callers (daemon, memory viewer's diary-import endpoint) resolve it via `resolve_tool_router_model(cfg)` so the best-child classification runs on the small warm router model instead of the big chat model. When `picker_model` is `None` the picker falls back to `ollama_chat_model`.
4. **Dedupe (fast-path)**: Before any LLM call, `GraphMemoryStore.node_contains_fact` compares the fact against each line of the chosen node's data under Unicode-aware folding (`unicodedata.NFKC` + `str.casefold` + whitespace collapse), so ASCII casing, locale quirks (Turkish `İ`/`ı`, German `ß`/`ss`), and incidental whitespace don't cause false negatives. Exact matches are skipped, **not** reported as newly learned, and do **not** touch the node's access score (a re-extraction isn't fresh reinforcement). The merge step below would also collapse re-extractions, but cumulative daily summaries re-emit the same lines often enough that catching them with a cheap SQL read avoids a flood of small-model calls — semantically equivalent, just faster. Skips are still counted: `update_graph_from_dialogue` returns a `GraphUpdateResult(stored, skipped)` so the CLI can log "nothing new (N duplicates skipped)" on all-duplicate flushes; silencing that line would make the memory pipeline look broken. The check only covers the picker's chosen node, so a later flush that routes the same fact to a different node within the branch can still leak through — caught by the merge step on that node instead.
5. **Merge** (batched per node): `merge_node_data(store, node_id, new_facts: list[str], ...)` sends the existing node data + **all** new facts routed to that node in this flush to the picker model and asks it to produce a clean, consolidated, contradiction-free fact list, which is written back as the node's full `data`. The orchestrator groups the flush by `node_id` first so a 5-fact flush against the User node fires **one** rewrite that incorporates all five facts, not five separate rewrites of the same `data`. The call returns a `MergeResult(success: bool, incorporated_indices: list[int])` so the orchestrator can report only the facts that actually survived as new lines (consolidated-out facts aren't claimed as "newly stored"). One LLM call subsumes four behaviours: (a) **supersession** — contradictions, negations, and same-attribute updates drop the old line ("user does not need a daily check-in" replaces both "user has a need for a daily check-in" and the same need framed as an interest); (b) **near-duplicate dedupe** — different wordings of the same fact collapse to one canonical phrasing; (c) **consolidation** — repeated daily activities fold into patterns ("ate sushi on Monday", "ate sushi on Thursday" → "regularly eats sushi"); (d) **meta-narrative pruning** — lines that narrate the assistant's own behaviour, capabilities, or denials ("The assistant is unable to navigate to a web page", "The assistant suggested grilled salmon") are extractor artefacts from earlier prompt versions and get dropped. Counterpart to the extractor's BANNED FACT FORMS list: the extractor blocks them at write-time, the merge prompt scrubs the historical leftovers a `consolidate-all` sweep can then surface. Genuine user-issued imperatives ("Always reply in British English") are not meta-narrative and survive. Independent facts coexist (a "user ate a Big Mac" line does not silently drop "user is vegetarian"; the contradiction stays visible). Because the latest prompt always rewrites the whole node, updated conventions propagate to old data without a separate migration. **Hallucination guard**: the rewrite is rejected if it returns more lines than `len(existing) + len(new) + 2` — a runaway model can't quietly inflate the node. Fail-open: empty/cold node, LLM error, parse failure, oversized rewrite, or an empty rewrite all fall back to plain `append_to_node` for each new fact so they still land — a contradiction is recoverable, a silent wipe or hallucinated bloat is not.
6. **Split**: If the merge or fallback append pushes the node past `SPLIT_THRESHOLD`, auto-split is triggered

Cold start: each fact lands directly on its tagged branch root (User / Directives / World) until enough data accumulates there for the first auto-split. The tree structure emerges organically under each branch.

LLM failure at any step is non-fatal — the diary update still succeeds, and the graph simply misses that cycle.

### Automatic Reads (via enrichment in `engine.py`)

At the start of each reply cycle, the reply engine enriches the system prompt with graph context:

1. **Question-driven**: Graph enrichment runs only when the query generator produced implicit personal questions. Utility queries (time, maths) and queries whose context is already live skip the graph entirely — the knowledge graph is a Q&A index, not a topic index.
2. **Question search**: Questions are joined, stop-worded, and used to find matching nodes (up to 5 results with data previews).
3. Results are injected as "Stored knowledge about the user" — separate from diary history to preserve provenance.

No tool calls needed. The LLM sees relevant graph memories as part of its system context.

Controlled by `memory_enrichment_source` config:
- `"all"` — both diary and graph enrich replies
- `"diary"` — only diary (conversation summaries) used for enrichment
- `"graph"` — only graph (structured knowledge) used for enrichment

Default is `"all"` — both channels enrich replies. The graph has graduated from alpha to beta with the purpose-driven taxonomy and warm profile now always-on, so the default flipped from `"diary"` to include graph recall too. Both systems always receive writes regardless of this setting.

Note: the always-on warm profile (User + Directives injected on every turn) is separate from query-driven enrichment. Warm profile covers "who the user is"; enrichment covers "what the user has said/seen about this specific topic". The graph contributes to both.

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
| `memory_enrichment_source` | `"all"` | Which system enriches replies: `"all"`, `"diary"`, or `"graph"` |

## UI: Memory Viewer Integration

The graph explorer appears as the **Knowledge** tab in the memory viewer, positioned between the Diary and Meals tabs.

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
| GET | `/api/graph/stats` | Node count and total data tokens (`total_tokens = 0` means the graph holds no knowledge) |
| POST | `/api/graph/import-diary` | Import all diary summaries into graph (streaming NDJSON) |
| POST | `/api/graph/consolidate-all` | Self-consolidate every populated node (streaming NDJSON) — runs the merge LLM with no new facts on each node so updated conventions and supersession rules apply to historical data |

### Import from Diary

The graph toolbar includes an "Import from Diary" button (📥) that bootstraps the graph with existing diary data. This is a one-time migration path so users don't lose their accumulated memories when switching from diary-only to graph enrichment.

The endpoint streams NDJSON progress events (`start`, `progress`, `complete`, `error`) so the UI shows real-time feedback. Each diary summary is processed through the standard `update_graph_from_dialogue()` pipeline (extract → traverse → append → split). Failures on individual summaries are non-fatal — the import continues with the remaining entries.

### Consolidate All (🧹)

The toolbar's 🧹 button walks every populated node and calls `merge_node_data` with an empty `new_facts` list, prompting the picker model to re-apply the latest supersession/dedupe/consolidation rules to data that landed before those rules existed (or before the prompt was tightened). Like Import from Diary, it streams NDJSON progress events. Per-node failures are non-fatal so a single bad node can't abort the sweep. The UI confirms before starting and reports the total line-count delta on completion.

## Relationship to Existing Systems

The graph memory system lives alongside the existing diary system (conversation_summaries + FTS + vector search). It shares the same SQLite database but uses its own table. The diary system remains the primary memory system for now; the graph is a v2 system being built in parallel.

Users can import existing diary data into the graph via the "Import from Diary" button in the Memory Viewer. This processes all historical summaries through the extract-and-place pipeline, building the graph structure organically.

### Diary Summariser Hygiene

Graph extraction ingests diary summaries, so the graph inherits whatever corruption the summary contains. Summariser hygiene rules (no deflection narration, attribution preservation, topic separation) are documented in [`summariser.spec.md`](summariser.spec.md).

## Privacy

All data is stored locally in the user's SQLite database. No data leaves the device. The graph store has no network dependencies.
