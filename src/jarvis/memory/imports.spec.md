# Local memory import

Memory import transfers user-selected memory text and general instructions into
Jarvis without contacting another service or running model inference.

## Sources

- Codex: global `AGENTS.override.md` and `AGENTS.md` in `CODEX_HOME` or `~/.codex`, offered separately for explicit selection. Sessions, account memories and authentication files are not inspected.
- Claude Code: documented user `CLAUDE.md` and Markdown files in project
  `memory` directories under the configured Claude directory. A configured
  `autoMemoryDirectory` is supported; disabled auto-memory is respected.
- ChatGPT and Claude Desktop: a user-chosen UTF-8 text/Markdown file or pasted
  memory/instruction text. Desktop installation does not imply readable local
  account memories. Application caches, credentials and conversation archives
  are not discovery sources.
- Discovery inventories known files without reading their contents. It is
  bounded and reports incomplete inventories. Source selection precedes reading.
  Files are regular, bounded UTF-8 text. Directory-relative opens retain ancestor descriptors on supported platforms. Symlinks, special files, malformed text
  and changes during reading are rejected. Source files remain untouched. Claude topic headers are bounded to 4,096 characters before redaction.

## Review

The desktop provides source, scope, editable plain-text entries and an explicit
import action. Nothing is saved before that action or a source-specific automatic-import grant. Imported secrets are scrubbed
before preview and persistence; source text cannot introduce rich content.

Each selected entry has an explicit destination:

- Personal memory: User branch, for facts/preferences the user confirms.
- Standing instruction: Directives branch, for behaviour the user requests.
- Reference: World branch, for project notes or other reference material.

File and pasted notes default to reference; documented user instruction files default to standing instructions. Project notes default to reference. Their instructions do not enter the standing
profile automatically. The user can select, edit or omit entries. Blank entries,
unsupported destinations and oversized batches/entries are rejected before any
write. Each entry has an optional original statement date supplied by the user; absent dates stay unknown.
Neither file modification time nor import time becomes the date of an event.

## Persistence

Import creates graph nodes with neutral provider/entry titles, using the selected destination and exact reviewed
text. It does not infer facts, merge contradictions, execute instructions, alter
the diary or overwrite existing human edits.

Graph nodes and provenance commit in one SQLite writer transaction. Any failure
rolls back the batch. Exact whole-entry duplicates within a branch reuse an
existing node. Case, whitespace, Unicode characters and surrounding context
remain distinct; different branches, negations and temporal qualifiers remain
distinct. Repeated imports do not duplicate nodes or provenance. A changed source
can add provenance without rewriting a node edited after an earlier import.
An exact repeat preserves a manually moved node. A new source reuses only a live
node in the selected destination; archived destination metadata does not override
the node's current branch.
The current statement date binds to the complete, exact reviewed entry. An edited
or split excerpt retains archival provenance without inheriting its date as a
current statement date.

Provenance retains provider, source identity/label/scope, source-content digest,
original scrubbed quote, reviewed text, optional source date and the import's
local timestamp. These records remain available when graph data is edited or
split. Node deletion preserves the source record with no live node association.
Provenance is available in the memory viewer's node detail, with the current binding distinguished from edited text. Memory context carries source dates and fenced current text within the existing character limits. Current text is budgeted before optional archived quotes. Blank current text supplies no recall block. Partial excerpts have no statement date. Standing instructions are included only when the complete entry fits. Source records are archival quotations, not fresh tool results. Existing graph
search and the User/Directives warm profile make saved entries usable without a
new inference context. Import does not claim to repair source or temporal
uncertainty in existing memories.

## Desktop lifecycle

Discovery, reading and persistence run outside the GUI thread. Workers use the
shared keep-alive contract. Late results cannot replace a newer preview. Closing
the window preserves an already authorised import and does not destroy a running
worker. Both source and review pages scroll while navigation stays reachable.
The action is available from the shared tray/face menu and works without a running
daemon. Errors preserve editable review text and permit retry.

## Checks

Synthetic files and databases cover source scope/configuration, bounded reads,
symlinks/special files, Unicode, redaction, dates, explicit selection, idempotence,
existing edits, concurrent writers, rollback and provenance after edits/splits.
Desktop checks cover keyboard navigation, resizing, stale worker results and the
actual saved-memory outcome. Live local checks verify imported personal facts
and standing preferences remain usable alongside existing memory.

## Automatic local imports

After a successful manual import, the active import window offers a Yes/No choice
for that source. No persists across restarts. Closing the window before the offer
does not enable anything. Setup remains available in the import window. Pasted
text requires choosing and reviewing a local file; private application caches and
account memories are not automatic sources.

A grant covers one exact selected file, its reviewed scope and one destination.
Enabling requires a complete, unedited review of that file in a single destination.
Partial or edited selections open setup rather than approving omitted text. Future
entries use the granted destination without another review. Dates are not inferred.
Disabling or starting another setup increments a durable generation; a refresh
rechecks the generation inside its SQLite writer transaction.

The desktop watches only granted files and their parent directories, including
atomic-save replacements. One worker coalesces event bursts, with a 400 ms quiet
period and a two-second maximum delay. At most 32 sources are enabled at once. Declined decisions do not consume watch
capacity. Management prioritises enabled sources and shows up to 64 recent
source decisions; a reviewed file can enter setup directly.
A once-per-minute metadata reconciliation covers restarts and lost watcher events.
File events force a content check even if size and timestamps match. Existing
256 KiB and 128-entry limits apply; refreshes require no inference or network.
Unchanged metadata avoids content reads; unchanged content avoids graph,
provenance and source-checkpoint writes. Missing or unsafe sources preserve saved
memories and retry with exponential backoff from one minute to one hour.

Source versions reconcile exact complete entries without semantic guessing.
Unchanged entries retain their nodes and explicit statement dates. New entries
have unknown statement dates. Retired entries may be deleted only when their
nodes were created by that source, remain exactly as imported, have no children
and are not shared with another source. Edited, moved, shared, deleted or otherwise
protected entries pause that source for review before any graph change. Human
edits are never overwritten or silently resurrected. Returning to an earlier file
version imports the appropriate current entries, retaining historical provenance.
The changed nodes, new source records and successful digest commit atomically.
Background refreshes use indexed import records, without scanning unrelated graph
subtrees. Import ownership excludes access bookkeeping from edit detection. Source-owned
retirement is distinguished from human deletion, so a complete manual review can
restore a retired file version without resurrecting human-deleted entries.

The import window lists source state and permits disabling, setup and review.
The desktop owns the watcher independently of the window. Explicit Quit and
direct Qt quit events keep the GUI alive until the worker has released its
source/database resources. Repeated quit requests share that completion barrier;
fallback teardown joins the stopped worker before destroying the application.
Worker failures preserve consent and show a retryable status; reopening the import
window retries a stopped worker. Closing a review window never creates consent.
