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
  Files are regular, bounded UTF-8 text. POSIX directory-relative opens retain the checked ancestry while selecting the file.
  Windows opens inspect native directory/file handles without following reparse
  points; retained parents request directory-list access and deny write/delete
  sharing while later components are selected.
  The selected Windows file denies write and delete sharing until its read closes.
  File identity, size and timestamps are checked against the
  open descriptor and current path after reading. Symlinks, special files, malformed text
  and changes during reading are rejected. Source files remain untouched. Claude topic headers are bounded to 4,096 characters before redaction.

## Review

The desktop provides source, scope, editable plain-text entries and an explicit
import action. Nothing is saved before that action. Imported secrets are scrubbed
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
existing node. Duplicate lookup returns only exact reviewed-text candidates
from the selected branches; unrelated note payloads do not enter Python. Case,
whitespace, Unicode characters and surrounding context
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
Provenance is available in the memory viewer's node detail, with the current binding distinguished from edited text. Memory context carries source dates and fenced current text within the existing character limits. Current text is budgeted before optional archived quotes. Warm-profile reads retain at most the import entry character limit plus one lookahead character. Oversized edited entries contribute only complete newline-delimited prefixes before that limit, so clipped credential tokens do not enter redaction or context. Such prefixes remain explicitly incomplete and undated; incomplete standing instructions are omitted. Complete entries preserve literal whitespace and source dates. Blank current text supplies no recall block. Partial excerpts have no statement date. Standing instructions are included only when the complete entry fits. Source records are archival quotations, not fresh tool results. Existing graph
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
