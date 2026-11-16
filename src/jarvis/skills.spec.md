# Optional local skills

Skills provide reusable instructions through the shared voice and text assistant.
`skills_enabled` is strictly opt-in and defaults to false. `skills_paths` defaults
to an empty list and authorises only immediate skill directories within those
configured local folders. No marketplace, download, automatic script execution or
additional filesystem or tool authority is implied by a skill.

A package contains `SKILL.md` with safe YAML frontmatter: a directory-matching,
lowercase alphanumeric/hyphen `name` (1–64 characters, no adjacent or edge hyphens)
and a non-empty `description` (at most 1,024 characters). Other frontmatter cannot
grant permissions. YAML aliases are rejected by lexical scanning before
construction, preventing merge-alias expansion. Discovery reads at most 8 KiB of frontmatter per file and
never reads instruction bodies. Invalid or inaccessible packages are skipped
with bounded diagnostics. Duplicate names remain distinct and require their IDs.

The installed catalogue has no count limit. A reply owns its metadata index and
loaded guidance, with no process-global or persisted catalogue cache. Unicode
normalised lexical search ranks name and description matches deterministically.
It does not promise semantic or cross-language matching. The initial user query
retrieves at most five entries; `searchSkills` supports refined queries during
the loop. All metadata disclosure together is capped at 4 KiB per reply.
The full catalogue is never sent to an LLM. Description excerpts are at most
512 UTF-8 bytes and explicitly labelled when partial; search indexes the full
validated description. Each hit includes its canonical exact-ID `/SKILL.md`
entrypoint so callers do not need to construct a filename.

`readSkill` loads the complete instructions or a UTF-8 reference relative to a
discovered package. Its single `path` argument joins the exact skill ID or an
unambiguous name with `/` and the complete relative filename, retaining
subfolders. Each file is at most 12 KiB, and all loaded files together
are at most 16 KiB per reply. Oversized files fail honestly without truncation.
Repeated reads do not duplicate guidance. Absolute paths, traversal, symlinks,
reparse points and non-regular files are rejected through pinned filesystem
reads. Cancellation prevents new reads.

Loaded guidance is explicitly subordinate to user intent, privacy rules and
the existing tool allow-list. Complete files are returned as JSON in the current
reply's `readSkill` results without LLM digestion, accompanied by a reminder to
apply relevant instructions and read required references. Duplicate reads return an
acknowledgement referring to the earlier result, without repeating the body.
Skill tool requests and results are excluded from tool carryover, retaining
paired native requests/results for other tools in mixed batches. Assistant prose
and reasoning accompanying native calls are removed throughout a skill-using
reply's tool carryover. Search metadata
and loaded guidance cannot persist through this route into later replies.

Both skill tools are absent from routing and schemas when disabled, and direct
execution also rejects disabled use. Enabled replies expose both tools so late
discovery does not depend on the tool router guessing a skill's name. The same
configuration and reply engine serve voice and text. Settings use the existing
metadata-driven Features controls.

Skill discovery and instruction preparation are not evidence of executing the
user's requested action. Both completion and failed-follow-up guards retain this
distinction.
