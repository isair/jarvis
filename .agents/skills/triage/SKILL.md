---
name: triage
description: >
  Triage open GitHub issues and discussions on the Jarvis repo. Sweep for
  untriaged reports, reply to awaiting-user threads when new info lands,
  apply the right labels, close duplicates, and edit past owner comments
  rather than stacking follow-ups. Use after a release or any time the user
  says "triage issues", "triage discussions", or similar.
---

# Triage Skill

You are triaging open issues and discussions on `isair/jarvis`. Work from data,
not memory. Stay friendly, specific, and short. Follow `AGENTS.md` and the
current user's publication and closure authorisation. Examples in this skill do
not grant permission to post, edit or delete public messages, close items, or
reset user configuration. Keep replies as private drafts when required.

## Step 1. Pull the state

Fetch issues and discussions independently:

```bash
gh issue list --repo isair/jarvis --state open --limit 1000 --json number,title,author,createdAt,updatedAt,labels,comments \
  --jq '[.[] | {number, title, author: .author.login, labels: [.labels[].name], commentCount: (.comments|length), updatedAt}]'
```

```bash
gh api graphql -f query='{repository(owner:"isair",name:"jarvis"){discussions(first:30,states:OPEN,orderBy:{field:UPDATED_AT,direction:DESC}){nodes{id number title author{login} category{name} updatedAt comments(last:5){totalCount nodes{id author{login} createdAt body replies(last:10){nodes{id author{login} createdAt body}}}}}}}}' \
  --jq '.data.repository.discussions.nodes'
```

Fetch the complete issue queue with pagination. If a result reaches its limit,
continue fetching rather than treating the first page as the backlog. The
discussion query is a discovery snapshot; paginate the open discussion queue
and each reviewed discussion's comments and replies before deciding who last
responded.

**Important**: GitHub Discussions are threaded. The top-level `comments` list does
not include sub-replies, so a fresh reporter question that lives under an owner
comment will look like an unanswered top-level thread if you forget to fetch
`replies`. The query above pulls both. When deciding "untriaged" vs "awaiting
reporter", scan the **last reply across the whole tree**, not just the last
top-level comment. A common shape: owner answers at the top level, reporter
replies underneath, owner replies underneath that. The newest message is two
levels deep, and you'll miss it if you only look at the top-level list.

Classify each thread into one of:

- **Untriaged**: no owner (`isair`) reply yet. Act now.
- **Awaiting reporter**: a published clarifying question still needs an answer after reading the complete thread. A `question` label or an unsent draft alone does not establish that anyone was asked. Review new information when it arrives; do not close for silence before two weeks after the actual published question and the latest reporter activity. Elapsed time alone neither resolves a defect nor authorises closure.
- **Owner tracking**: filed by `isair` as an internal task. Assess its current implementation and next action under the maintenance loop; do not invent a reporter troubleshooting request. New community information still needs review.
- **Resolved-pending-release**: a verified fix is on `develop`. Compare the report with the actual implementation and linked changes, not just a title or commit reference. Recent merged PRs are discovery leads; extend the search when needed. Never close manually merely because the fix landed on `develop`. The release fast-forwards `main` and auto-closes references written as `Closes #NNN`. Draft the verified release status rather than requesting evidence already supplied.

## Step 2. Fetch details for the untriaged

For issues:

```bash
gh issue view <N> --json title,body,author,labels,comments \
  --jq '{title, author: .author.login, labels: [.labels[].name], body, comments: [.comments[] | {author: .author.login, createdAt, body}]}'
```

Read the **logs** and traceback carefully before replying. Separate what the
report demonstrates from what still needs investigating.

**Always fetch the full issue body.** Do not truncate with `body[:N]` — the
interaction lines (`📝 Heard:`) often appear near the bottom, past a 2000-char
cutoff. Use the unbounded jq selector (`.body`) and read every line.

**Read the log top to bottom, don't stop at the startup section.** The startup
(lines like "Whisper loaded", "Piper download 73%") is often misleading — a
Piper download that shows 73% in the truncated view may have completed at 100%
ten lines later. Always scroll to the end to check for `📝 Heard:`, `🧠 Intent
judge:`, or `❌ Failed` lines before diagnosing.

**Scan for `📝 Heard:` lines first.** They are the most actionable signal in any
log. A `Heard:` line proves that recognition produced a transcript. Compare it
with the reported utterance and subsequent filtering and intent decisions; it
does not prove that a person spoke.

## Step 3. Diagnose from evidence

A transcript records what speech recognition produced, not proof of the words a
person said. Whisper can produce transcripts from silence, background sound or
assistant speech. Read the complete log before deciding whether the problem is
capture, recognition, intent, model availability, tool execution or playback.

| Evidence in the log | Supported conclusion and next action |
|---------------------|--------------------------------------|
| Wake name without an accompanying request | The transcript contains no request. Confirm what the person intended before diagnosing microphone or model failure. |
| Wake name transcribed differently | Recognition and the reported utterance may differ. Check the configured wake name/aliases and ask what was actually said if the report does not say; do not tell the person they used the wrong name based only on ASR output. |
| Intent or reply request exceeds its logged timeout | That request did not finish in time. Check provider availability, selected models and resource evidence. A Whisper CUDA warning does not establish the LLM backend's device. |
| Repeated unrelated or multilingual transcripts with no reported speech | Possible recognition hallucinations or background/echo capture. Check readiness, input selection, confidence and speech/echo filtering; do not assume a broken microphone. |
| Only low-confidence lines | Audio reached the recogniser but was filtered. The log alone does not prove a person spoke or that a command was missed. |
| No transcripts | The log does not distinguish no speech, missing capture, loading, backlog or filtering. Check the end of startup and ask for a short attempted interaction with fresh logs. |
| Warmup failure | The probe failed at that time. Follow the later interaction before attributing an unanswered request to warmup; warmup is not a full-request timing test. |
| Piper or Whisper download error | Identify the downloader, exact exception and platform. Connection, TLS, cache, permission and timeout failures need different remedies. Ollama pulls do not repair Whisper or Piper downloads. |
| Download progress ends before completion | Completion is not shown. Check later lines before calling it stuck or successful. Ask for the outcome if the log ends during loading. |
| Empty report or cleared logs | There is insufficient evidence. Request a brief description and fresh logs covering the failure. |
| Application Control names a blocked native library | Windows policy blocked that file. Consider the named policy and packaging/signing evidence; do not advise disabling organisational controls. |
| Native abort during a window opening | Preserve the crash location, OS/build details and native crash report. Python startup logs alone may not identify the native fault. |
| Optional location data unavailable | Location setup is incomplete, not necessarily the cause of the reported failure. Read the failing interaction first. |

### Follow the actual interaction

- Compare reported speech with the transcript, then inspect echo/confidence
  filtering, intent and the reply/tool outcome. Do not ask whether recognition
  produced anything when the log already contains the relevant transcript.
- The configured wake name can occur at the start or end of a request. A brief
  hot window allows directed follow-ups without repeating it; unrelated
  conversation is not automatically addressed to Jarvis.
- Ask only for evidence still missing. Successful loading or a model pull proves
  that operation succeeded then, not that the current provider, microphone or
  complete request is healthy. Respect the chosen provider rather than assuming
  every installation uses Ollama.
- Use the current model catalogue, computed setup recommendations and logged
  limits as the source of truth. Avoid stale model tags, fixed size rules or a
  blanket instruction to use one tiny model for every role.
- Distinguish speech recognition, model language understanding and TTS language
  support. A multilingual chat model does not prove the configured voice can
  speak that language.
- Reopen Setup from the tray when setup needs completing. Preserve saved choices
  and data. Do not recommend deleting configuration or memory as routine
  troubleshooting, and do not disable TLS verification to bypass a download
  failure.

If a diagnosis changes, prepare a concise correction to the existing reply when
appropriate. Keep the evidence and uncertainty clear; do not hide mistakes by
deleting a public correction trail.

## Step 4. Label, retitle, reply

Available labels: `bug`, `question`, `duplicate`, `enhancement`, `documentation`, `good first issue`, `help wanted`, `invalid`, `wontfix`, `voice`, `spike`.

Conventions:

- Empty-body or needs-info bug reports: label `bug,question`, retitle to `"<one-line symptom> (needs details)"` or similar so the backlog is scannable.
- Duplicates: label `duplicate`, leave one short comment pointing at the canonical issue, close with `--reason "not planned"`.
- Real confirmed crashes: label `bug` (and `voice` if audio-related), retitle to pin the failure site from the traceback (e.g. `"Crash on first-run setup wizard during model install (macOS, v1.26.0)"`).

Reply tone:

- Thank the reporter briefly, state what the evidence shows, and ask only for missing details.
- Match the current user's writing preferences rather than imposing a fixed greeting or formatting template.
- Keep replies short and distinguish an observed failure from a hypothesis.
- **No em dashes (—) anywhere in user-facing writing.** Use commas, full stops, colons, or parentheses.
- **British English** (colour, behaviour, initialise).
- Do not promise fixes or ETAs.

## Step 5. Apply authorised changes

Use the following commands only within the user's current authorisation. When
publication is not authorised, provide the exact reply and target as a private
draft. Do not describe an unsent question as awaiting a reporter response.

Issue comment:

```bash
gh issue comment <N> --body "..."
gh issue edit <N> --add-label "bug,question" --title "..."
gh issue close <N> --reason "not planned"   # duplicates / wontfix only
```

Discussion comment (GraphQL, and **use `-f body=` not `-F body=`** if the body
starts with `@`, because `gh` treats `-F` values starting with `@` as file
paths):

```bash
gh api graphql -f query='mutation($id:ID!,$body:String!){addDiscussionComment(input:{discussionId:$id,body:$body}){comment{url}}}' \
  -F id=<discussion node id> -f body="@user, ..."
```

Get the discussion `id` field from the Step 1 GraphQL output. It's the outer `id` on the discussion node, not the inner `id` inside `comments.nodes` (that one is the comment's node id, used in Step 6 for edits).

**Verify the node id before posting.** Discussion node ids look like `D_kwDOPgt_k84Albb5` and a single-character typo will silently route the comment to a completely unrelated repo's discussion (the prefix encodes the repo, but neighbouring ids belong to other repos). Two safeguards:

1. Copy the id straight from the Step 1 output, never retype it.
2. The mutation response returns the comment URL: `addDiscussionComment.comment.url`. Inspect it. If the host path is anything other than `github.com/isair/jarvis/discussions/<N>`, you posted to the wrong repo. Delete the comment immediately:
   ```bash
   gh api graphql -f query='mutation($id:ID!){deleteDiscussionComment(input:{id:$id}){comment{id}}}' -F id=<comment node id>
   ```
   Repost only within the existing publication authorisation, after verifying the correct discussion ID.

To reply to a specific comment (threaded sub-reply) rather than at the top level, pass `replyToId` in the mutation input. Otherwise the reply goes to the root.

If a `body` you want to post starts with `@`, use `-f body="..."`, not `-F body="..."`. `gh` interprets `-F` values starting with `@` as file paths.

## Step 6. Clean up your own past comments

When authorised to edit an existing owner comment, correct its diagnosis and
keep any necessary correction clear. Otherwise prepare an edit draft with the
comment ID and replacement text. Do not delete public messages merely to make
the thread look as though the original diagnosis was correct.

Issue comment edit:

```bash
gh api -X PATCH repos/isair/jarvis/issues/comments/<commentId> -f body="..."
```

Discussion comment edit. First grab the comment node id (the `last:5` window usually covers recent owner replies):

```bash
gh api graphql -f query='{repository(owner:"isair",name:"jarvis"){discussion(number:N){comments(last:5){nodes{id author{login} createdAt body}}}}}'
```

Then update it:

```bash
gh api graphql -f query='mutation($id:ID!,$body:String!){updateDiscussionComment(input:{commentId:$id,body:$body}){comment{url}}}' \
  -F id=<comment node id> -f body="..."
```

## Step 7. Summarise to the user

At the end, list what you touched per thread: labels changed, titles changed,
comments posted, closures. Use markdown links like `[#241](https://github.com/isair/jarvis/issues/241)`. Keep it short.

## Hard rules

- Never close an issue because its fix landed on `develop`. Let the release auto-close.
- Never close for reporter silence under 2 weeks after a clarifying question.
- Never ask a question the log already answers.
- Never use em dashes in user-facing text.
- Never invent facts about a reporter's environment. Ask, or infer only from the log.
- When in doubt, label `question` and ask rather than guess.
