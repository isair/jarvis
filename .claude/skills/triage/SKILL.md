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
not memory. Stay friendly, specific, and short.

## Step 1. Pull the state

Run these as parallel Bash tool calls (one message, two tool uses), not as chained shell commands:

```bash
gh issue list --state open --limit 50 --json number,title,author,createdAt,updatedAt,labels,comments \
  --jq '[.[] | {number, title, author: .author.login, labels: [.labels[].name], commentCount: (.comments|length), updatedAt}]'
```

```bash
gh api graphql -f query='{repository(owner:"isair",name:"jarvis"){discussions(first:30,states:OPEN,orderBy:{field:UPDATED_AT,direction:DESC}){nodes{id number title author{login} category{name} updatedAt comments(last:5){totalCount nodes{id author{login} createdAt body replies(last:10){nodes{id author{login} createdAt body}}}}}}}}' \
  --jq '.data.repository.discussions.nodes'
```

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
- **Awaiting reporter**: labelled `question` or the last comment is from the owner asking for details. Leave it unless the reporter has replied with new info. Per repo policy, do not close for silence before 2 weeks of reporter inactivity.
- **Owner tracking**: filed by `isair` as an internal task. Skip unless a non-owner has commented with a question or new information, in which case treat it like a normal untriaged thread.
- **Resolved-pending-release**: fix is on `develop`. Never close manually. Release (`git merge --ff-only develop` → `main`) auto-closes via `Closes #NNN`. Detect this by scanning recent `develop` commits (`gh pr list --base develop --state merged --limit 20`) for references to the issue number before you reply, so you can tell the reporter "this is fixed in the next release" rather than asking for more info.

## Step 2. Fetch details for the untriaged

For issues:

```bash
gh issue view <N> --json title,body,author,labels,comments \
  --jq '{title, author: .author.login, labels: [.labels[].name], body, comments: [.comments[] | {author: .author.login, createdAt, body}]}'
```

Read the **logs** and traceback carefully before replying. The vast majority of
reports contain the answer in the log; the reporter just didn't know what to
look for.

## Step 3. Diagnose from the log

Common Jarvis patterns and what they mean:

| Symptom in log | Likely cause | Ask for |
|----------------|--------------|---------|
| Repeated `📝 Heard: "Thank you."`, `"you..."`, `"Thanks for watching!"` with no real commands | Whisper hallucinations on near-silent audio. Wrong default mic or broken mic/driver. | Ask them to check the input level bar (Windows Sound settings, or macOS System Settings → Sound → Input) actually moves when they speak, and confirm which mic they intend to use. |
| `🧠 Intent judge: unavailable (timeout or error)` | Known; improved in v1.25.1 (bump this version as newer fixes ship). | Version they're on, and retry on latest. |
| `huggingface_hub.snapshot_download` crash (thread pool / ssl.create_default_context) | Download-time crash, platform-specific. Not the same as 429 throttling. | Keep open as its own bug. Workaround: manual `ollama pull ...` and relaunch. |
| `LLM connection error: ... RemoteDisconnected` | Ollama dropped. Upstream, not Jarvis. | `ollama run <model>` health check; Ollama version. |
| `setup_wizard.py ... _install_next_model` fatal | Real bug on our side. | Which model had just finished, which was about to start; `ollama list` after crash; `~/Library/Logs/DiagnosticReports/Jarvis-*.ips` on macOS. |
| `Low confidence` lines only, no `Heard:` ever | Mic is captured but utterances are under the confidence floor. Usually mic placement or wrong device. | Same as first row. |
| `📍 Location features are not available` | Not an error. Location is optional and only affects weather / local-time context. | Reassure, don't diagnose. Point at the MaxMind GeoLite2 signup if they actually want it. |

**Do not ask obviously-answered questions.** If the log shows the wizard was
pulling models, Ollama is by definition installed and running. If the log shows
Whisper loaded, Whisper is installed. Read before asking.

Other recurring user-environment answers:

- **Windows "Error 4551: Application Control policy has blocked this file"**: WDAC / AppLocker / corporate MDM, not Jarvis. Point at IT allow-listing, `secpol.msc`, or install-from-source.
- **"missing AI models"**: `ollama pull gemma4:e2b` + `ollama pull nomic-embed-text`, or tray → 🔧 Setup Wizard.
- **Setup wizard was closed early, nothing works**: tray → 🔧 Setup Wizard reopens it. Fallback: `rm -rf ~/.config/jarvis ~/.local/share/jarvis/config`.
- **`gemma4:e2b` quality complaints**: it is a very small model. Suggest 7B+ if hardware allows, note that capability scales with model size.
- **"Can Jarvis speak <language>?"**: yes if the chat model supports it; for voice, Whisper handles most languages. Point at README.

## Step 4. Label, retitle, reply

Available labels: `bug`, `question`, `duplicate`, `enhancement`, `documentation`, `good first issue`, `help wanted`, `invalid`, `wontfix`, `voice`, `spike`.

Conventions:

- Empty-body or needs-info bug reports: label `bug,question`, retitle to `"<one-line symptom> (awaiting details)"` or similar so the backlog is scannable.
- Duplicates: label `duplicate`, leave one short comment pointing at the canonical issue, close with `--reason "not planned"`.
- Real confirmed crashes: label `bug` (and `voice` if audio-related), retitle to pin the failure site from the traceback (e.g. `"Crash on first-run setup wizard during model install (macOS, v1.26.0)"`).

Reply tone:

- Open with `Hi @user, thanks for filing this! 👋`
- State the diagnosis (what the log shows) before the asks.
- Use bullet lists with **bold labels** for asks. Keep to 3 to 5 asks max.
- Friendly emojis: 👋 🙏 🚀 🧠 🎤 🔊 📝.
- **No em dashes (—) anywhere in user-facing writing.** Use commas, full stops, colons, or parentheses.
- **British English** (colour, behaviour, initialise).
- Do not promise fixes or ETAs.

## Step 5. Post the reply

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
   Then repost with the correct discussion id.

To reply to a specific comment (threaded sub-reply) rather than at the top level, pass `replyToId` in the mutation input. Otherwise the reply goes to the root.

If a `body` you want to post starts with `@`, use `-f body="..."`, not `-F body="..."`. `gh` interprets `-F` values starting with `@` as file paths.

## Step 6. Clean up your own past comments

If a previous owner comment was premature, wrong, or asked an
obviously-answered question, **edit it in place**. A clean thread beats a trail
of self-corrections.

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
