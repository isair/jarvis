---
name: triage
description: >
  Triage open GitHub issues and discussions on the Jarvis repo. Sweeps for
  untriaged reports, replies to awaiting-user threads when new info lands,
  applies the right labels, closes duplicates, and edits past owner comments
  rather than stacking follow-ups. Use after a release or any time the user
  says "triage issues", "triage discussions", or similar.
---

# Triage Skill

You are triaging open issues and discussions on `isair/jarvis`. Work from data,
not memory. Stay friendly, specific, and short.

## Step 1. Pull the state

Run these in parallel:

```bash
gh issue list --state open --limit 50 --json number,title,author,createdAt,updatedAt,labels,comments \
  --jq '[.[] | {number, title, author: .author.login, labels: [.labels[].name], commentCount: (.comments|length), updatedAt}]'
```

```bash
gh api graphql -f query='{repository(owner:"isair",name:"jarvis"){discussions(first:30,states:OPEN,orderBy:{field:UPDATED_AT,direction:DESC}){nodes{number title author{login} category{name} updatedAt comments(last:5){totalCount nodes{author{login} createdAt body}}}}}}' \
  --jq '.data.repository.discussions.nodes'
```

Classify each thread into one of:

- **Untriaged**: no owner (`isair`) reply yet. Act now.
- **Awaiting reporter**: labelled `question` or the last comment is from the owner asking for details. Leave it unless the reporter has replied with new info. Per repo policy, do not close for silence before 2 weeks of reporter inactivity.
- **Owner tracking**: filed by `isair` as an internal task. Skip unless a user has commented.
- **Resolved-pending-release**: fix is on `develop`. Never close manually. Release (`git merge --ff-only develop` → `main`) auto-closes via `Closes #NNN`.

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
| Repeated `📝 Heard: "Thank you."`, `"you..."`, `"Thanks for watching!"` with no real commands | Whisper hallucinations on near-silent audio. Wrong default mic or broken mic/driver. | Windows Sound → Input level bar check; which mic they intend to use. |
| `🧠 Intent judge: unavailable (timeout or error)` | Known; improved in v1.25.1. | Version, and retry on latest. |
| `huggingface_hub.snapshot_download` crash (thread pool / ssl.create_default_context) | Download-time crash, platform-specific. Not the same as 429 throttling. | Keep open as its own bug. Workaround: manual `ollama pull ...` and relaunch. |
| `LLM connection error: ... RemoteDisconnected` | Ollama dropped. Upstream, not Jarvis. | `ollama run <model>` health check; Ollama version. |
| `setup_wizard.py ... _install_next_model` fatal | Real bug on our side. | Which model had just finished, which was about to start; `ollama list` after crash; `~/Library/Logs/DiagnosticReports/Jarvis-*.ips` on macOS. |
| `Low confidence` lines only, no `Heard:` ever | Mic is captured but utterances are under the confidence floor. Usually mic placement or wrong device. | Same as first row. |
| `📍 Location features are not available` | Cosmetic. Optional. | Reassure, don't diagnose. |

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

- Empty-body or needs-info bug reports: label `bug,question`, retitle to `"<one-line symptom> (awaiting details)"` or similar so the backlog scannable.
- Duplicates: label `duplicate`, leave one short comment pointing at the canonical issue, close with `--reason "not planned"`.
- Real confirmed crashes: label `bug` (and `voice` if audio-related), retitle to pin the failure site from the traceback (e.g. `"Crash on first-run setup wizard during model install (macOS, v1.26.0)"`).

Reply tone:

- Open with `Hi @user, thanks for filing this! 👋`
- State the diagnosis (what the log shows) before the asks.
- Use bullet lists with **bold labels** for asks. Keep to 3–5 asks max.
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

Get the discussion `id` field from the Step 1 GraphQL output.

## Step 6. Clean up your own past comments

If a previous owner comment was premature, wrong, or asked an
obviously-answered question, **edit it in place**. A clean thread beats a trail
of self-corrections.

Issue comment edit:

```bash
gh api -X PATCH repos/isair/jarvis/issues/comments/<commentId> -f body="..."
```

Discussion comment edit (get the comment node id via `gh api graphql` on the discussion):

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
