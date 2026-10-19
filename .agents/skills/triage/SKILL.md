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

**Always fetch the full issue body.** Do not truncate with `body[:N]` — the
interaction lines (`📝 Heard:`) often appear near the bottom, past a 2000-char
cutoff. Use the unbounded jq selector (`.body`) and read every line.

**Read the log top to bottom, don't stop at the startup section.** The startup
(lines like "Whisper loaded", "Piper download 73%") is often misleading — a
Piper download that shows 73% in the truncated view may have completed at 100%
ten lines later. Always scroll to the end to check for `📝 Heard:`, `🧠 Intent
judge:`, or `❌ Failed` lines before diagnosing.

**Scan for `📝 Heard:` lines first.** They are the most actionable signal in any
log. If a `Heard:` line exists, the system detected speech — the question is
what it heard and whether it matches the wake word. Do not ask "did it hear
you?" when a `Heard:` line proves it did.

## Step 3. Diagnose from the log

Read every line of the log. Do not skip what looks like normal startup — the
answer is almost always there. Follow this diagnosis flow:

### Quick-reference symptom table

| Symptom in log | Likely cause | Action |
|----------------|--------------|--------|
| `📝 Heard: "Jarvis."` then `🧠 Intent (wake word): not directed (Wake word detected, but no query followed it.)` | User said the wake word but didn't follow with a command. The wake word alone does nothing. | Explain that they need to say a command after the wake word (e.g. "Jarvis, what time is it?"). |
| `📝 Heard: "...George..."` or `"...Georg..."` | Wake word "Jarvis" misheard as "George" by Whisper (very common phonetic confusion). | Tell them to use the correct wake word. |
| `📝 Heard: "Jarvis, ..."` then `🧠 Intent judge: unavailable (timeout)` | Intent judge model too slow for hardware. Usually CPU-only or large model mismatch. | Advise running setup wizard lowest option. |
| `📝 Heard: "Jarvis, ..."` then `⏱️ LLM request timed out` | Chat model too slow for hardware. Same root cause as above. | Advise running setup wizard lowest option. |
| Repeated `📝 Heard: "Thank you."` / `"you..."` / `"Thanks for watching!"` / garbled text (e.g. Devanagari characters, random phrases like "Mission success!") with no real commands | Whisper hallucinations on near-silent audio. Wrong default mic or broken mic/driver. | Check input level in OS sound settings; confirm intended mic. If garbled text appears in a different script, it's still a hallucination — the script doesn't matter. |
| `Low confidence` lines only, no `Heard:` ever | Mic captures audio but utterances are under the confidence floor. Wrong device or mic placement. | Same as above. |
| `⚠️  Chat model '...' warmup failed` + `⚠️  Intent judge '...' warmup failed` | Two different model variants loaded (e.g. gemma4:e4b for chat, gemma4:e2b for intent). The `ℹ️ CUDA not available` warning in the log is Whisper-only (STT speed), but the LLM models (intent judge, chat) are served by Ollama and may also be on CPU. Two variants competing on CPU overwhelms most machines. | Run setup wizard lowest option to use one model for everything. |
| Normal startup log, zero interaction lines (`📝 Heard:`, `🧠 Intent judge:`, `💬 Generating`) | User never spoke after launch, or didn't use the wake word. This is by far the most common. | Ask what they said and whether they said "Jarvis" first. |
| `huggingface_hub.snapshot_download` crash (thread pool / ssl.create_default_context) | Platform-specific download crash. | Manual `ollama pull ...` workaround. |
| `LLM connection error: ... RemoteDisconnected` | Ollama process crashed or unreachable. | `ollama run <model>` health check; Ollama version. |
| `⚠️  large-v3-turbo is not supported by the installed Whisper engine, using large-v3 instead` | User selected `large-v3-turbo` as Whisper model but the installed engine doesn't support it. Falls back to `large-v3`, which is much larger and slower. | Run setup wizard and pick a supported model like `medium` or `small`. |
| Piper voice download stuck (percentage stops advancing) | First-run ~60 MB TTS voice download with slow/unstable connection. Check the log fully — if download reached 100%, this isn't the issue. | Wait or relaunch; check internet. |
| `❌ Failed to load Whisper model: ConnectTimeout` | HF blocked by firewall (common in China). | `HF_ENDPOINT=https://hf-mirror.com` env var or VPN. |
| `❌ Microphone permission check failed: Error querying device -1` | No microphone detected on Windows. | Check Windows Sound Settings for input devices. |
| `🔊 Downloading Piper voice` line at startup, nothing after | First launch — Piper voice is still downloading. Normal, just slow. | Reassure, ask them to wait or relaunch. |
| `🗑️ Logs Cleared` or empty bug template | User cleared logs before filing, or the template was submitted unfilled. No data to diagnose. | Label `bug,question`, ask for description and fresh logs. |
| `📍 Location features are not available` | Not an error. Location is optional. | Reassure, point at MaxMind GeoLite2 signup if they want it. |
| `Fatal Python error: Aborted` during setup wizard launch | macOS app bundle crash during first-run wizard. | Ask for macOS version, system crash reports, Ollama status. |
| `Fatal Python error: Illegal instruction` in pynput | macOS dictation keyboard listener incompatibility. | Suggest disabling dictation in settings. |
| `mkl_malloc: failed to allocate memory` when loading Whisper | Intel MKL memory allocation failure. | Try CPU-only mode or run setup wizard again. |

### Diagnosis flow (read in this order)

**1. Check for `📝 Heard:` lines.** These are the most actionable signal.

- **No `📝 Heard:` line at all** — the user never spoke to Jarvis, or their
  mic isn't capturing speech. Either:
  - They don't know to use the wake word (common — log shows healthy startup,
    no interaction)
  - Mic is wrong/broken (check for "Error querying device" or "Low confidence")
  - Mic permissions denied
  → Ask what they said and whether they used the wake word. Explain that
    the wake word only needs to appear once in the first utterance, and can
    be at the start or end of the sentence (e.g. "What time is it, Jarvis?").
    After that, during the hot window, you can keep talking without repeating
    it — even a long tangent, as long as you started with the wake word.

- **`📝 Heard:` line exists** — the system detected speech. Check *what* it heard:
  - **"George" / "Georg"** — the wake word "Jarvis" was misheard by Whisper.
    Point it out directly.
  - **Wake word present** (e.g. "Jarvis, open Chrome" or "What do you think,
    Jarvis?") — good. Move to step 2. The wake word can be at the start **or
    end** of the sentence.
  - **Random words, no wake word** — user didn't use the wake word. Explain
    that only the first utterance needs it; within the hot window (follow-up
    period after a response) you can keep talking without repeating it.
  - **"Thank you" / "you..." / "Thanks for watching"** — Whisper hallucinating
    on silent audio. Wrong mic device. Troubleshoot mic.

**2. Check what happened after the `Heard:` line.**

- **`🧠 Intent judge: unavailable (timeout after 15.0s)`** — the intent judge
  model is too slow for the hardware. Usually CPU-only or large model mismatch.
  → Advise setup wizard lowest option.

- **`⏱️ LLM request timed out`** — the chat model is too slow. Same cause.
  → Same advice: setup wizard lowest option.

- **`✨ Working on it:` followed by a response** — everything worked. The issue
  may be intermittent. Ask for the exact scenario that failed.

- **Nothing after `Heard:`** — log truncated or cleared. Ask for fresh logs.

**3. Check startup for model loading issues.**

- `⚠️ ... warmup failed` with **two different models** loaded
  (e.g. `gemma4:e4b` for chat, `gemma4:e2b` for intent): two variants
  competing on CPU overwhelms most machines. The `ℹ️ CUDA not available`
  warning in the log is Whisper-only (STT speed); the LLM models are served
  by Ollama and may be on CPU too, but those are separate systems.
  → Advise setup wizard lowest option to unify on `gemma4:2b`.

- `⚠️ ... warmup failed` but everything else healthy and no interaction lines:
  warmup failures are non-fatal (models load on first use). The real issue is
  likely step 1 (no wake word). Focus on that, not the warnings.

**4. Check for download failures (common on first run).**

- Piper voice stuck: first-run ~60 MB download, slow connection.
- Whisper model download timeout: HF blocked (China) or slow network.
- HuggingFace hub errors.

**5. Startup looks perfectly normal, no errors, no interaction lines.**

This is the most common pattern. Everything loaded and is listening, but zero
`📝 Heard:` or `🧠 Intent judge:` lines. The user almost certainly never tried
speaking, or didn't use the wake word.
  → Do not ask about warmup warnings or model config. Ask what they said and
    whether they said "Jarvis" first.

**Do not ask obviously-answered questions.** If the log shows the wizard was
pulling models, Ollama is by definition installed and running. If Whisper
loaded, Whisper is installed. If a `📝 Heard:` line exists, the system heard
speech — do not ask "did it hear you?". Read the log, identify the branch, and
diagnose.

**If you realise you were wrong, edit — do not stack.** A single clean comment
with the correct diagnosis is always better than the original plus two
follow-ups saying "actually..." and "scratch that...". If you're not confident
in the diagnosis, re-read the log before posting rather than guessing and
correcting later. The thread is public and every "sorry, let me correct myself"
undermines trust in the triage.

### Other recurring user-environment answers

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
obviously-answered question, **edit it in place** and **delete any follow-up
correction comments** you left. A clean thread beats a trail of
self-corrections. If you stacked "actually..." or "scratch that..." comments
after the original, delete them once the first comment is accurate — the thread
should read as if you got it right the first time.

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
