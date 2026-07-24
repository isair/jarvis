---
name: review-pr
description: >
  Single-pass PR review for code quality, potential issues, CI failures, test
  coverage, and adherence to project standards. Lighter than `/review-pr-ultra`.
  Use for routine reviews; use `/review-pr-ultra` for high-risk, security-sensitive,
  or large changes where deeper multi-agent analysis is warranted.
argument-hint: "[PR number or URL]"
---

# Pull Request Review

Review a pull request thoroughly for code quality, potential issues, and
adherence to project standards. Diagnose and fix any CI failures. Verify test
coverage is appropriate for the repo.

## Step 1 — Gather Context

Determine the PR:
- If `$ARGUMENTS` is provided, use it (a PR number, URL, or branch name).
- Otherwise, detect the current branch and find its open PR.

```bash
gh pr view <PR> --json title,body,author,baseRefName,headRefOid,labels,files,additions,deletions,commits,reviews,comments,statusCheckRollup
gh pr diff <PR>
```

Read the project's `CLAUDE.md` for coding conventions to enforce.

## Step 2 — Diagnose CI

For each failing check in `statusCheckRollup`:

```bash
gh run view <RUN_ID> --log-failed
```

Classify as **caused-by-PR** or **pre-existing/flaky**. If caused by the PR,
fix the issue. Surface in the final report.

## Step 3 — Review the Diff

Read through the diff file by file. For each changed file, also read the
surrounding context (not just the diff lines) to understand how the change
fits into the module.

### Review Checklist

#### Correctness
- Logic bugs, off-by-one, null/undefined handling, race conditions
- Broken invariants, incorrect control flow
- State management issues (missing assignments, leaked state)
- Regressions: does this change break existing behaviour?

#### Security
- Injection (command, path traversal, XSS)
- Secrets or credentials in code
- Unsafe deserialisation / SSRF / open redirects
- Cryptographic misuse, insecure randomness

#### Performance
- N+1 queries, unnecessary allocations, missing caching
- O(n²) or worse where linear is possible
- Blocking calls in async/event-loop contexts
- Memory leaks, unbounded growth (queues, buffers, caches)

#### Maintainability
- SOLID violations, excessive coupling
- Code duplication (DRY violations)
- Naming clarity (variables, functions, classes)
- Inconsistency with project conventions (from CLAUDE.md)

#### Completeness
- Missing test coverage for new/changed code paths
- Missing error handling for failure modes
- Undocumented behaviour changes (README, specs)
- Spec drift: do changes contradict any spec files?

## Step 4 — Synthesise Report

```markdown
## PR Review: <title>

### Summary
<2-3 sentences: what the PR does and overall assessment>

### CI Status
<All green, OR per-failure diagnosis>

### Issues
<Finding per item: **path/to/file.py:LINE** — severity — description — suggestion>

### What Looks Good
<Positive observations — good patterns, thorough tests, clean design>

### Verdict
<APPROVE | REQUEST_CHANGES | COMMENT — brief justification>
```

### Verdict rules
| State | Verdict |
|-------|---------|
| Any critical or high issue | **REQUEST_CHANGES** |
| Only medium/low issues | **COMMENT** |
| No issues | **APPROVE** |

### Report rules
- Lead with the most important issues.
- Every criticism includes a concrete suggestion.
- British English; emojis for emphasis (per project conventions).
- Be constructive, not nitpicky.

## Step 5 — Present and Ask for Go-Ahead

After the report is ready, offer to post it to GitHub as inline comments. Use
the `ask` tool:

```markdown
**Review complete for #{pr_number}.** Ready to post to GitHub?

I can post the key findings as inline comments (one review, anchored to lines)
with a brief summary body. The full report is above.
```

Only proceed after user confirmation. Do not post automatically.

## Step 6 — Post to GitHub (on confirmation)

Post a single review with all inline comments in one API call:

```bash
gh pr view <PR> --json headRefOid
```

```bash
gh api repos/<owner>/<repo>/pulls/<PR>/reviews --input - <<'EOF'
{
  "commit_id": "<HEAD_SHA>",
  "event": "APPROVE",
  "body": "<2-3 line summary>",
  "comments": [
    { "path": "src/file.py", "line": 42, "body": "<terse note>" }
  ]
}
EOF
```

| Verdict | `event` |
|---------|---------|
| APPROVE | `APPROVE` |
| REQUEST_CHANGES | `REQUEST_CHANGES` |
| COMMENT | `COMMENT` |

- `line` is the 1-indexed line in the post-change file (right side), must fall within a changed hunk.
- `commit_id` — set to the PR head SHA.
- `body` — keep it to 2–3 lines; the full report is already in-thread.
- Do NOT post individual comments via `POST /pulls/{number}/comments` — that creates separate threads. The `reviews` endpoint with a `comments` array keeps them together.
- Include only the most important findings as inline comments. Terse, conversational, one point each.
- Confirm the PR URL and verdict once posted.

## Important Guidelines

- **Do NOT make changes to code** — this is a read-only review
- **Respect the author's intent** — understand why before criticising what
