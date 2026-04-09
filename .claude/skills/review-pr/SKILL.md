---
name: review-pr
description: >
  Multi-agent adversarial PR review. Spawns parallel specialist agents
  (correctness, security, performance, maintainability, completeness) then
  a verifier agent that challenges every finding. Only verified issues survive.
  Accepts an optional PR number or URL; defaults to the current branch's open PR.
argument-hint: "[PR number or URL]"
---

# Multi-Agent Adversarial PR Review

You are an orchestrator for a thorough, multi-perspective pull request review.
Your job is to gather PR context, spawn specialist review agents in parallel,
then run a verification pass to filter out false positives.

## Step 1 — Gather PR Context

Determine the PR to review:
- If `$ARGUMENTS` is provided, use it (a PR number, URL, or branch name).
- Otherwise, detect the current branch and find its open PR.

Use the GitHub MCP tools (or `gh` CLI if MCP is unavailable) to fetch:
1. **PR metadata**: title, body, author, base branch, labels
2. **Full diff**: the complete code diff
3. **Changed file list**: just the filenames for targeted exploration
4. **PR comments/reviews**: any existing review feedback
5. **CI status**: check if CI is passing or failing

Also read the project's `CLAUDE.md` for coding conventions the review should enforce.

Store all this context — you will include it in each specialist agent's prompt.

## Step 2 — Spawn Specialist Agents (Parallel)

Launch **all five** specialist agents simultaneously using the Agent tool.
Each agent receives the full diff, changed file list, PR description, and
project conventions. Each must output a structured list of findings.

### Agent 1: Correctness Reviewer
Focus: Logic bugs, edge cases, regressions.
- Off-by-one errors, null/undefined handling, race conditions
- Broken invariants, incorrect control flow
- State management issues (missing assignments, leaked state)
- Regressions: does this change break existing behaviour?
- Read surrounding code (not just the diff) to understand context

### Agent 2: Security Reviewer
Focus: Vulnerabilities and unsafe patterns.
- Injection (SQL, command, XSS, path traversal)
- Authentication/authorisation bypass
- Secrets or credentials in code
- Unsafe deserialisation, SSRF, open redirects
- Cryptographic misuse, insecure randomness
- Dependency vulnerabilities (if new deps added)

### Agent 3: Performance Reviewer
Focus: Efficiency and scalability.
- N+1 queries, unnecessary allocations, missing caching
- O(n²) or worse algorithms where linear is possible
- Blocking calls in async/event-loop contexts
- Memory leaks, unbounded growth (queues, buffers, caches)
- Unnecessary I/O, redundant network calls

### Agent 4: Maintainability Reviewer
Focus: Design quality and readability.
- SOLID principle violations, excessive coupling
- Code duplication (DRY violations)
- Naming clarity (variables, functions, classes)
- Missing or misleading comments/docstrings
- Overly complex logic that could be simplified
- Inconsistency with project conventions (from CLAUDE.md)

### Agent 5: Completeness Reviewer
Focus: What's missing.
- Missing test coverage for new/changed code paths
- Missing error handling for failure modes
- Undocumented behaviour changes (README, specs, CHANGELOG)
- Spec drift: do changes contradict any spec files?
- Missing migration steps or configuration updates
- Edge cases not addressed in the implementation

### Agent Prompt Template

Each agent's prompt MUST include:
1. The full diff
2. The changed file list
3. The PR description
4. Relevant project conventions from CLAUDE.md
5. Instruction to READ the surrounding code in changed files (not just the diff lines) for full context
6. Instruction to output findings as a structured list:

```
For each finding, output:
- **File**: path/to/file.py:LINE
- **Severity**: critical / high / medium / low
- **Category**: bug / security / performance / design / missing
- **Confidence**: high / medium / low
- **Description**: What the issue is and why it matters
- **Suggestion**: Concrete fix or alternative approach
```

7. Instruction: if no issues found in your area, explicitly state "No issues found" — do not invent findings to appear thorough.
8. Instruction: only report issues with confidence >= medium. Do not report style nits unless they violate project conventions.

## Step 3 — Verification Phase (Adversarial)

After ALL specialist agents complete, spawn a single **Verifier Agent** that
receives every finding from all specialists. The verifier's job is to
**challenge and disprove** each finding:

### Verifier Agent Instructions

You are a devil's advocate. For EACH finding from the specialist reviewers:

1. **Read the actual code** (not just the diff) — the "bug" may be handled
   elsewhere in the codebase.
2. **Check if the concern is mitigated** by framework defaults, type system
   guarantees, or existing validation.
3. **Verify the severity** — is this really critical, or is it a cosmetic issue
   dressed up as a bug?
4. **Check for duplicates** — multiple specialists may report the same issue
   in different words.
5. **Assess confidence** — is the specialist making assumptions about runtime
   behaviour without evidence?

For each finding, output one of:
- **VERIFIED** — the issue is real and correctly categorised
- **DOWNGRADED** — the issue exists but severity/confidence should be lower (explain why)
- **DISMISSED** — the issue is a false positive (explain why)
- **DUPLICATE** — already covered by another finding (reference which one)

## Step 4 — Synthesise Final Report

Collect all VERIFIED and DOWNGRADED findings. Produce a final review report:

### Report Format

```markdown
## PR Review: <PR title>

### Summary
<2-3 sentence overview of the PR and overall assessment>

### Critical / High Issues
<Only VERIFIED findings with severity critical or high>

### Medium Issues
<VERIFIED findings with severity medium>

### Suggestions
<DOWNGRADED findings and low-severity items, briefly>

### What Looks Good
<Positive observations — good patterns, thorough tests, clean design>

### Verdict
<One of: APPROVE / REQUEST_CHANGES / COMMENT>
<Brief justification>
```

### Rules for the Final Report
- Lead with the most important issues
- Be specific: include file paths, line numbers, and code snippets
- Be constructive: every criticism must include a concrete suggestion
- Acknowledge what's done well — reviews should be balanced
- If no critical/high issues exist, lean towards APPROVE
- Use the project's conventions (British English, emojis for emphasis)

## Important Guidelines

- **Do NOT make changes to code** — this is a read-only review
- **Do NOT post the review to GitHub** unless explicitly asked
- **Be thorough but not noisy** — quality over quantity
- **Respect the author's intent** — understand why before criticising what
- Each specialist agent should use `subagent_type: "Explore"` for efficient codebase reading
- The verifier agent should use `subagent_type: "general-purpose"` for deeper reasoning
- When spawning agents, always include the full diff and context in the prompt — agents have no memory of this conversation
