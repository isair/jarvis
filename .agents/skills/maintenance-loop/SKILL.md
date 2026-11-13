---
name: maintenance-loop
description: >
  Run or resume the Jarvis reliability and maintenance loop across open issues
  and pull requests. Prioritise core product quality, general fixes and
  simplification, then validate, privately review and merge authorised work.
---

# Jarvis maintenance loop

The primary goal is a high-quality, reliable, coherent Jarvis product. Reaching
zero open issues and pull requests is secondary. Resolve genuine problems,
consolidate duplicates and identify unsuitable or superseded work without
closing valid defects, hiding blockers, weakening assertions or merging
unqualified features to improve queue counts.

## Resume safely

1. Read `AGENTS.md`, the current conversation and the existing checkpoint.
   Use paths and scope supplied by the user or automation. If no checkpoint
   exists, create a durable task artifact outside the source tree and record
   its location. Keep machine-specific state out of this skill.
2. Reuse the loop's isolated worktree. Inspect its branch, status and ownership
   before changing anything. Preserve unfinished work and another session's
   changes; do not switch or reset their checkout. If no suitable worktree
   exists, create an isolated one using the available worktree tooling.
3. Follow the user's active usage policy. Apply a budget or percentage cutoff
   only when requested. When the user relies on automatic resets, continue
   across weekly usage thresholds. Do not manually consume reset credits or
   purchase capacity without authorisation. If ordinary usage actually becomes
   unavailable, checkpoint and pause the associated heartbeat through the
   automation tool, then report completed work and what remains.
4. Refresh live GitHub state and `origin/develop`. Treat checkpoint facts as
   leads, not evidence of current heads, checks or issue status.

This skill organises work within the user's current authorisation. It does
not itself authorise merges, releases, public messages or unrelated work.
Continue already authorised routine actions without asking again. Keep public
replies, reviews and closure messages as friendly private drafts for the user
to send, including thanks to contributors.

## Sweep issues and pull requests

Use [triage](../triage/SKILL.md) for issue and discussion diagnosis, subject to
the private-draft boundary above. Fetch the complete open issue and PR queues
with pagination, including drafts and community contributions. Check threaded
discussion replies when relevant. Read attachments and logs before diagnosing;
flag unavailable evidence explicitly.

For every PR, read its full description, diff, linked issues, review threads
and current checks. Compare it with current `develop` for resolved or
superseded behaviour, duplicates, overlapping changes, stacked dependencies,
specification and privacy fit, maintenance cost and core product impact.
Preserve the source SHA and evidence location. Unchanged evidence packets may
reuse a documented review; do not claim a new full review or reuse approval
after a material head or base change.

Give each retained issue and PR a reason, next action and readiness or decision
gate. Useful dispositions are:

- Review or repair now.
- Await contributor evidence or a user decision.
- Retain an experimental draft with explicit qualification gates.
- Propose closure as duplicate, superseded or out of scope, with a private draft.
- Resolved on `develop`, awaiting release and the repository's auto-close flow.

Never treat draft status as readiness or merge unrelated dependencies blindly.
Keep valid unresolved defects open. Do not release merely to clear issue counts.

## Choose and complete the next batch

Prefer confirmed core failures and useful fixes or simplifications already
proposed in PRs over optional features. Assess user impact, confidence in the
reproduction, effort, regression risk and dependencies. Meaningful test
consolidation and removal of unnecessary code are worthwhile outcomes; fewer
tests alone is not evidence of improvement.

Apply the **Product vision and scope** gate in `AGENTS.md` before selecting
implementation or repair work. Record a brief product-fit rationale alongside
the retained item's disposition: user benefit, fit with the shared voice/chat
experience, simplest existing alternative and material complexity cost. Reassess
optional or overlapping proposals before spending effort making them mergeable;
technical repairability alone does not make a feature worth adding. Keep genuine
core defects actionable while scope decisions are considered.

For each independent change:

1. Establish the reproduction and read related `*.spec.md` files. Use TDD for
   code changes: demonstrate failing observable behaviour before implementing
   the general solution. Follow the repository's privacy, language and single
   source of truth requirements.
2. Before any app tests or probes, read
   [probe safety](references/probe-safety.md). Qualify isolation before broad
   runs. Never edit the user's diary, graph, configuration or browser profiles
   as test data. Keep failed memory prototypes held and their assertions intact.
3. Implement the smallest complete solution. Update affected specs, technical
   documentation and `docs/llm_contexts.md` as required by `AGENTS.md`. Keep the
   README concise and about user outcomes. If the solution changes a specified
   contract, obtain the required user confirmation before merging; present a
   concrete reviewable proposal and identify the affected spec.
4. Run appropriate automated and manual checks. For changes affecting agent
   accuracy, run relevant real local evals. Prompt changes require an eval
   demonstrating the baseline and improvement, adding a case first if needed.
   Never relax assertions, exclude a supported model or confuse a warmup,
   mocked response or unit pass with live accuracy qualification.
5. Commit a completed fix before moving to another. Keep independent changes
   in separate Conventional Commit PRs targeting `develop`. Website-only work
   uses `docs:` or `chore:`; GitHub Pages deploys only from `main`. Keep each
   PR title and body accurate for its entire final changeset and attach every
   created PR using available task tooling.

## Review and merge gate

Run [review-pr](../review-pr/SKILL.md) privately after creating a PR and for
eligible existing PRs. Use [review-pr-ultra](../review-pr-ultra/SKILL.md) when
the change warrants its deeper review. Keep review outputs as private task
artifacts; the loop's draft-only boundary overrides offers to post them.

Merge only within existing user authorisation, after all of these are true:

- The final changeset passes `AGENTS.md`'s product-fit gate. Its user benefit
  justifies the added complexity; unresolved product-scope questions block
  feature merges even when technical checks pass.
- The final exact head SHA has no unresolved review findings and is qualified
  by appropriate tests, manual checks and relevant live evals.
- The PR is ready, targets `develop`, and its current head matches the reviewed
  SHA. Derive expected checks from the applicable workflows and requirements,
  not just the currently visible rollup. All expected checks and their complete
  workflow runs have finished successfully.
- Refresh both PR and base immediately before merging. Qualification covers
  the current `develop` base. If the head or base changes, reassess, refresh
  affected tests and review before proceeding. Use an exact-head merge guard.

Verify the resulting merge live. Follow the repository squash-title and issue
reference conventions. Failed or flaky checks need diagnosis and repair, not
an unrecorded bypass. A blocked item does not prevent independent eligible work.

## Checkpoint, compact and continue

After each merge, update and compact the checkpoint with the resulting commit,
reviewed head and base, private review, tests/evals and CI evidence, queue
dispositions, held experiments and next actions. Preserve ownership and cleanup
instructions for active processes. Archive detailed evidence without losing
links, failed outcomes or decision gates.

Then refresh usage and live state and start the next actionable batch in the
same wake. A merge is a checkpoint, not a stopping condition. Continue until
there is no eligible work, a required decision blocks all useful progress,
the usage gate is reached, or the user stops or redirects the work. Record the
specific stopping reason rather than promising unexecuted background work.

Release only when authorised, with meaningful batched progress, validation and
observation time. Avoid multiple routine stable releases in a day. Follow
`AGENTS.md` for the fast-forward-only release procedure; never rewrite history
or release unrelated work as routine loop activity.

For scheduled wakes, stay quiet on unchanged or non-actionable state. Notify
only meaningful progress, completion, failure or a required decision using the
host's heartbeat format. Do not create a schedule merely because this skill
was invoked. Preserve the existing schedule and notification preferences when
updating an authorised automation.
