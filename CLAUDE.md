Data privacy comes first, always.

All user-facing command line output should make use of emojis. Especially an initial emoji to start off the lines that depict what the line is about. Output should make use of indentation spacing to establish a visual hierarchy and aim to make output as easy to sift through as possible. Exception: Windows .bat scripts cannot use emojis (cmd.exe doesn't render Unicode properly).

Any important point in our logical flows should have debug logs using the `debug_log` method from `src/jarvis/debug.py`. Avoid excessive logging to keep the logs easily readable and actionable.

Any code change must either adhere to our spec files perfectly or you should ask the user to confirm changes, which should also propagate to the specs themselves. Spec files follow the \*.spec.md format and live next to the code that implements them. Always search for related spec files before starting any work. When corrected about how something should work, check if there's a spec for it and whether it needs updating.

### Spec File Registry

| Spec file | Covers | Key principles |
|-----------|--------|----------------|
| `src/desktop_app/desktop_app.spec.md` | System tray app, startup flow, daemon integration, windows, theme, updates | Desktop is separate from core; jarvis has no knowledge of desktop_app |
| `src/desktop_app/settings_window.spec.md` | Auto-generated settings UI from config metadata | Metadata-driven; only non-default values written; preserves unknown keys |
| `src/desktop_app/setup_wizard.spec.md` | First-run wizard (Ollama, models, Whisper, location) | Minimal friction; only shown when user action required; doesn't configure everything |
| `src/jarvis/dictation/dictation.spec.md` | Hold-to-dictate engine, hotkey, clipboard paste | Independent from assistant pipeline; shared Whisper model; pause flag on listener |
| `src/jarvis/listening/listening.spec.md` | Voice listener, wake word detection, audio pipeline | — |
| `src/jarvis/reply/reply.spec.md` | LLM reply generation, tool use, profiles | Tools return raw data; profiles handle formatting |
| `src/jarvis/reply/prompts/prompts.spec.md` | System/user prompt templates | — |
| `src/jarvis/utils/location.spec.md` | GeoIP location detection | Privacy-first; local GeoLite2 DB only |
| `src/jarvis/memory/graph.spec.md` | Node graph memory (v2), self-organising tree, UI explorer | Dynamic structure; access-aware; auto-split/merge (future) |

Avoid hardcoded language patterns as this assistant needs to support an arbitrary amount of different languages.

Tools define when/how to be used and return raw data without LLM processing. The unified system prompt in `src/jarvis/system_prompt.py` handles response formatting and personality through the daemon's LLM loop.

## Git Workflow

The default branch is `develop`. All PRs and feature branches must target `develop`, not `main`.

Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages and PR titles (e.g. `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`).

When pushing commits to a PR, always update the PR title and body to cover the entire changeset.

Squash-merged commits on `develop` should only carry the PR number in the title (e.g. `(#171)`), never the originating issue number. Issue references belong in the commit body as `Closes #NNN` so that they auto-close when the commit reaches `main` on release.

## Issue Triage

Never close an issue because its fix landed on `develop`. GitHub auto-closes issues when the `Closes #NNN` commit reaches `main` during a release, so a manual close on a develop-only fix is premature and drops the auto-close signal. The only legitimate manual closures are: duplicates (reference the canonical issue), insufficient info, user-environment issues, or not-a-bug.

## Releases

"Release" means fast-forwarding `main` to the current tip of `develop` and pushing it. No merge commit, no force push — just `git checkout main && git merge --ff-only develop && git push origin main`. This is what triggers the release workflow and the auto-close of issues referenced by `Closes #NNN` in the develop commits.

## Development Environment

The project uses a micromamba environment at `.mamba_env/`. Always activate it before running builds, tests, or the app:

```bash
eval "$(micromamba.exe shell hook --shell bash)" && micromamba activate "C:/Users/baris/projects/jarvis/.mamba_env"
```

## README Maintenance

Keep README.md up-to-date when making changes that affect user-facing functionality. Update the README when:
- Adding or removing built-in tools (update Features → Built-in Tools list)
- Changing configuration options (update Configuration section)
- Adding new MCP integration examples
- Changing system requirements or installation steps
- Fixing or introducing known limitations

README priorities (in order of importance):
1. **Privacy-first messaging** - The local/offline nature is a core selling point
2. **Quick install** - Users should get running in minutes
3. **Features list** - High-level capabilities at a glance
4. **Known limitations** - Be transparent about what doesn't work yet
5. **Configuration** - Only document options users actually need
6. **MCP integrations** - Examples for popular tools
7. **Troubleshooting** - Common issues with solutions

Keep sections concise. Use collapsible `<details>` for lengthy content. Avoid documenting internal implementation details - the README is for end users, not developers.

---

When the user says "remember" something, add it to CLAUDE.md in the appropriate section (project-specific above the ---, or portable below).

Run your changes and test them manually, iterate until everything is good.

Always use TDD: write failing tests first, then implement the fix. Tests should verify **behaviours**, not implementation details. Test what the system does (observable outcomes), not how it does it (internal state, mock call counts, etc.).

Ensure all your changes are covered by all appropriate form of automated tests - unit, integration, visual regression, evals, etc.

Tests should verify mechanisms, not current values. Assert against config-driven or computed references rather than hardcoding specifics that change between migrations.

Run evals after finalising a change that can affect agent accuracy.

Any change to LLM prompts (system prompts, tool incentives, constraints, etc.) must be verified against a relevant eval case. If no eval exists for the behaviour being changed, write one first. The eval should demonstrate the improvement — i.e. it should fail or show worse results before the prompt change and pass or improve after.

Commit your changes when you finish a fix or feature before moving on to the next task.

Before running `git commit --amend`, always check `git log --oneline -3` first to verify you're amending the correct commit.

Always use British English everywhere (e.g. "colour" not "color", "behaviour" not "behavior", "initialise" not "initialize").
