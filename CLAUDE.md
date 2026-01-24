Data privacy comes first, always.

All user-facing command line output should make use of emojis. Especially an initial emoji to start off the lines that depict what the line is about. Output should make use of indentation spacing to establish a visual hierarchy and aim to make output as easy to sift through as possible.

Any important point in our logical flows should have debug logs using the `debug_log` method from `src/jarvis/debug.py`. Avoid excessive logging to keep the logs easily readable and actionable.

Any code change must either adhere to our spec files perfectly or you should ask the user to confirm changes, which should also propagate to the specs themselves. Spec files follow the \*.spec.md format and live next to the code that implements them. Always search for related spec files before starting any work.

Avoid hardcoded language patterns as this assistant needs to support an arbitrary amount of different languages.

Tools define when/how to be used. Profiles define what to do after tools execute. Keep these concerns separate in `tools.py` and `profiles.py`.

Tools return raw data without LLM processing. Profiles handle all response formatting and personality through the daemon's LLM loop. This ensures consistent response style across all profiles.

Always use the shared theme for UI under `src/jarvis/themes.py`.

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

Ensure all your changes are covered by all appropriate form of automated tests - unit, integration, visual regression, evals, etc.

Run evals after finalising a change that can affect agent accuracy.

Commit your changes when you finish a fix or feature before moving on to the next task.

Before running `git commit --amend`, always check `git log --oneline -3` first to verify you're amending the correct commit.
