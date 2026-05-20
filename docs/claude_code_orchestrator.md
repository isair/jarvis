# Claude Code as orchestrator (Jarvis)

This repo is meant to be driven by **Claude Code** with a local **Ollama** backend. The working directory is the orchestrator scope: plans, edits, and shell commands apply here.

**Full machine-wide guide:** `%USERPROFILE%\CLAUDE-ORCHESTRATOR.md`

## Launch on this project

```cmd
"%USERPROFILE%\launch-claude-local.bat" "C:\Users\janso\.cursor\projects\empty-window\Jancuks"
```

With a coding-heavy model:

```cmd
"%USERPROFILE%\launch-claude-local.bat" "C:\path\to\jarvis-clone" qwen2.5-coder:14b
```

Prerequisites: Ollama running; `ANTHROPIC_AUTH_TOKEN=ollama` and `ANTHROPIC_BASE_URL=http://localhost:11434` (set globally on your PC).

## What Claude reads automatically

| File | Role |
|------|------|
| `CLAUDE.md` | Project rules, spec registry, test commands, orchestration boundaries |
| `docs/cursor_brief_rust_cafe_agents.md` | Café agent phases and acceptance criteria |
| `*.spec.md` | Module behaviour contracts (must stay in sync with code) |

## Jarvis layout (orchestrator mental model)

```
apps/jarvis_shell/     Tauri UI, spawns :5050, shell_daemon.py
src/jarvis/            Voice, LLM, tools, memory (no Rust imports)
src/desktop_app/       PyQt tray, Flask memory_viewer, settings API
cafe-agent/            Rust sidecar :8787, POST /task
static/sulainis/       Operator café ops UI
```

## Scoped prompts

**Café Rust only:**

```text
Orchestrate only cafe-agent/. Do not change src/jarvis unless the spec requires it.
cargo test --workspace after each step.
```

**Shell only:**

```text
Orchestrate only apps/jarvis_shell/ and Flask settings routes in src/desktop_app/.
Match jarvis_shell.spec.md.
```

**Python Jarvis core:**

```text
Orchestrate only src/jarvis/. Run pytest -m unit on affected tests; update docs/llm_contexts.md if LLM calls change.
```

## Monorepo / extra dirs

From repo root:

```cmd
claude --model qwen3.5 --add-dir cafe-agent --add-dir apps/jarvis_shell
```

## Headless one-shot

```cmd
cd /d C:\path\to\jarvis
claude --model qwen3.5 -p "List open work from docs/cursor_brief_rust_cafe_agents.md section H"
```

## Parallel tracks

```cmd
claude --worktree feat-shell-settings --model qwen3.5
```

Second terminal: `claude --worktree feat-cafe-schedule --model qwen2.5-coder:14b`

## Resume

```cmd
cd /d C:\same\jarvis\clone
claude --continue
```

## Model picks (local)

| Model | Use for |
|-------|---------|
| `qwen3.5` | Default orchestrator (planning + mixed work) |
| `qwen2.5-coder:14b` | Heavy coding / refactors |
| `gpt-oss:20b` | Backup if others are slow |

## Checklist

1. `launch-claude-local.bat "<repo path>"`
2. Trust folder on first run
3. Confirm `CLAUDE.md` and brief are current
4. Start with planning prompt; execute stepwise with tests
5. `claude --continue` for follow-ups
