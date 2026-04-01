# Task State & Approval Spec

This document specifies the task state tracker and approval logic introduced to implement the JARVIS autonomy requirements.

## Modules

| Module | Path |
|--------|------|
| Task State | `src/jarvis/task_state.py` |
| Approval | `src/jarvis/approval.py` |

---

## 1. Task State (`task_state.py`)

### Purpose

Tracks the active task during multi-step workflow execution so that:
- The desktop console can display real-time progress.
- The engine can detect when a task is resumable within a session.
- Debug logs contain a structured summary of what was executed.

### Lifecycle

```
IDLE → PLANNING → EXECUTING → DONE
                            → FAILED
                → AWAITING_APPROVAL → (re-issued by user) → EXECUTING
```

### TaskStatus

| Value | Meaning |
|-------|---------|
| `IDLE` | No active task |
| `PLANNING` | Intent recorded, steps not yet running |
| `EXECUTING` | Tool steps are executing |
| `AWAITING_APPROVAL` | Halted pending user confirmation |
| `DONE` | Completed successfully |
| `FAILED` | Terminated with an error |

### TaskStep

Each tool execution is recorded as a `TaskStep`:

| Field | Type | Description |
|-------|------|-------------|
| `description` | str | Human-readable step description |
| `tool_name` | Optional[str] | Tool identifier |
| `status` | StepStatus | PENDING / RUNNING / SUCCEEDED / FAILED / SKIPPED |
| `result_summary` | Optional[str] | First 120 chars of result or error |
| `started_at` | Optional[float] | Unix timestamp when step started |
| `finished_at` | Optional[float] | Unix timestamp when step completed |

### Module-Level Singleton

A single `TaskState` instance is maintained per daemon session:

```python
from jarvis.task_state import begin_task, get_active_task, reset_task
```

- `begin_task(intent)` – resets and begins a new task; returns the singleton.
- `get_active_task()` – returns the current singleton (thread-safe).
- `reset_task()` – returns to IDLE state.

### Resumption

`TaskState.can_resume()` returns `True` when the task is `EXECUTING` or `AWAITING_APPROVAL` and at least one step is still `PENDING`. This supports within-session task resumption.

---

## 2. Approval (`approval.py`)

### Purpose

Implements the **Decision Policy**:
- Act automatically on clear, specific instructions.
- Require approval for destructive or high-impact actions.

### Risk Levels

| Level | Meaning | Requires Approval |
|-------|---------|-------------------|
| `SAFE` | Read-only / reversible | No |
| `MODERATE` | Writes that are easily undone | No |
| `HIGH` | Destructive or hard to undo | **Yes** |

### Per-Tool Risk Table

| Tool | Risk |
|------|------|
| `screenshot` | SAFE |
| `recallConversation` | SAFE |
| `fetchMeals` | SAFE |
| `webSearch` | SAFE |
| `fetchWebPage` | SAFE |
| `getWeather` | SAFE |
| `refreshMCPTools` | SAFE |
| `stop` | SAFE |
| `logMeal` | MODERATE |
| `deleteMeal` | **HIGH** |
| `localFiles` (list/read) | SAFE |
| `localFiles` (write/append) | MODERATE |
| `localFiles` (delete) | **HIGH** |
| MCP tools | MODERATE (default) |
| Unknown tools | MODERATE (cautious default) |

### Request Classification

`classify_request(text)` returns `RequestType.OPERATIONAL` or `RequestType.INFORMATIONAL` using a keyword heuristic. This guides the engine on whether side-effects are expected, without changing core execution behaviour.

### Approval Flow

1. Engine extracts `tool_name` and `tool_args` from LLM response.
2. Calls `requires_approval(tool_name, tool_args)`.
3. If `True`: sets task to `AWAITING_APPROVAL`, returns `approval_prompt(...)` as the reply.
4. Execution halts; user must explicitly re-issue the command.
5. If `False`: execution proceeds automatically.

### Public API

```python
from jarvis.approval import (
    RiskLevel,          # Enum: SAFE, MODERATE, HIGH
    RequestType,        # Enum: INFORMATIONAL, OPERATIONAL
    assess_risk,        # (tool_name, tool_args) -> RiskLevel
    requires_approval,  # (tool_name, tool_args) -> bool
    approval_prompt,    # (tool_name, tool_args) -> str
    classify_request,   # (text) -> RequestType
)
```

---

## 3. Integration with Reply Engine

The reply engine (`src/jarvis/reply/engine.py`) integrates both modules:

```
run_reply_engine(...)
  ├─ redact(text)
  ├─ classify_request(redacted)     ← approval.py
  ├─ begin_task(redacted)           ← task_state.py
  ├─ [profile selection, enrichment, messages build]
  ├─ task.set_executing()
  └─ agentic loop:
       └─ for each tool call:
            ├─ if requires_approval(tool, args):
            │    task.set_awaiting_approval()
            │    return approval_prompt(...)   ← halts, no tool run
            ├─ step = task.add_step(...)
            ├─ step.start()
            ├─ run_tool_with_retries(...)
            └─ step.complete() or step.fail()
  └─ task.complete() or task.fail()
```

---

## 4. Testing

- **`tests/test_task_state.py`** – unit tests for `TaskState`, `TaskStep`, and the module-level singleton.
- **`tests/test_approval.py`** – unit tests for `assess_risk`, `requires_approval`, `approval_prompt`, and `classify_request`.
