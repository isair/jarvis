# Task State, Approval & Undo Spec

This document specifies the task state tracker, risk assessment, and undo
registry introduced to implement the JARVIS autonomy requirements.

## Modules

| Module | Path |
|--------|------|
| Task State | `src/jarvis/task_state.py` |
| Approval / Risk | `src/jarvis/approval.py` |
| Undo Registry | `src/jarvis/undo_registry.py` |

---

## 1. Task State (`task_state.py`)

### Purpose

Tracks the active task during multi-step workflow execution so that:
- The desktop console can display real-time progress.
- The engine can detect when a task is resumable within a session.
- Debug logs contain a structured summary of what was executed.
- Reversible actions are visible in the task history.

### Lifecycle

```
IDLE -> PLANNING -> EXECUTING -> DONE
                             -> REVERSIBLE   (completed; undo window open)
                             -> FAILED
```

`REVERSIBLE` is a terminal variant of `DONE`. The task completed successfully
but at least one step has a matching `UndoEntry` in the registry that has not
yet expired.

### TaskStatus

| Value | Meaning |
|-------|---------|
| `IDLE` | No active task |
| `PLANNING` | Intent recorded, steps not yet running |
| `EXECUTING` | Tool steps are executing |
| `REVERSIBLE` | Completed; last action(s) can still be undone |
| `DONE` | Completed successfully, no undo pending |
| `FAILED` | Terminated with an error |

> `TaskStatus.AWAITING_APPROVAL` is kept as a backwards-compat alias for
> `TaskStatus.REVERSIBLE`. New code should use `REVERSIBLE`.

### StepStatus

| Value | Meaning |
|-------|---------|
| `PENDING` | Step queued, not yet started |
| `RUNNING` | Step is executing |
| `SUCCEEDED` | Completed successfully (no undo registered) |
| `REVERSIBLE` | Completed successfully; `UndoEntry` registered in registry |
| `REVERSED` | Was successfully undone by the user |
| `FAILED` | Terminated with an error |
| `SKIPPED` | Deliberately bypassed |

### TaskStep

Each tool execution is recorded as a `TaskStep`:

| Field | Type | Description |
|-------|------|-------------|
| `description` | str | Human-readable step description |
| `tool_name` | Optional[str] | Tool identifier |
| `status` | StepStatus | Current lifecycle status |
| `result_summary` | Optional[str] | First 120 chars of result or error |
| `started_at` | Optional[float] | Unix timestamp when step started |
| `finished_at` | Optional[float] | Unix timestamp when step completed |
| `step_id` | str | UUID hex -- links this step to its `UndoEntry` |
| `reversible` | bool | True when a matching `UndoEntry` exists in the registry |
| `undo_entry_id` | Optional[str] | `UndoEntry.step_id` of the registered undo |

**Methods:**
- `start()` -- transition to RUNNING
- `complete(result_summary)` -- transition to SUCCEEDED
- `mark_reversible(undo_entry_id)` -- transition to REVERSIBLE after undo pushed
- `mark_reversed(result_summary)` -- transition to REVERSED after undo executed
- `fail(reason)` -- transition to FAILED
- `skip(reason)` -- transition to SKIPPED

### TaskState methods

| Method | Description |
|--------|-------------|
| `begin(intent)` | Reset and start a new task |
| `set_executing()` | Transition PLANNING -> EXECUTING |
| `set_reversible()` | Transition EXECUTING -> REVERSIBLE (at least one reversible step) |
| `complete()` | Transition EXECUTING -> DONE |
| `fail(reason)` | Transition -> FAILED |
| `reset()` | Return to IDLE |
| `can_resume()` | True when EXECUTING and pending steps remain |
| `can_undo()` | True when REVERSIBLE and at least one REVERSIBLE step exists |
| `add_step(...)` | Append a new PENDING step |

### Properties

| Property | Description |
|----------|-------------|
| `completed_steps` | Steps with status SUCCEEDED or REVERSIBLE |
| `reversible_steps` | Steps with status REVERSIBLE |
| `failed_steps` | Steps with status FAILED |

### Module-Level Singleton

```python
from jarvis.task_state import begin_task, get_active_task, reset_task
```

- `begin_task(intent)` -- resets and begins a new task; returns the singleton.
- `get_active_task()` -- returns the current singleton (thread-safe).
- `reset_task()` -- returns to IDLE state.

---

## 2. Approval / Risk (`approval.py`)

### Purpose

Implements the **Decision Policy**:
- Act automatically on clear, specific instructions.
- Warn the user before truly irreversible destructive actions.
- Register an undo entry after reversible destructive actions.
- Never block execution waiting for approval -- Jarvis is voice-first and hands-free.

### Risk Levels

| Level | Meaning | Engine behaviour |
|-------|---------|-----------------|
| `SAFE` | Read-only / clearly reversible | Execute silently |
| `MODERATE` | Writes that are easily undone | Execute; register undo silently |
| `HIGH` + undoable | Destructive, state can be restored | Execute; register undo; speak "say undo" note |
| `HIGH` + irreversible | Destructive, state cannot be restored | Speak brief warning, then execute |

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
| `deleteMeal` | HIGH |
| `localFiles` (list/read) | SAFE |
| `localFiles` (write/append) | MODERATE |
| `localFiles` (delete) | HIGH |
| MCP tools | MODERATE (default) |
| Unknown tools | MODERATE (cautious default) |

### Undoable Operations

| Tool / Operation | Undoable | Requires Snapshot |
|-----------------|----------|-------------------|
| `localFiles / write` | Yes | Yes (original file content) |
| `localFiles / append` | Yes | Yes (original file content) |
| `localFiles / delete` | Yes | Yes (file content before deletion) |
| `logMeal` | No (future work) | -- |
| `deleteMeal` | No (future work) | -- |
| All MCP tools | No | -- |

### Public API

```python
from jarvis.approval import (
    RiskLevel,               # Enum: SAFE, MODERATE, HIGH
    RequestType,             # Enum: INFORMATIONAL, OPERATIONAL
    assess_risk,             # (tool_name, tool_args) -> RiskLevel
    is_undoable,             # (tool_name, tool_args) -> bool
    pre_execution_warning,   # (tool_name, tool_args) -> Optional[str]
    post_execution_note,     # (tool_name, tool_args) -> Optional[str]
    build_undo_args,         # (tool_name, tool_args, snapshot) -> Optional[tuple]
    requires_approval,       # Deprecated -- retained for policy engine compat
    approval_prompt,         # Deprecated -- retained for audit compat
    classify_request,        # (text) -> RequestType
)
```

---

## 3. Undo Registry (`undo_registry.py`)

### Purpose

Maintains a session-scoped, time-bounded stack of reversible operations so
that users can say "undo that" or "undo the last 3 actions" after a task
completes.

### UndoEntry

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | str | Matches the `TaskStep.step_id` that created it |
| `description` | str | Human-readable, e.g. "deleted shopping_list.txt" |
| `tool_name` | str | Tool that was executed |
| `tool_args` | dict | Original arguments |
| `undo_tool` | str | Tool to call for reversal |
| `undo_args` | dict | Arguments for the reversal call |
| `snapshot` | Optional[Any] | State captured before execution |
| `created_at` | float | Unix timestamp |
| `expires_at` | float | Unix timestamp; default +300 seconds |

### UndoRegistry methods

| Method | Description |
|--------|-------------|
| `push(entry)` | Add entry; prune expired; cap at MAX_ENTRIES (20) |
| `pop_last(n=1)` | Remove and return last n non-expired entries, most-recent first |
| `pop_by_id(step_id)` | Remove and return entry with matching step_id |
| `peek_all()` | View all non-expired entries without removing |
| `clear()` | Empty the stack |
| `size` | Count of non-expired entries |

### Singleton access

```python
from jarvis.undo_registry import (
    get_undo_registry,    # -> UndoRegistry singleton
    push_undo,            # (entry) -> None
    pop_last_undo,        # (n=1) -> List[UndoEntry]  (most-recent first)
    pop_undo_by_id,       # (step_id) -> Optional[UndoEntry]
)
```

---

## 4. Integration with Reply Engine

```
run_reply_engine(text, ...)
  |-- redact(text)
  |-- _detect_undo_intent(redacted)       <- short-circuits if undo command
  |    |-- pop_last_undo(n) / pop_undo_by_id(id)
  |    |-- for each entry (reverse-chrono order):
  |    |    `-- run_tool_with_retries(undo_tool, undo_args)
  |    `-- return "Reversed: ..."         <- no LLM involved
  |
  |-- classify_request(redacted)
  |-- begin_task(redacted)
  |-- [profile selection, enrichment, messages build]
  |-- task.set_executing()
  `-- agentic loop:
       `-- for each tool call:
            |-- policy evaluation
            |-- pre_execution_warning()   <- spoken before HIGH+irreversible
            |-- _capture_snapshot()       <- reads original state if undoable
            |-- step = task.add_step(...)
            |-- step.start()
            |-- run_tool_with_retries(tool_name, tool_args)
            |-- step.complete(result)
            |-- [if undoable and snapshot available]:
            |    |-- build_undo_args(tool_name, tool_args, snapshot)
            |    |-- push_undo(UndoEntry(...))
            |    |-- step.mark_reversible(entry.step_id)
            |    `-- _post_notes.append(post_execution_note())
            `-- step.fail(err)
  |-- if task.reversible_steps: task.set_reversible()
  |   else: task.complete()
  `-- reply = pre_warnings + llm_reply + post_notes
```

### Undo intent patterns

| User says | Resolved as |
|-----------|-------------|
| "undo" / "undo that" / "undo it" | Reverse last 1 action |
| "undo last 3 actions" | Reverse last 3 actions (reverse-chrono order) |
| "undo a1b2c3d4" | Reverse the specific step by ID |

---

## 5. Testing

- **`tests/test_task_state.py`** -- unit tests for `TaskState`, `TaskStep`, the singleton, `can_undo()`, reversible/reversed step transitions.
- **`tests/test_approval.py`** -- unit tests for `assess_risk`, `is_undoable`, `pre_execution_warning`, `post_execution_note`, `build_undo_args`, `classify_request`.
- **`tests/test_undo_registry.py`** -- unit tests for `UndoRegistry`: push, pop_last, pop_by_id, expiry, multi-pop ordering, cap enforcement.
