# Audit Package Specification

## Purpose

Records a durable, tamper-evident log of every task, tool execution, policy
decision, and approval event to a SQLite database. The audit system is
designed to support post-hoc investigation and operator accountability
without compromising user privacy.

## Privacy

All text stored in the audit database originates from the reply engine's
_redacted_ text path. The same PII-scrubbing that removes emails, tokens,
and passwords from the dialogue is applied before the intent string reaches
the audit recorder. No raw user input is written to disk.

The audit database is an _additional_ SQLite file (default: `audit.db` in
the same directory as the main database). Its existence should be
documented to users. Auditing is entirely opt-in: leaving `audit_db_path`
unset in the configuration disables the recorder and all calls to
`get_recorder()` return `None`.

## Components

### `db.py` — `AuditDB`

Low-level SQLite wrapper. Creates and migrates the schema on first open.

**Tables:**

| Table              | Description                                        |
|--------------------|----------------------------------------------------|
| `tasks`            | One row per user request (intent, profile, tools)  |
| `task_steps`       | One row per tool invocation within a task          |
| `policy_decisions` | One row per policy evaluation (allow/deny reasons) |
| `approvals`        | One row per user approval grant or denial          |

All timestamps are stored as UNIX epoch floats.

### `recorder.py` — `AuditRecorder`

High-level facade. Provides `begin_task()`, `finish_task()`,
`record_step()`, `record_policy_decision()`, `record_approval()`.

All methods are safe to call when the database is unavailable — they log a
debug message and return silently. This ensures a database error never
crashes the reply engine.

**Module-level singleton:**

```python
from jarvis.audit.recorder import configure, get_recorder

configure("/path/to/audit.db")   # call once at daemon startup
recorder = get_recorder()        # returns None if not configured
```

### `models.py`

Pure dataclasses used as arguments to `AuditRecorder` methods:

- `TaskRecord` — intent, request_type, profile, status
- `TaskStepRecord` — tool_name, args_hash, success, duration_ms
- `PolicyDecisionRecord` — tool_class, risk_level, allowed, constraints_json
- `ApprovalRecord` — tool_name, operation, path_prefix, decision, expires_at

## Configuration Fields Used

| Field          | Type  | Default | Effect                                          |
|----------------|-------|---------|-------------------------------------------------|
| `audit_db_path`| `str` | `None`  | Path to the SQLite audit database. If omitted, auditing is disabled. Defaults to `audit.db` alongside the main DB when set up via the bootstrap. |

## Lifecycle

```
daemon startup
  └─ configure(audit_db_path)       sets module singleton

reply engine — per request
  ├─ recorder.begin_task(…)
  ├─ [per tool call] recorder.record_step(…)
  │                  recorder.record_policy_decision(…)
  │                  recorder.record_approval(…)           (if approval sought)
  └─ recorder.finish_task(task_id, final_status)

daemon shutdown
  └─ recorder.close()
```
