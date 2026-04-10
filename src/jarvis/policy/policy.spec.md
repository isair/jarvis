# Policy Package Specification

## Purpose

Evaluates every tool invocation before it is executed and produces a
`PolicyDecision` that the reply engine checks before dispatching to the
`ToolRunner`. The policy engine enforces workspace confinement and
provides a kill-switch to disable all tool execution.

Risk-based approval is handled by the **act-then-undo model** in the
reply engine (see `task_state.spec.md`), not by the policy engine.

## Architecture

```
reply/engine.py
    │
    └─ policy.engine.evaluate(tool_name, tool_args)
            │
            ├─ _classify_tool()           → ToolClass
            ├─ approval.assess_risk()     → RiskLevel
            ├─ PathGuard.check()          → constraints / deny
            ├─ _evaluate_mcp_capability()
            └─ PolicyDecision(allowed, constraints, …)
```

## Components

### `engine.py` — `evaluate()`

Central evaluation function. Returns a `PolicyDecision`.

**Evaluation order:**
1. `PolicyMode.DENY` → deny immediately (kill-switch).
2. Classify tool into `ToolClass` via `tool.classify(args)`.
3. Assess risk level via `tool.assess_risk(args)`.
4. File-system operations: run `PathGuard`.
5. MCP tools: check declared capability metadata.
6. Emit `PolicyDecision` — risk handling delegated to act-then-undo model.

`configure(cfg, approval_store)` sets the module-level singleton.

### `models.py`

Value types used throughout the policy package:
- `PolicyMode` — `ACTIVE` (default, act-then-undo), `DENY` (kill-switch)
- `ToolClass` — `INFORMATIONAL`, `READ_ONLY_OPERATIONAL`, `WRITE_OPERATIONAL`, `DESTRUCTIVE`, `EXTERNAL_DELEGATED`
- `RiskLevel` — re-exported from `tools.types` (`SAFE`, `MODERATE`, `HIGH`)
- `PolicyDecision` — `allowed`, `constraints`, `denied_reason`, `tool_class`, `risk_level`
- `PolicyDeniedError` — raised when `allowed=False` and caller calls `assert_allowed()`
- `AppliedConstraint`, `AccessMode`, `NetworkClass`

### `approvals.py` — `ApprovalStore`

Durable store for scoped approval grants. Currently unused by the policy
engine (approval gates have been removed for the voice-first model) but
retained for potential future use (e.g. MCP server trust grants).

### `path_guard.py` — `PathGuard`

Validates file-system paths against operator-defined root lists:
- `workspace_roots` — allowed path prefixes (deny if outside all roots)
- `blocked_roots` — explicitly forbidden prefixes (deny if within any)
- `read_only_roots` — write/delete denied; read/list permitted

All paths are resolved to absolute canonical form before comparison.
Configuration values are cached at `PathGuard.__init__()` time.

## Configuration Fields Used

| Field              | Type        | Default       | Effect                                     |
|--------------------|-------------|---------------|--------------------------------------------|
| `policy_mode`      | `str`       | `"active"`    | `"active"` or `"deny"` (legacy names accepted) |
| `workspace_roots`  | `list[str]` | `[]`          | Allowed file-system roots                  |
| `blocked_roots`    | `list[str]` | `[]`          | Explicitly forbidden path prefixes         |
| `read_only_roots`  | `list[str]` | `[]`          | Read-only path prefixes                    |
| `local_files_mode` | `str`       | `"home_only"` | Extra constraint for local file tool       |

## Graceful Degradation

When the policy engine is not configured (daemon started without calling
`configure()`), `get_engine()` returns `None`. The reply engine treats a
`None` engine as permissive so that existing deployments without explicit
policy configuration are unaffected.
