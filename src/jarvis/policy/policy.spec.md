# Policy Package Specification

## Purpose

Evaluates every tool invocation before it is executed and produces a
`PolicyDecision` that the reply engine checks before dispatching to the
`ToolRunner`. The policy engine is the primary enforcement point for
workspace confinement, risk-based approval, and operator-defined rules.

## Architecture

```
reply/engine.py
    │
    └─ policy.engine.evaluate(tool_name, tool_args)
            │
            ├─ _classify_tool()           → ToolClass
            ├─ approval.assess_risk()     → RiskLevel (legacy bridge)
            ├─ PathGuard.check()          → constraints / deny
            ├─ _approval_required_for_mode()
            ├─ ApprovalStore.is_granted() → existing grant covers?
            └─ PolicyDecision(allowed, approval_required, constraints, …)
```

## Components

### `engine.py` — `evaluate()`

Central evaluation function. Called with `(tool_name, tool_args, cfg,
approval_store, path_guard)`. Returns a `PolicyDecision`.

**Evaluation order:**
1. `PolicyMode.DENY_ALL` → deny immediately.
2. Classify tool into `ToolClass`.
3. Assess legacy risk level (bridges `approval.RiskLevel`).
4. `PolicyMode.ALWAYS_ALLOW` → allow with no further checks.
5. File-system operations: run `PathGuard`.
6. Determine whether approval is required for the current mode + class.
7. Check `ApprovalStore` for an existing un-expired grant.
8. Emit `PolicyDecision`.

`configure(cfg, approval_store)` sets the module-level singleton accessed
by the reply engine via `get_engine()`.

### `models.py`

Value types used throughout the policy package:
- `PolicyMode` — `ALWAYS_ALLOW`, `ASK_DESTRUCTIVE`, `ASK_WRITE`, `DENY_ALL`
- `ToolClass` — `INFORMATIONAL`, `READ_ONLY_OPERATIONAL`, `WRITE_OPERATIONAL`, `DESTRUCTIVE`, `EXTERNAL_DELEGATED`
- `RiskLevel` — `SAFE`, `MODERATE`, `HIGH`
- `PolicyDecision` — `allowed`, `approval_required`, `constraints`, `denied_reason`, `tool_class`, `risk_level`
- `PolicyDeniedError` — raised when `allowed=False` and caller calls `assert_allowed()`
- `AppliedConstraint`, `AccessMode`, `NetworkClass`

### `approvals.py` — `ApprovalStore`

Durable store for scoped approval grants.  Grants cover a
`(tool_name, operation, path_prefix)` triple where any component may be
`"*"` (wildcard).

**Session scoping:** The store accepts a `default_ttl_sec` parameter
(default 3 600 s). Every new grant receives an expiry computed as
`granted_at + default_ttl_sec` unless the caller provides an explicit
`expires_at`.  This prevents grants from persisting indefinitely across
daemon restarts when SQLite backing is enabled.

Call `prune_expired()` periodically (or at startup) to remove stale rows.

### `path_guard.py` — `PathGuard`

Validates file-system paths against operator-defined root lists:
- `workspace_roots` — allowed path prefixes (deny if outside all roots)
- `blocked_roots` — explicitly forbidden prefixes (deny if within any)
- `read_only_roots` — write/delete denied; read/list permitted

All paths are resolved to absolute canonical form before comparison.

## Configuration Fields Used

| Field              | Type        | Default              | Effect                                     |
|--------------------|-------------|----------------------|--------------------------------------------|
| `policy_mode`      | `str`       | `"ask_destructive"`  | Overall policy strictness                  |
| `workspace_roots`  | `list[str]` | `[]`                 | Allowed file-system roots                  |
| `blocked_roots`    | `list[str]` | `[]`                 | Explicitly forbidden path prefixes         |
| `read_only_roots`  | `list[str]` | `[]`                 | Read-only path prefixes                    |
| `local_files_mode` | `str`       | `"workspace"`        | Extra constraint for local file tool       |

## Graceful Degradation

When the policy engine is not configured (daemon started without calling
`configure()`), `get_engine()` returns `None`.  The reply engine treats a
`None` engine as `ALWAYS_ALLOW` so that existing deployments without
explicit policy configuration are unaffected.
