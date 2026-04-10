# Runtime Package Specification

## Purpose

Provides the service lifecycle infrastructure for the Jarvis daemon:
health tracking, graceful degradation, and coordinated shutdown.

## Components

### `health.py` — `HealthRegistry`

Thread-safe registry that tracks the operational status of every Jarvis
subsystem. Each subsystem transitions through:

```
(not registered) → INITIALISING → READY | DEGRADED | UNAVAILABLE
```

**Module-level singleton:** `configure()` creates and registers the
registry; `get_registry()` returns it. The daemon calls `configure()` once
at startup; all other modules call `get_registry()`.

**Well-known service names** are defined on `ServiceName`:
`DATABASE`, `OLLAMA`, `WHISPER`, `MICROPHONE`, `TTS`, `MCP`, `LOCATION`,
`POLICY`, `AUDIT`, `VOICE`.

`health.summary()` returns a one-line human-readable status string
suitable for log output.

### `shutdown_manager.py` — `ShutdownManager`

Coordinates orderly shutdown:
1. Flushes the diary (with a configurable `shutdown_diary_timeout_sec`).
2. Stops TTS.
3. Closes the audit recorder.
4. Closes the database.

Registered services are shut down in reverse dependency order to avoid
use-after-free.

## Service Initialisation

Services are initialised directly in `daemon.py` in dependency order.
Each initialisation step is wrapped in `try/except` so that a failure
marks the service as `DEGRADED` or `UNAVAILABLE` rather than aborting
startup.

**Initialisation order (in `daemon.py`):**
1. Health registry
2. Policy engine + approval store
3. Audit recorder (opt-in via `audit_db_path`)
4. Main database + dialogue memory
5. TTS engine
6. MCP tool discovery
7. Location service probe

## Configuration Fields Used

| Field                       | Type    | Default | Effect                                  |
|-----------------------------|---------|---------|-----------------------------------------|
| `shutdown_diary_timeout_sec`| `float` | `5.0`   | Maximum time to wait for diary flush    |

## Health States

| State            | Meaning                                                  |
|------------------|----------------------------------------------------------|
| `INITIALISING`   | Service is starting up                                   |
| `READY`          | Service is fully operational                             |
| `DEGRADED`       | Partial operation; some features may be unavailable      |
| `UNAVAILABLE`    | Service is not available; dependent features are disabled|

## Graceful Degradation Guarantee

Every service initialisation in `daemon.py` catches all exceptions and
marks the affected service as `DEGRADED` or `UNAVAILABLE`. The daemon
will always reach a running state even if optional services (TTS, MCP,
audit, location) fail to initialise.
