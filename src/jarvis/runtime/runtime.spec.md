# Runtime Package Specification

## Purpose

Provides the service lifecycle infrastructure for the Jarvis daemon:
health tracking, ordered startup with graceful degradation, coordinated
shutdown, and an optional centralised service container.

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

### `bootstrap.py` — `build_service_container()`

Assembles all Jarvis subsystem instances into a `ServiceContainer` in
dependency order. Each initialisation step is wrapped in `try/except` so
that a failure marks the service as `DEGRADED` or `UNAVAILABLE` rather
than aborting startup.

**Initialisation order:**
1. Audit recorder
2. Policy engine + approval store
3. Main database + dialogue memory
4. TTS engine
5. MCP tool discovery
6. Location service probe
7. Shutdown manager (wires all services together)

Intended for future use as the primary entry point for daemon startup.
Current deployments wire services directly in `daemon.py`.

### `service_container.py` — `ServiceContainer`

Single owner of all live service instances. Provides typed property
accessors (`db`, `tts`, `policy_engine`, `audit_recorder`, …) and a
`stop_event` that the daemon waits on.

Attributes are set by `build_service_container()` during bootstrap.

### `shutdown_manager.py` — `ShutdownManager`

Coordinates orderly shutdown:
1. Flushes the diary (with a configurable `shutdown_diary_timeout_sec`).
2. Stops TTS.
3. Closes the audit recorder.
4. Closes the database.

Registered services are shut down in reverse dependency order to avoid
use-after-free.

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

Every `_init_*` function in `bootstrap.py` catches all exceptions and
marks the affected service as `DEGRADED` or `UNAVAILABLE`. The daemon
will always reach a running state even if optional services (TTS, MCP,
audit, location) fail to initialise.
