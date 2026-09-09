# Local services

Everything listed here appears in **Settings → Experimental → Integrations**. The
panel reads [`registry.json`](registry.json), probes each entry **on demand only**
(there is no polling), and runs the `Start-*` / `Stop-*` pair in this directory in
a visible PowerShell window so failures have somewhere to be read.

Host owner: [`src/services/local-services/LocalServicesManager.ts`](../../src/services/local-services/LocalServicesManager.ts).

## Kinds

`kind` decides how a row is probed and what the panel is allowed to offer.

| Kind       | Probe                                                   | Buttons           |
| ---------- | ------------------------------------------------------- | ----------------- |
| `http`     | `GET healthUrl`, anything under 500 counts as listening | Start / Stop      |
| `docker`   | `docker ps` filtered by `containerName`                 | Start / Stop      |
| `embedded` | the module specifier resolves from the extension host   | Install / Disable |
| `binary`   | run `binaryName versionArgs` and see whether it exits 0 | Install / Disable |
| `remote`   | `GET healthUrl` with a longer timeout                   | none              |

Three of these exist because the honest answer was not "running" or "stopped":

- **`embedded`** is a native module linked into the extension host — LadybugDB.
  There is no port and no process. "Installing" it makes `@ladybugdb/core`
  resolvable; the change lands on the next window reload, because the backend
  resolver runs once at activation.
- **`binary`** is an executable we invoke per call — Joern. Also no port and no
  daemon: the adapter runs it with an argument array and a timeout, and it exits.
  The probe is the same `--version` call the backend resolver makes, so the panel
  and the fabric cannot disagree about whether the plane is usable.
- **`remote`** is a third-party endpoint we depend on but do not run — OSV. It
  gets a status light and no buttons, because a Start button for somebody else's
  service would be a lie about what pressing it does. It is listed at all
  precisely so that "the feed is down" stays distinguishable from "no advisories
  affect this package" — downstream those produce the same empty result.

**A `port` in the registry is a claim that some code connects to it.** Joern was
briefly registered as `http` on a made-up port with a `Start-Joern.ps1` that
launched `joern --server`; nothing in this repository has ever spoken that
protocol, so the light was reporting on a socket nobody dialled — green while the
tool was missing, red while it worked. The port number was contested as well
(8080 is SearXNG on a typical box here _and_ BeeLlama in this registry; 8081 is
BeeLlama in [`start-local-models.ps1`](../start-local-models.ps1)), but the port
being wrong was the smaller half of the bug. Do not give a port to an integration
that is not reached over TCP.

## Adding a service

1. Write `Start-<Name>.ps1` and `Stop-<Name>.ps1` here. Dot-source
   [`lib/LocalService.Common.ps1`](lib/LocalService.Common.ps1) and use
   `Assert-ServiceNotRunning` so a second start reuses a healthy instance rather
   than fighting it for the port.
2. Add the entry to `registry.json`. `startScript` / `stopScript` are reduced to
   bare filenames by the host — a path there will not escape this directory.
3. Set `group` so the row lands under a heading instead of in _Other_.

Both scripts must be idempotent. Start on a healthy service is a no-op; stop on
a stopped service succeeds.

## BLACKGLASS planes

Four rows back the BLACKGLASS security memory fabric. Each has a distinct
failure mode, and the panel's `detail` line reports what the fabric actually did
with the plane — which is **not** the same question as whether the port is open,
because backends are resolved once at activation. A service started afterwards
shows as running and unattached until the window reloads.

| Row                       | Plane                  | Absent means                                                                                  |
| ------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| Qdrant Vector DB          | vector (§17)           | reduced recall — exact anchors and graph traversal still answer                               |
| LadybugDB Security Graph  | graph (§3.2/§30)       | the hypergraph is in-memory and does not survive a reload                                     |
| Joern Code Property Graph | reachability (§10/§11) | reachability is **unknown**, never "not reachable" (§35); no chain reaches `STATIC_CONFIRMED` |
| OSV Advisory Feed         | advisories (§10.1)     | advisory coverage is unknown, not empty                                                       |

Resolution lives in
[`BlackglassBackends.ts`](../../src/kelvin/security/blackglass/BlackglassBackends.ts)
and can be overridden per plane:

```
ZOO_BLACKGLASS_QDRANT_URL         default http://127.0.0.1:6333
ZOO_BLACKGLASS_QDRANT_API_KEY
ZOO_BLACKGLASS_JOERN_BIN          default `joern` on PATH
ZOO_NPU_EMBEDDER_BASE_URL         default http://127.0.0.1:8010
ZOO_BLACKGLASS_DISABLE_QDRANT     1 to force the in-memory vector store
ZOO_BLACKGLASS_DISABLE_LADYBUG    1 to force the in-memory graph
ZOO_BLACKGLASS_DISABLE_JOERN      1 to force UnavailableCodeGraphService
ZOO_BLACKGLASS_DISABLE_EMBEDDINGS 1 to disable dense entry retrieval
```

Attaching a backend does not populate it. Evidence enters the graph only through
the host-owned pipelines the spec permits (§19.4) — the user-facing one is
**`Kelvin Clyne: BLACKGLASS — Sync OSV advisories for this workspace`** from the
command palette. It walks the workspace's dependency manifests, queries OSV, and
writes through `AdvisoryIngestor`. The scan is capped and cancellable, and it
reports what it skipped rather than presenting a truncated scan as a clean one.

Advisories ingested this way are `SOURCE_GROUNDED`: _this advisory exists and
names this package_. They are not a claim that the workspace is vulnerable —
that needs the resolved lockfile version and, for a real claim, reachability from
Joern.
