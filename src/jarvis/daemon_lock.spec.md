# Daemon ownership

`daemon.main()` holds one per-user OS file lock for initialisation, execution
and cleanup. Independent handles contend both within a process and between
processes. Unix uses non-blocking `flock`; Windows locks one byte with
`msvcrt.locking`. Refused entry returns before resetting stop flags or
changing runtime globals. Smoke initialisation follows the same ownership
rule and raises on contention so it cannot report successful initialisation.

The default file is `jarvis_daemon.lock` under macOS Application Support/Jarvis,
Windows LOCALAPPDATA/Jarvis or Unix ~/.jarvis. `JARVIS_DAEMON_LOCK` overrides
the path. Parent directories are created on acquisition. The owner writes a
fixed-width diagnostic PID field before the Windows lock byte. PID contents
are advisory; lock acquisition determines ownership.

Contention returns no handle. Other filesystem and lock errors propagate.
Failure while recording ownership closes the handle. Normal release,
exceptions from the runtime and process death release the OS lock. The file
is retained and never unlinked during release, preserving a common inode for
all contenders.
