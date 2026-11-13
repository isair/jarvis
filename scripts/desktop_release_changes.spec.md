# Develop installer build gate

The release workflow builds develop installers when a complete push diff contains
any path outside `tests/`. Changes confined to tests, and empty diffs, skip both
the installer build and develop release publication. Other paths remain eligible,
including source, dependencies, packaging, workflows and documentation.

The classifier reads the push event's before and after commit IDs and examines
the full Git diff with rename detection disabled and null-delimited paths. Moving
a runtime file into tests therefore remains a runtime deletion requiring a build.
It uses no GitHub changed-file list or filename decoding.

Manual dispatch, missing or invalid commit IDs, unavailable history, Git failure
and timeout require a build. Output is appended to `GITHUB_OUTPUT` as
`required=true` or `required=false`; missing output configuration fails the job.

Develop workflow runs have independent concurrency groups. Only qualifying
installer jobs share a cancellation group, so a test-only push cannot cancel a
running installer build. Develop publication requires a successful installer job;
skipped, cancelled and failed builds cannot publish. Main retains serial release
analysis with cancellation and stable build/publication behaviour.
