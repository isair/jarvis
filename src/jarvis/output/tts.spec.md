# Speech output specification

Speech synthesis runs locally. Piper uses a local ONNX voice and its JSON
configuration; Chatterbox uses locally loaded model weights. Neither backend
sends user text or audio to a cloud synthesis service.

## Startup availability

Both speech backends load their local model before accepting speech work. A
model or dependency failure reports its reason and disables speech output for
that engine instance. No worker or queued speech is started for an unavailable
backend. The daemon reports availability after startup, and continues initialising
the listener and text chat when speech output is unavailable. Saved settings are
unchanged; restarting Jarvis creates a fresh engine that can load a repaired model.

## Queued utterance ownership

Each speech submission returns a cancellable `SpeechRequest` that owns its
preprocessed text, duration callback, success callback and terminal callback.
The terminal callback receives exactly one outcome: `completed`, `cancelled`,
`interrupted`, `failed` or `rejected`. Disabled or empty submissions release
ownership as rejected; synthesis and playback errors do not report success.
Cancelling pending work prevents its playback and releases it immediately.
Cancelling active work interrupts it and releases ownership after the worker
finishes that request. Terminal callback failures cannot prevent later work.
Success callbacks run before terminal release only for completed playback.
Piper requires a normal final-buffer drain, rather than stream inactivity alone,
and treats unexpected native callback failures as failed playback. Created
output streams are closed before terminal release, including failed startup.
Interruption
and cancellation are resolved atomically with the terminal outcome.
Both local backends share the same admission and queue ownership lifecycle.
Later submissions, including submissions made by a callback, cannot replace or clear another utterance's callbacks. Synthesis
reports its exact duration to that utterance's listener, and completion reports
only that utterance's completion. An interrupted utterance does not report
successful completion. Callback failures remain isolated from later work. The
worker releases the completed request and its callback references before waiting for more speech.

## Playback observers and activity

Both backends publish playback start after synthesis and device preparation,
immediately before starting audio. Queue admission and synthesis do not publish
speaking activity or replace the last played text. Observers receive the
preprocessed utterance and exact audio duration. The observer pair is selected
at playback start and retained until that request terminates. A queued request
cancelled before playback emits no playback events.

The listener tracks every played utterance, including announcements and error
feedback, through these shared observers. Matching termination records echo-tail
timing; stale termination cannot finish a newer utterance's echo window. Only a
successful conversational reply can schedule follow-up listening.

Playback temporarily owns speaking activity and restores idle only while its
publication remains current. Dictation recording and processing remain protected.
Newer activity, including a same-valued publication, supersedes older ownership.
In-process transitions share the state manager's lock; publication identity also
preserves already observed changes from other publishers. The state file remains
an atomic observer snapshot, rather than a cross-process transaction protocol.
Callback exceptions, including observer task cancellation, cannot skip terminal
cleanup or abandon later queued speech. Cancelling speech uses its request handle.

## Piper voice downloads

Missing voice files are downloaded from the configured public model-file source.
Existing model and configuration files are retained. Each missing file is written
to a unique temporary sibling and atomically published only after the response
stream completes. Concurrent downloads own separate partial files; failure cleanup
removes only the failing invocation's unpublished file.

Connection failures, connect/read timeouts, interrupted response bodies and HTTP
429 responses share a bounded retry budget per file. Backoff increases between
attempts. Each retry starts the incomplete file from the beginning; completed
files are retained if downloading the other file fails. Responses are closed on
success and failure. Certificate/TLS errors and other HTTP failures are not
retried. Exhausted retries report failure and remove unpublished temporary files.
Locked temporary files are reported without crashing startup or publishing them.

Progress includes transferred bytes and transfer rate, with a percentage when
the response supplies a total size. Completion is reported only after publication.
Download diagnostics use `debug_log` without recording user speech or reply text.

## Shutdown and restart

Stopping speech interrupts the active utterance and cancels queued utterances.
Queued requests release their terminal ownership without running their success
callbacks. Callback references are released after termination; reentrant user
callbacks run outside the worker lifecycle lock. A worker still synthesising
after the bounded join retains its shutdown state and ownership until it exits. Speech submitted while that
worker drains is discarded; no second worker starts beside it. Stopped synthesis
does not begin playback or report completion. Once the worker exits, a new
submission can start a fresh worker without replaying discarded speech.
Completion callbacks may stop their own worker without joining themselves.

Stopping also invalidates submissions and startup requests waiting for their
initial model load, even when no worker exists yet. Finishing that preparation
cannot start a cancelled request or enqueue it onto a newer worker. A fresh
submission after stopping can use the prepared model. Model preparation itself
is not forcibly terminated.
