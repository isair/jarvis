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

Each queued utterance owns its preprocessed text, duration callback and
completion callback. Later submissions, including submissions made by a
callback, cannot replace or clear another utterance's callbacks. Synthesis
reports its exact duration to that utterance's listener, and completion reports
only that utterance's completion. An interrupted utterance does not report
successful completion. Callback failures remain isolated from later work. The worker releases the
completed request and its callback references before waiting for more speech.

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

Stopping speech interrupts the active utterance and discards queued utterances
and their callbacks. A worker still synthesising after the bounded join retains
its shutdown state and ownership until it exits. Speech submitted while that
worker drains is discarded; no second worker starts beside it. Stopped synthesis
does not begin playback or report completion. Once the worker exits, a new
submission can start a fresh worker without replaying discarded speech.
Completion callbacks may stop their own worker without joining themselves.

Stopping also invalidates submissions and startup requests waiting for their
initial model load, even when no worker exists yet. Finishing that preparation
cannot start a cancelled request or enqueue it onto a newer worker. A fresh
submission after stopping can use the prepared model. Model preparation itself
is not forcibly terminated.
