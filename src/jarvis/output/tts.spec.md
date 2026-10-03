# Speech output specification

Speech synthesis runs locally. Piper uses a local ONNX voice and its JSON
configuration; Chatterbox uses locally loaded model weights. Neither backend
sends user text or audio to a cloud synthesis service.

## Piper voice downloads

Missing voice files are downloaded from the configured public model-file source.
Existing model and configuration files are retained. Each missing file is written
to a temporary sibling and published only after the response stream completes.

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
