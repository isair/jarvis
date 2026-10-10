# Thinking-sound playback specification

The thinking sound is synthesised locally as a cached mono int16 PCM buffer.
Playback loops that buffer through sounddevice, preserving every frame across
the seam. The fallback displays a local processing indicator when audio output
is unavailable. Neither path uses speech text or sends data to another service.

## Worker lifecycle

A player owns at most one worker. A disabled player does not start one; repeated
start requests while its worker is active or closing are ignored. A lifecycle
lock serialises worker publication, stop signalling and retirement.

Stop sets the worker's stop event and waits at most one second. A slow teardown
retains ownership until the worker finishes; a restart request cannot clear its
stop event or start another stream during that interval. The worker retires
itself and clears the playing indicator in its final cleanup, including failures
before playback. Subsequent starts can create a fresh worker without another
stop call.

The owning worker opens, starts and closes its audio stream. These device
operations use the shared PortAudio lock. The caller does not abort or close
the stream across threads, and does not hold the lifecycle lock while joining.
Start, stop, deferred restart and worker completion have debug diagnostics.
