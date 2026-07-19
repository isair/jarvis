"""
Process-wide serialisation of PortAudio stream lifecycle calls.

PortAudio explicitly documents opening and closing streams as NOT thread
safe (https://github.com/PortAudio/portaudio/wiki/Tips_Threading), and its
Windows host APIs abort the whole process on internal assertion failures.
Jarvis touches PortAudio from several independent threads — the voice
listener's run loop, the Windows mic-permission check thread, the dictation
engine's worker threads, the TTS playback thread, and the thinking-tune
player. Unserialised open/start/stop/close/abort calls across those threads
are the root cause of the Windows "Fatal Python error: Aborted" crash
family (#462, #401, #422).

Every stream lifecycle call (constructing an ``InputStream``/``OutputStream``
and calling ``start``/``stop``/``close``/``abort`` on it, plus blocking
``sd.play`` which opens a stream internally) must run inside
``portaudio_lock``. Do NOT hold the lock across playback waits or other
blocking work, and never acquire another lock while holding it — take it
innermost, around the direct sounddevice call only.

Accepted exceptions:
- The dictation beep holds the lock across its blocking ``sd.play`` —
  sub-200ms, and splitting play/wait would leave the internal stream's
  teardown unguarded.
- The Windows mic-permission probe opens its stream WITHOUT the lock: that
  open can hang indefinitely when Windows blocks mic access, and hanging
  while holding this lock would freeze every audio user in the process.
"""

import threading

# Re-entrant so a guarded close inside a guarded open-fallback path is safe.
portaudio_lock = threading.RLock()
