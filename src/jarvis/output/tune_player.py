from __future__ import annotations
import io
import struct
import threading
import time
from typing import Optional

import numpy as np

from ..debug import debug_log


def _generate_thinking_pad_samples() -> tuple[np.ndarray, int]:
    """Generate the thinking pad as a raw int16 mono buffer.

    Designed to run indefinitely while Jarvis thinks. Two tricks make
    the looping imperceptible:

    1. Mathematical seam: every sine frequency (in Hz) is an integer,
       so start and end samples match exactly — no click at the wrap
       point.
    2. Short duration (10s): the sounddevice callback loops the
       buffer natively in the OS audio thread, so there's no
       per-iteration gap. A shorter buffer keeps generation cheap
       (~70ms) and memory small.

    Tone character — choir-"ahh" / bowed-string pad:
    - A major triad (A3 / C#4 / E4) with a natural harmonic spectrum
      (fundamental only) so each voice has real
      timbre instead of sounding like a pure sine.
    - Three-way unison detune per chord tone (-1 Hz, 0, +1 Hz) —
      mirrors how an ensemble of human singers or strings is never
      perfectly in tune, giving chorus-like warmth and body and a
      gentle ~1 Hz beat between the outer layers.

    Returns (int16 mono samples, sample_rate).
    """
    sample_rate = 44100
    # 10s buffer = 5 pulse cycles of 2s each (1s tone + 1s silence).
    duration_s = 10
    pulse_cycle_s = 2.0
    tone_s = 1.0  # audible portion per cycle
    attack_s = 0.008  # ~8ms fast attack gives the slight "click"

    chord_roots = (220, 275, 330)  # A3, ~C#4, ~E4 — integer Hz for seamless seam
    unison_offsets = (-1, 0, 1)

    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    two_pi = 2 * np.pi

    # Single-cycle envelope: fast linear attack → exponential decay →
    # silence for the rest of the cycle. Tiles across the whole buffer.
    cycle_len = int(sample_rate * pulse_cycle_s)
    tone_len = int(sample_rate * tone_s)
    attack_len = max(1, int(sample_rate * attack_s))
    decay_len = tone_len - attack_len
    one_cycle = np.zeros(cycle_len, dtype=np.float64)
    one_cycle[:attack_len] = np.linspace(0.0, 1.0, attack_len, endpoint=True)
    # Exponential decay from 1.0 down to effectively 0 over the tone body.
    decay = np.exp(-4.0 * np.arange(decay_len) / decay_len)
    one_cycle[attack_len:tone_len] = decay
    # Tile three cycles across the 9s buffer (matches duration_s exactly).
    num_cycles = n // cycle_len
    envelope = np.zeros(n, dtype=np.float64)
    for i in range(num_cycles):
        envelope[i * cycle_len:(i + 1) * cycle_len] = one_cycle

    # Build the triad once: three pure sines per chord tone with ±1 Hz
    # unison detune for the characteristic beat.
    tone = np.zeros(n, dtype=np.float64)
    for root in chord_roots:
        for offset in unison_offsets:
            f = root + offset
            tone += np.sin(two_pi * f * t)
    peak = float(np.max(np.abs(tone))) or 1.0
    tone = tone / peak

    signal = tone * envelope * 0.38

    samples = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    return samples, sample_rate


def _generate_thinking_pad_wav() -> bytes:
    """WAV-wrapped version of the thinking pad (kept for test coverage)."""
    samples, sample_rate = _generate_thinking_pad_samples()
    num_samples = samples.size

    wav_buffer = io.BytesIO()
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', 36 + data_size))
    wav_buffer.write(b'WAVE')

    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))
    wav_buffer.write(struct.pack('<H', 1))
    wav_buffer.write(struct.pack('<H', num_channels))
    wav_buffer.write(struct.pack('<I', sample_rate))
    wav_buffer.write(struct.pack('<I', byte_rate))
    wav_buffer.write(struct.pack('<H', block_align))
    wav_buffer.write(struct.pack('<H', bits_per_sample))

    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', data_size))
    wav_buffer.write(samples.tobytes())

    return wav_buffer.getvalue()


_THINKING_PAD_WAV: Optional[bytes] = None
_THINKING_PAD_SAMPLES: Optional[tuple[np.ndarray, int]] = None


def _get_thinking_pad_wav() -> bytes:
    """Get cached thinking-pad WAV data, generating on first call."""
    global _THINKING_PAD_WAV
    if _THINKING_PAD_WAV is None:
        _THINKING_PAD_WAV = _generate_thinking_pad_wav()
    return _THINKING_PAD_WAV


def _get_thinking_pad_samples() -> tuple[np.ndarray, int]:
    """Get cached raw int16 samples for sounddevice playback."""
    global _THINKING_PAD_SAMPLES
    if _THINKING_PAD_SAMPLES is None:
        _THINKING_PAD_SAMPLES = _generate_thinking_pad_samples()
    return _THINKING_PAD_SAMPLES


def _prewarm_cache() -> None:
    """Pre-generate samples off the hot path so the first start_tune()
    doesn't compete with the first LLM call for CPU."""
    try:
        _get_thinking_pad_samples()
    except Exception as exc:
        debug_log(f"thinking tune: prewarm failed: {exc!r}", category="tune")


threading.Thread(target=_prewarm_cache, daemon=True).start()


class TunePlayer:
    """Plays a thinking-pad tune in a loop while Jarvis is processing.

    Uses sounddevice (PortAudio) for playback, which is the same API TTS
    uses. This matters: if the tune held the audio output device via a
    separate path (e.g. afplay subprocess killed mid-stream), macOS
    CoreAudio could take seconds to release the device, stalling TTS.
    Using one API means clean release — stop returns in milliseconds and
    TTS can open the device immediately after.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = threading.Event()
        self._stream_lock = threading.Lock()
        self._stream = None  # sounddevice.OutputStream, set while playing

    def start_tune(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        debug_log("thinking tune: start", category="tune")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_tune, daemon=True)
        self._thread.start()

    def stop_tune(self) -> None:
        """Stop the tune immediately, releasing the audio device."""
        if self._thread is None:
            return

        debug_log("thinking tune: stop", category="tune")
        self._stop_event.set()

        with self._stream_lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception as exc:
                debug_log(f"thinking tune: stream abort failed: {exc!r}", category="tune")

        self._thread.join(timeout=1.0)
        self._thread = None
        self._is_playing.clear()

    def is_playing(self) -> bool:
        return self._is_playing.is_set()

    def _play_tune(self) -> None:
        self._is_playing.set()
        try:
            try:
                import sounddevice as sd
            except Exception as exc:
                debug_log(f"thinking tune: sounddevice unavailable: {exc!r}", category="tune")
                self._play_fallback_tune()
                return

            try:
                samples, sample_rate = _get_thinking_pad_samples()
            except Exception as exc:
                debug_log(f"thinking tune: sample generation failed: {exc!r}", category="tune")
                self._play_fallback_tune()
                return

            position = [0]  # list so the callback closure can mutate it
            total = samples.size

            def callback(outdata, frames, time_info, status):
                # No I/O here — this runs in the realtime audio thread.
                start = position[0]
                end = start + frames
                if end <= total:
                    outdata[:, 0] = samples[start:end]
                    position[0] = end % total
                else:
                    # Wrap around the seamless seam.
                    first = total - start
                    outdata[:first, 0] = samples[start:total]
                    remainder = frames - first
                    outdata[first:, 0] = samples[:remainder]
                    position[0] = remainder

            try:
                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    # Large block + high latency: fewer callbacks, fewer
                    # GIL acquisitions, lighter touch on the rest of the
                    # app. 8192 frames ≈ 186ms per wakeup vs 23ms before.
                    blocksize=8192,
                    latency='high',
                    callback=callback,
                )
            except Exception as exc:
                debug_log(f"thinking tune: stream open failed: {exc!r}", category="tune")
                self._play_fallback_tune()
                return

            try:
                with self._stream_lock:
                    self._stream = stream
                stream.start()
                # Hand off to the OS audio thread. Wake when stop is
                # requested — no polling loop, no per-iteration gap.
                self._stop_event.wait()
            except Exception as exc:
                debug_log(f"thinking tune: stream playback failed: {exc!r}", category="tune")
            finally:
                try:
                    stream.close()
                except Exception as exc:
                    debug_log(f"thinking tune: stream close failed: {exc!r}", category="tune")
                with self._stream_lock:
                    self._stream = None
        finally:
            self._is_playing.clear()

    def _play_fallback_tune(self) -> None:
        """Fallback for environments without a usable audio output."""
        patterns = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not self._stop_event.is_set():
            try:
                print(f"\r[jarvis] {patterns[i % len(patterns)]} processing...",
                      end="", flush=True)
                time.sleep(0.2)
                i += 1
            except Exception:
                break
        try:
            print("\r" + " " * 30 + "\r", end="", flush=True)
        except Exception:
            pass
