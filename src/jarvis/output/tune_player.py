from __future__ import annotations
import io
import struct
import threading
import time
import platform
import subprocess
import shutil
import tempfile
from typing import Optional

import numpy as np

from ..debug import debug_log


def _generate_thinking_pad_wav() -> bytes:
    """Generate a long, seamlessly looping warm ambient pad as WAV bytes.

    Designed to run indefinitely while Jarvis thinks. Two tricks make the
    looping imperceptible:

    1. Mathematical seam: every sine frequency (in Hz) is an integer, and
       every LFO period evenly divides the clip duration (in seconds), so
       start and end samples match exactly — no click at the wrap point.
    2. Long duration (60s): subprocess-relaunch gaps on some platforms
       can add a noticeable pause between plays. Making the clip long
       means the relaunch happens at most once per minute, which is rare
       enough to stay invisible for normal thinking sessions.

    Tone character — choir-"ahh" / bowed-string pad:
    - A major triad (A3 / C#4 / E4) with a natural harmonic spectrum
      (partials 1-6 with 1/n-style rolloff) so each voice has real
      timbre instead of sounding like a pure sine.
    - Three-way unison detune per chord tone (-1 Hz, 0, +1 Hz) —
      mirrors how an ensemble of human singers or strings is never
      perfectly in tune, giving chorus-like warmth and body.
    - Phase-offset cross-fading LFOs on the three chord voices so the
      blend shifts continuously while total amplitude stays steady.
    """
    sample_rate = 44100
    duration_s = 60  # seconds — rarely relaunches, stays seamless at seam

    # A major triad one octave below middle A. Integer Hz → integer
    # cycles in any whole-second duration, guaranteeing a seam-free loop.
    chord_roots = (220, 275, 330)  # A3, ~C#4, ~E4

    # Harmonic spectrum — softly rolled-off partials give a warm
    # bowed/choir timbre instead of a pure-sine "synth" character.
    # Weights are gently below 1/n so high partials don't buzz.
    harmonic_gains = (1.00, 0.55, 0.30, 0.18, 0.10, 0.06)
    # Per-partial starting phase — small offsets avoid an overly
    # synchronised attack and sound more like an ensemble.
    harmonic_phases = (0.0, 0.7, 1.3, 2.1, 0.4, 1.9)

    # Unison detune offsets (Hz). At duration=60s each offset is
    # still an integer Hz step, so all cycles close seamlessly.
    unison_offsets = (-1, 0, 1)

    lfo_period = 6.0   # amplitude cross-fade: 10 cycles per 60s loop

    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    two_pi = 2 * np.pi

    # Cross-fading amplitudes, phase-offset 120° so the sum is roughly
    # constant — the texture never drops to silence.
    lfo_w = two_pi / lfo_period
    voice_amps = [
        0.55 + 0.25 * np.sin(lfo_w * t + k * two_pi / 3)
        for k in range(3)
    ]

    # Build each chord voice as a sum of harmonic partials, summed
    # across unison-detuned copies. All partial frequencies are
    # integer Hz × duration → integer cycles → seam is inaudible.
    # Motion is provided by the unison beating between detuned
    # copies and the cross-fading LFO; no explicit vibrato is needed.
    voice_signals = []
    for root in chord_roots:
        unison_sum = np.zeros(n, dtype=np.float64)
        for offset in unison_offsets:
            f = root + offset
            for h_idx, (gain, phase0) in enumerate(
                zip(harmonic_gains, harmonic_phases)
            ):
                partial_freq = (h_idx + 1) * f
                phase = two_pi * partial_freq * t + phase0
                unison_sum += gain * np.sin(phase)
        voice_signals.append(unison_sum)

    tone = sum(a * v for a, v in zip(voice_amps, voice_signals))

    # Normalise to roughly [-1, 1] based on observed peak, then set
    # a gentle overall level.
    peak = float(np.max(np.abs(tone))) or 1.0
    signal = (tone / peak) * 0.32

    samples = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    num_samples = samples.size

    # Build WAV file in memory
    wav_buffer = io.BytesIO()

    # WAV header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    # RIFF header
    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', 36 + data_size))
    wav_buffer.write(b'WAVE')

    # fmt chunk
    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))  # chunk size
    wav_buffer.write(struct.pack('<H', 1))   # PCM format
    wav_buffer.write(struct.pack('<H', num_channels))
    wav_buffer.write(struct.pack('<I', sample_rate))
    wav_buffer.write(struct.pack('<I', byte_rate))
    wav_buffer.write(struct.pack('<H', block_align))
    wav_buffer.write(struct.pack('<H', bits_per_sample))

    # data chunk
    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', data_size))
    wav_buffer.write(samples.tobytes())

    return wav_buffer.getvalue()


# Cache the generated WAV data
_THINKING_PAD_WAV: Optional[bytes] = None


def _get_thinking_pad_wav() -> bytes:
    """Get cached thinking-pad WAV data, generating on first call."""
    global _THINKING_PAD_WAV
    if _THINKING_PAD_WAV is None:
        _THINKING_PAD_WAV = _generate_thinking_pad_wav()
    return _THINKING_PAD_WAV


class TunePlayer:
    """Plays a simple tune while processing is happening."""
    
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def start_tune(self) -> None:
        """Start playing the processing tune."""
        if not self.enabled or self._thread is not None:
            return

        debug_log("thinking tune: start", category="tune")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_tune, daemon=True)
        self._thread.start()

    def stop_tune(self) -> None:
        """Stop the processing tune immediately.

        Terminates any in-flight player subprocess so we don't wait out
        the remainder of a long clip after thinking has finished.
        """
        if self._thread is None:
            return

        debug_log("thinking tune: stop", category="tune")
        self._stop_event.set()

        proc = self._current_process
        if proc is not None:
            try:
                proc.terminate()
            except Exception as exc:
                debug_log(f"thinking tune: terminate failed: {exc}", category="tune")

        if platform.system().lower() == "windows":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as exc:
                debug_log(f"thinking tune: winsound purge failed: {exc}", category="tune")

        self._thread.join(timeout=2.0)
        self._thread = None
        self._current_process = None
        self._is_playing.clear()
        
    def is_playing(self) -> bool:
        """Check if tune is currently playing."""
        return self._is_playing.is_set()
        
    def _play_tune(self) -> None:
        """Play a gentle processing tune in a loop."""
        self._is_playing.set()

        system = platform.system().lower()

        try:
            if system == "darwin":
                self._play_macos_tune()
            elif system == "linux":
                self._play_linux_tune()
            elif system == "windows":
                self._play_windows_tune()
            else:
                debug_log(f"thinking tune: unknown platform {system!r}; falling back", category="tune")
                self._play_fallback_tune()
        except Exception as exc:
            debug_log(f"thinking tune: platform player failed ({exc!r}); falling back", category="tune")
            self._play_fallback_tune()
        finally:
            self._is_playing.clear()
            
    def _run_and_wait(self, cmd: list) -> None:
        """Launch a player subprocess and wait for it, interruptible by stop.

        Tracks the Popen handle on self so stop_tune can terminate it
        immediately instead of waiting out the clip.
        """
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_process = proc
        try:
            while proc.poll() is None:
                if self._stop_event.wait(0.05):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return
        finally:
            self._current_process = None

    def _play_macos_tune(self) -> None:
        """Play the thinking pad on macOS via afplay, looping until stop."""
        import os

        afplay = shutil.which("afplay")
        if not afplay:
            debug_log("thinking tune: afplay not found; falling back", category="tune")
            self._play_fallback_tune()
            return

        wav_data = _get_thinking_pad_wav()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name

            while not self._stop_event.is_set():
                try:
                    self._run_and_wait([afplay, tmp_path])
                except Exception as exc:
                    debug_log(f"thinking tune: afplay run failed: {exc!r}", category="tune")
                    break
                if self._stop_event.is_set():
                    break
                time.sleep(0.05)
        except Exception as exc:
            debug_log(f"thinking tune: macOS playback setup failed: {exc!r}", category="tune")
            self._play_fallback_tune()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception as exc:
                    debug_log(f"thinking tune: temp cleanup failed: {exc!r}", category="tune")
                            
    def _play_linux_tune(self) -> None:
        """Play the thinking pad on Linux via paplay/aplay/ffplay, looping until stop."""
        import os

        paplay = shutil.which("paplay")  # PulseAudio
        aplay = shutil.which("aplay")    # ALSA
        ffplay = shutil.which("ffplay")  # FFmpeg

        player = paplay or aplay or ffplay
        if not player:
            debug_log("thinking tune: no Linux audio player found; falling back", category="tune")
            self._play_fallback_tune()
            return

        wav_data = _get_thinking_pad_wav()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name

            if player == ffplay:
                cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            else:
                cmd = [player, tmp_path]

            while not self._stop_event.is_set():
                try:
                    self._run_and_wait(cmd)
                except Exception as exc:
                    debug_log(f"thinking tune: linux player run failed: {exc!r}", category="tune")
                    break
                if self._stop_event.is_set():
                    break
                time.sleep(0.05)
        except Exception as exc:
            debug_log(f"thinking tune: linux playback setup failed: {exc!r}", category="tune")
            self._play_fallback_tune()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception as exc:
                    debug_log(f"thinking tune: temp cleanup failed: {exc!r}", category="tune")
            
    def _play_windows_tune(self) -> None:
        """Play tune on Windows using winsound SND_LOOP for seamless playback.

        SND_LOOP + SND_ASYNC tells the OS to loop the buffer natively so
        there's no per-iteration gap at all, and stop_tune calls
        PlaySound(None) to halt it instantly.
        """
        try:
            import winsound
        except ImportError:
            debug_log("thinking tune: winsound unavailable; falling back", category="tune")
            self._play_fallback_tune()
            return

        try:
            wav_data = _get_thinking_pad_wav()
            flags = (
                winsound.SND_MEMORY
                | winsound.SND_ASYNC
                | winsound.SND_LOOP
                | winsound.SND_NODEFAULT
            )
            winsound.PlaySound(wav_data, flags)
            # Block this worker thread until stop is requested; the OS
            # handles looping, so we just wait.
            self._stop_event.wait()
        except Exception as exc:
            debug_log(f"thinking tune: winsound playback failed: {exc!r}", category="tune")
            self._play_fallback_tune()
        finally:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as exc:
                debug_log(f"thinking tune: winsound stop failed: {exc!r}", category="tune")
            
    def _play_fallback_tune(self) -> None:
        """Fallback tune using print statements (silent but indicates activity)."""
        # Very subtle fallback - just a brief visual indicator
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
        # Clear the spinner
        try:
            print("\r" + " " * 30 + "\r", end="", flush=True)
        except Exception:
            pass
