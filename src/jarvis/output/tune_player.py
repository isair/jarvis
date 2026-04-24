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


def _generate_sonar_ping_wav() -> bytes:
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

    Tone character:
    - Warm low-mid chord: A3, C#4, E4 (A major triad, one octave down
      from the earlier version so it's no longer high-pitched)
    - Three voices with phase-offset cross-fading LFOs so the blend
      shifts continuously but amplitude never drops to silence
    - Gentle chorus detune layer for warmth
    - Subtle vibrato for organic motion
    """
    sample_rate = 44100
    duration_s = 60  # seconds — rarely relaunches, stays seamless at seam

    # Voices: A major triad one octave below the earlier D-chord.
    # Integer Hz → integer cycles over any whole-second duration.
    f1 = 220   # A3
    f2 = 275   # ~C#4 (within 1.3 cents of 277.18, imperceptible)
    f3 = 330   # ~E4  (within 2 cents of 329.63, imperceptible)
    f1_det = 221  # chorus layer, ~8 cents above f1

    # LFO periods that divide duration evenly → they wrap seamlessly too.
    lfo_period = 6.0   # amplitude cross-fade: 10 cycles per loop
    vib_period = 4.0   # vibrato: 15 cycles per loop

    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    two_pi = 2 * np.pi

    # Cross-fading amplitudes, phase-offset by 120° so the sum stays
    # roughly constant — the texture never drops to silence.
    lfo_w = two_pi / lfo_period
    a1 = 0.55 + 0.25 * np.sin(lfo_w * t)
    a2 = 0.55 + 0.25 * np.sin(lfo_w * t + two_pi / 3)
    a3 = 0.55 + 0.25 * np.sin(lfo_w * t + 2 * two_pi / 3)

    # Gentle vibrato (±0.15% pitch modulation).
    vibrato = 1 + 0.0015 * np.sin(two_pi * t / vib_period)

    v1 = np.sin(two_pi * f1 * vibrato * t)
    v2 = np.sin(two_pi * f2 * vibrato * t + 0.9)
    v3 = np.sin(two_pi * f3 * vibrato * t + 1.7)
    vd = np.sin(two_pi * f1_det * t + 0.3)

    tone = a1 * v1 + a2 * v2 + a3 * v3 + 0.18 * vd
    signal = (tone / 2.2) * 0.32  # normalise and set overall level

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
_SONAR_PING_WAV: Optional[bytes] = None


def _get_sonar_ping_wav() -> bytes:
    """Get cached sonar ping WAV data, generating if needed."""
    global _SONAR_PING_WAV
    if _SONAR_PING_WAV is None:
        _SONAR_PING_WAV = _generate_sonar_ping_wav()
    return _SONAR_PING_WAV


class TunePlayer:
    """Plays a simple tune while processing is happening."""
    
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = threading.Event()
        
    def start_tune(self) -> None:
        """Start playing the processing tune."""
        if not self.enabled or self._thread is not None:
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_tune, daemon=True)
        self._thread.start()
        
    def stop_tune(self) -> None:
        """Stop the processing tune."""
        if self._thread is None:
            return
            
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None
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
        except Exception:
            # If we can't play system sounds, fall back to simple beeps
            self._play_fallback_tune()
        finally:
            self._is_playing.clear()
            
    def _play_macos_tune(self) -> None:
        """Play tune on macOS using afplay with generated pop sound."""
        import os

        afplay = shutil.which("afplay")
        if not afplay:
            self._play_fallback_tune()
            return

        # Write WAV to temp file
        wav_data = _get_sonar_ping_wav()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name

            while not self._stop_event.is_set():
                try:
                    subprocess.run(
                        [afplay, tmp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120.0
                    )
                    time.sleep(0.05)  # Minimal gap — breaths flow into each other
                except Exception:
                    break
        except Exception:
            self._play_fallback_tune()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                            
    def _play_linux_tune(self) -> None:
        """Play tune on Linux using paplay, aplay, or ffplay with generated sonar ping."""
        import os

        # Find a suitable audio player
        paplay = shutil.which("paplay")  # PulseAudio
        aplay = shutil.which("aplay")    # ALSA
        ffplay = shutil.which("ffplay")  # FFmpeg

        player = paplay or aplay or ffplay
        if not player:
            self._play_fallback_tune()
            return

        # Write WAV to temp file (these players need a file)
        wav_data = _get_sonar_ping_wav()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_file.write(wav_data)
                tmp_path = tmp_file.name

            # Build command based on player
            if player == ffplay:
                cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            else:
                cmd = [player, tmp_path]

            while not self._stop_event.is_set():
                try:
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120.0
                    )
                    time.sleep(0.05)  # Minimal gap — breaths flow into each other like macOS
                except Exception:
                    break
        except Exception:
            self._play_fallback_tune()
        finally:
            # Clean up temp file
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            
    def _play_windows_tune(self) -> None:
        """Play tune on Windows using winsound with generated sonar ping."""
        try:
            import winsound
            wav_data = _get_sonar_ping_wav()
            while not self._stop_event.is_set():
                try:
                    # Play the sonar ping from memory
                    winsound.PlaySound(
                        wav_data,
                        winsound.SND_MEMORY | winsound.SND_NODEFAULT
                    )
                    time.sleep(0.05)  # Minimal gap — breaths flow into each other like macOS
                except Exception:
                    break
        except ImportError:
            # winsound not available, fallback to visual indicator
            self._play_fallback_tune()
            
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
