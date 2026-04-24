from __future__ import annotations
import io
import math
import struct
import threading
import time
import platform
import subprocess
import shutil
import tempfile
from typing import Optional


def _generate_sonar_ping_wav() -> bytes:
    """Generate a seamlessly looping ambient pad as WAV bytes.

    Designed to run indefinitely while Jarvis thinks — the clip has no
    silent endpoints, so the loop point is inaudible and the texture
    reads as one unbroken flowing sound rather than repeated swells.

    Seamlessness is guaranteed mathematically: every sine in the mix uses
    a frequency that completes an integer number of cycles across the
    clip duration, and every LFO (amplitude cross-fades, vibrato) has a
    period that divides the duration evenly. Start and end samples are
    therefore identical, so the loop wraps with zero click.

    - Three voices tuned to a D major-ninth-ish chord (D5, F#5, A5)
    - Slow cross-fading LFOs give the illusion of melodic motion
    - Gentle detune layer adds chorus-like warmth
    - Subtle vibrato for organic feel
    - Amplitude never drops to zero — texture is continuous
    """
    sample_rate = 44100
    duration = 4.0  # seamless loop length

    # Frequencies chosen so f * duration is an integer → perfect loop seam.
    # duration = 4s, so any integer Hz works; these form a consonant chord.
    f1 = 588.0   # ~D5
    f2 = 740.0   # ~F#5
    f3 = 880.0   # A5
    f1_detune = 589.0  # slight chorus layer against f1

    num_samples = int(sample_rate * duration)
    samples = []

    two_pi = 2 * math.pi
    lfo_w = two_pi / duration  # fundamental LFO, one cycle per loop
    vib_w = two_pi * 2 / duration  # 0.5 Hz vibrato, two cycles per loop

    for i in range(num_samples):
        t = i / sample_rate

        # Cross-fading amplitudes — always sum to a steady continuous level.
        # Each voice breathes in while another breathes out, so the overall
        # texture never drops to silence.
        a1 = 0.55 + 0.25 * math.sin(lfo_w * t)
        a2 = 0.55 + 0.25 * math.sin(lfo_w * t + two_pi / 3)
        a3 = 0.55 + 0.25 * math.sin(lfo_w * t + 2 * two_pi / 3)

        # Gentle vibrato, periodic across the loop (±0.15%).
        vibrato = 1 + 0.0015 * math.sin(vib_w * t)

        v1 = math.sin(two_pi * f1 * vibrato * t)
        v2 = math.sin(two_pi * f2 * vibrato * t + 0.9)
        v3 = math.sin(two_pi * f3 * vibrato * t + 1.7)
        vd = math.sin(two_pi * f1_detune * t + 0.3)

        tone = a1 * v1 + a2 * v2 + a3 * v3 + 0.18 * vd

        # Normalise roughly to [-1, 1]; the cross-fade keeps sum bounded.
        sample = tone / 2.2

        sample_int = int(sample * 32767 * 0.32)
        samples.append(max(-32768, min(32767, sample_int)))

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
    for sample in samples:
        wav_buffer.write(struct.pack('<h', sample))

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
                        timeout=30.0
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
                        timeout=30.0
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
