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
    """Generate a pleasant pop sound as WAV bytes.

    Creates a sound similar to macOS Pop.aiff:
    - Clean sine tone
    - Quick decay
    - Subtle tremolo flutter for "shake" character
    """
    sample_rate = 44100
    duration = 0.12  # 120ms - short and clean

    freq = 520  # C5 area - clean mid-range tone

    # Generate samples
    num_samples = int(sample_rate * duration)
    samples = []

    for i in range(num_samples):
        t = i / sample_rate

        # Smooth envelope - quick attack, clean decay
        attack = 1 - math.exp(-t * 600)
        decay = math.exp(-t * 22)
        envelope = attack * decay

        # Tremolo flutter - fast amplitude wobble that fades
        tremolo_rate = 55  # Hz
        tremolo_depth = 0.25 * math.exp(-t * 30)  # Fades quickly
        tremolo = 1 + tremolo_depth * math.sin(2 * math.pi * tremolo_rate * t)

        # Clean sine tone
        sample = envelope * tremolo * math.sin(2 * math.pi * freq * t)

        # Convert to 16-bit PCM
        sample_int = int(sample * 32767 * 0.7)
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
                        timeout=2.0
                    )
                    time.sleep(0.8)  # Gentle spacing
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
                        timeout=2.0
                    )
                    time.sleep(0.8)  # Gentle spacing like macOS
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
                    time.sleep(0.8)  # Gentle spacing like macOS
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
