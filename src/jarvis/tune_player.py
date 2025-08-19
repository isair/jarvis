from __future__ import annotations
import threading
import time
import platform
import subprocess
import shutil
from typing import Optional


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
        """Play tune on macOS using afplay or say with tones."""
        # Try to use afplay with system sounds first
        afplay = shutil.which("afplay")
        if afplay:
            # Use a subtle system sound that loops
            sounds = [
                "/System/Library/Sounds/Pop.aiff"
            ]
            
            while not self._stop_event.is_set():
                for sound in sounds:
                    if self._stop_event.is_set():
                        break
                    try:
                        # Play sound quietly in background
                        subprocess.run([afplay, sound], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL,
                                     timeout=2.0)
                        time.sleep(0.8)  # Gentle spacing
                    except Exception:
                        continue
        else:
            # Fallback to say command with musical notes
            say = shutil.which("say")
            if say:
                notes = ["[[rate 200]][[pbas 40]]♪", "[[rate 200]][[pbas 45]]♫", "[[rate 200]][[pbas 40]]♪"]
                while not self._stop_event.is_set():
                    for note in notes:
                        if self._stop_event.is_set():
                            break
                        try:
                            subprocess.run([say, note], 
                                         stdout=subprocess.DEVNULL, 
                                         stderr=subprocess.DEVNULL,
                                         timeout=1.0)
                            time.sleep(0.5)
                        except Exception:
                            continue
                            
    def _play_linux_tune(self) -> None:
        """Play tune on Linux using aplay, paplay, or beep."""
        # Try paplay (PulseAudio) first
        paplay = shutil.which("paplay")
        if paplay:
            # Simple tone generation (if we had audio files)
            pass
            
        # Try beep command
        beep = shutil.which("beep")
        if beep:
            while not self._stop_event.is_set():
                try:
                    # Gentle ascending/descending tone pattern
                    frequencies = [440, 523, 440, 349]  # A4, C5, A4, F4
                    for freq in frequencies:
                        if self._stop_event.is_set():
                            break
                        subprocess.run([beep, "-f", str(freq), "-l", "200"], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL,
                                     timeout=0.5)
                        time.sleep(0.3)
                except Exception:
                    break
        else:
            # Fallback to speaker-beep
            self._play_fallback_tune()
            
    def _play_windows_tune(self) -> None:
        """Play tune on Windows using PowerShell beeps."""
        pwsh = shutil.which("powershell") or shutil.which("pwsh")
        if pwsh:
            while not self._stop_event.is_set():
                try:
                    # Gentle beep pattern
                    frequencies = [440, 523, 440, 349]
                    for freq in frequencies:
                        if self._stop_event.is_set():
                            break
                        script = f"[Console]::Beep({freq}, 200)"
                        subprocess.run([pwsh, "-NoProfile", "-Command", script],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     timeout=0.5)
                        time.sleep(0.3)
                except Exception:
                    break
        else:
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
