from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import os
import warnings
from typing import Optional, Callable


class TextToSpeech:
    def __init__(self, enabled: bool = True, voice: Optional[str] = None, rate: Optional[int] = None) -> None:
        self.enabled = enabled
        self.voice = voice
        self.rate = rate
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._should_interrupt = threading.Event()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Ensure any active speech is interrupted immediately
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread on first speak
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        try:
            self._q.put_nowait(text)
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately"""
        self._should_interrupt.set()
        with self._process_lock:
            if self._current_process is not None:
                try:
                    if platform.system().lower() == "windows":
                        self._current_process.terminate()
                    else:
                        self._current_process.send_signal(signal.SIGTERM)
                    # Give process a moment to terminate gracefully
                    try:
                        self._current_process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        self._current_process.kill()
                        self._current_process.wait()
                except Exception:
                    pass
                self._current_process = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                self._speak_once(text)
            except Exception:
                continue

    def _speak_once(self, text: str) -> None:
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        system = platform.system().lower()
        interrupted = False
        try:
            if system == "darwin":
                interrupted = self._mac_say(text)
            elif system == "windows":
                interrupted = self._win_sapi(text)
            else:
                interrupted = self._linux_say(text)
        finally:
            with self._process_lock:
                self._current_process = None
            self._is_speaking.clear()
            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception:
                    pass
                self._completion_callback = None

    def _mac_say(self, text: str) -> bool:
        """Returns True if interrupted, False if completed normally"""
        say_path = shutil.which("say")
        if not say_path:
            return False
        cmd = [say_path]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            # mac 'say' rate via -r words per minute
            cmd += ["-r", str(int(self.rate))]
        cmd += [text]
        
        try:
            with self._process_lock:
                self._current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for process to complete or be interrupted
            while self._current_process.poll() is None:
                if self._should_interrupt.is_set():
                    return True  # Interrupted
                try:
                    self._current_process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            return False  # Completed normally
        except Exception:
            return False

    def _win_sapi(self, text: str) -> bool:
        """Returns True if interrupted, False if completed normally"""
        # Prefer Windows PowerShell if available, else PowerShell 7
        pwsh = shutil.which("powershell") or shutil.which("pwsh")
        # Convert words-per-minute to Windows SAPI rate scale (-10 to 10)
        # We apply a small Windows-specific multiplier so that the perceived speed
        # better matches macOS "say" defaults at the same WPM.
        # Mapping: rate = round(((WPM * multiplier) - 200) / 10), clamped to [-10, 10]
        multiplier = 1.2
        base_wpm = 200.0 if self.rate is None else float(self.rate)
        adjusted_wpm = base_wpm * multiplier
        rate = (adjusted_wpm - 200.0) / 10.0
        rate = int(max(-10, min(10, round(rate))))  # Clamp to SAPI bounds and ensure int

        voice_set = f"$v.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet);"
        if pwsh:
            # Read the text to speak from stdin to avoid any quoting/interpolation issues
            script = (
                "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Speech; "
                "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$v.Volume = 100; $v.Rate = {rate}; "
                f"{voice_set} "
                "$t = [Console]::In.ReadToEnd(); "
                "$v.Speak($t);"
            )
            try:
                with self._process_lock:
                    self._current_process = subprocess.Popen(
                        [pwsh, "-NoProfile", "-Command", script],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                try:
                    if self._current_process.stdin is not None:
                        self._current_process.stdin.write(text.encode("utf-8", errors="replace"))
                        self._current_process.stdin.close()
                except Exception:
                    pass
                while self._current_process.poll() is None:
                    if self._should_interrupt.is_set():
                        return True
                    try:
                        self._current_process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        continue
                return False
            except Exception:
                pass  # Fall back to cscript below

        # Fallback: use Windows Script Host with VBScript (broadly available)
        cscript = shutil.which("cscript")
        if not cscript:
            return False
        vbs_text = text.replace('"', '""')
        vbs_code = (
            'Dim v\n'
            'Set v = CreateObject("SAPI.SpVoice")\n'
            f'v.Rate = {rate}\n'
            'v.Volume = 100\n'
            f'v.Speak("{vbs_text}")\n'
        )
        tmp_path = None
        try:
            # Use UTF-16 for VBScript source to preserve non-ASCII characters reliably on Windows
            with tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False, encoding="utf-16") as tf:
                tf.write(vbs_code)
                tmp_path = tf.name
            with self._process_lock:
                self._current_process = subprocess.Popen([cscript, "//nologo", tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while self._current_process.poll() is None:
                if self._should_interrupt.is_set():
                    return True
                try:
                    self._current_process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            return False
        except Exception:
            return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _linux_say(self, text: str) -> bool:
        """Returns True if interrupted, False if completed normally"""
        spd = shutil.which("spd-say")
        if spd:
            cmd = [spd, text]
            return self._run_speech_process(cmd)
        espeak = shutil.which("espeak")
        if espeak:
            cmd = [espeak]
            if self.rate:
                cmd += ["-s", str(int(self.rate))]
            if self.voice:
                cmd += ["-v", self.voice]
            cmd += [text]
            return self._run_speech_process(cmd)
        # If no TTS command found, silently skip
        return False

    def _run_speech_process(self, cmd: list[str]) -> bool:
        """Helper method to run speech process with interruption support"""
        try:
            with self._process_lock:
                self._current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for process to complete or be interrupted
            while self._current_process.poll() is None:
                if self._should_interrupt.is_set():
                    return True  # Interrupted
                try:
                    self._current_process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            return False  # Completed normally
        except Exception:
            return False

    # Loopback guard helpers
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


class ChatterboxTTS:
    """Experimental TTS implementation using Resemble AI's Chatterbox model."""
    
    def __init__(self, enabled: bool = True, voice: Optional[str] = None, rate: Optional[int] = None, 
                 device: str = "cuda", audio_prompt_path: Optional[str] = None, 
                 exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Chatterbox, kept for interface compatibility
        self.rate = rate    # Not directly supported in Chatterbox, kept for interface compatibility
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        
        # Threading and queue setup (same as TextToSpeech)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._should_interrupt = threading.Event()
        
        # Chatterbox model (eagerly loaded during initialization)
        self._model = None
        self._model_error = None
        self._system_tts = None  # For setup announcements
        
        # Lazy initialization flags
        self._initialized = False
        self._init_lock = threading.Lock()
        
    def _initialize_with_logging(self) -> None:
        """Initialize Chatterbox with proper logging and system announcements."""
        import sys
        
        print("🔧 [TTS] Initializing Chatterbox neural voice synthesis...", file=sys.stderr)
        
        # Create system TTS for announcements during setup
        self._system_tts = TextToSpeech(enabled=True)
        self._system_tts.start()
        self._system_tts.speak("Setting up advanced voice synthesis")
        
        try:
            print("📦 [TTS] Loading Chatterbox dependencies...", file=sys.stderr)
            
            # Import dependencies
            import torch
            import torchaudio as ta
            from chatterbox.tts import ChatterboxTTS as ChatterboxModel
            
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  [TTS] CUDA requested but not available, falling back to CPU", file=sys.stderr)
                actual_device = "cpu"
            else:
                actual_device = self.device
            
            print(f"🚀 [TTS] Loading Chatterbox model on {actual_device.upper()}...", file=sys.stderr)
            
            # Load model with proper device specification
            self._model = ChatterboxModel.from_pretrained(device=actual_device)
            
            print("✅ [TTS] Chatterbox neural voice synthesis ready!", file=sys.stderr)
            self._system_tts.speak("Advanced voice synthesis ready")
            
        except ImportError as e:
            self._model_error = f"Chatterbox dependencies not available: {e}"
            print(f"❌ [TTS] Missing dependencies: {self._model_error}", file=sys.stderr)
            self._system_tts.speak("Voice synthesis dependencies missing, using system voice")
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")
        except Exception as e:
            self._model_error = f"Failed to load Chatterbox model: {e}"
            print(f"❌ [TTS] Model loading failed: {self._model_error}", file=sys.stderr)
            self._system_tts.speak("Voice synthesis setup failed, using system voice")
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")
        finally:
            # Clean up system TTS after announcements
            if self._system_tts:
                # Give a moment for the last announcement to finish
                import time
                time.sleep(1.0)
                self._system_tts.stop()
                self._system_tts = None
                
    def _ensure_initialized(self) -> None:
        """Initialize heavy dependencies only once, when actually needed."""
        if self._initialized or not self.enabled:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_with_logging()
            self._initialized = True

    def _ensure_model(self) -> bool:
        """Check if Chatterbox model is loaded. Returns True if successful."""
        # Ensure lazy initialization happens before checking model
        self._ensure_initialized()
        if self._model is not None:
            return True
        if self._model_error is not None:
            return False
        return False
    
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize on first actual start
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Ensure any active speech is interrupted immediately
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread and lazy init on first speak
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        try:
            self._q.put_nowait(text)
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately"""
        self._should_interrupt.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                self._speak_once(text)
            except Exception:
                continue

    def _speak_once(self, text: str) -> None:
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False
        
        try:
            # Check if model is available
            if not self._ensure_model():
                # Fall back to system TTS if Chatterbox fails
                warnings.warn("Chatterbox TTS not available, skipping speech synthesis")
                return
                
            # Generate audio using Chatterbox
            import tempfile
            import pygame
            import os
            
            # Generate speech
            wav = self._model.generate(
                text, 
                audio_prompt_path=self.audio_prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
            try:
                # Save audio
                import torchaudio as ta
                ta.save(tmp_path, wav, self._model.sr)
                
                # Play audio using pygame (cross-platform)
                pygame.mixer.init(frequency=self._model.sr, size=-16, channels=1, buffer=1024)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to complete or interruption
                while pygame.mixer.music.get_busy():
                    if self._should_interrupt.is_set():
                        pygame.mixer.music.stop()
                        interrupted = True
                        break
                    pygame.time.wait(100)  # Check every 100ms
                    
            finally:
                # Cleanup
                pygame.mixer.quit()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            warnings.warn(f"Chatterbox TTS error: {e}")
        finally:
            self._is_speaking.clear()
            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception:
                    pass
                self._completion_callback = None

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


def create_tts_engine(engine: str = "system", enabled: bool = True, voice: Optional[str] = None, 
                      rate: Optional[int] = None, device: str = "cuda", audio_prompt_path: Optional[str] = None,
                      exaggeration: float = 0.5, cfg_weight: float = 0.5):
    """Factory function to create the appropriate TTS engine."""
    if engine.lower() == "chatterbox":
        return ChatterboxTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            device=device,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
    else:
        # Default to system TTS
        return TextToSpeech(enabled=enabled, voice=voice, rate=rate)


def json_escape_ps(s: str) -> str:
    # For PowerShell, use double quotes and escape internal double quotes
    # This avoids issues with apostrophes in contractions like "you're"
    escaped = s.replace('"', '""')
    return '"' + escaped + '"'
