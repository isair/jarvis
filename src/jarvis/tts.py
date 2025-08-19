from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
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
        pwsh = shutil.which("powershell") or shutil.which("pwsh")
        if not pwsh:
            return False
        # Use SAPI via PowerShell
        # Optionally set Rate (-10..10) and Voice if specified
        rate = 0 if self.rate is None else max(-10, min(10, int(self.rate)))
        voice_set = f"$v.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet);"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$v.Rate = {rate}; "
            f"{voice_set} "
            f"$v.Speak({json_escape_ps(text)});"
        )
        
        try:
            with self._process_lock:
                self._current_process = subprocess.Popen([pwsh, "-NoProfile", "-Command", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
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


def json_escape_ps(s: str) -> str:
    # Escape for PowerShell single-quoted string: double single quotes
    return "'" + s.replace("'", "''") + "'"
