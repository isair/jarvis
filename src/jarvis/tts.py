from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import os
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
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$v.Volume = 100; $v.Rate = {rate}; "
                f"{voice_set} "
                f"$v.Speak({json_escape_ps(text)});"
            )
            try:
                with self._process_lock:
                    self._current_process = subprocess.Popen([pwsh, "-NoProfile", "-Command", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
            with tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False, encoding="utf-8") as tf:
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


def json_escape_ps(s: str) -> str:
    # For PowerShell, use double quotes and escape internal double quotes
    # This avoids issues with apostrophes in contractions like "you're"
    escaped = s.replace('"', '""')
    return '"' + escaped + '"'
