"""Line-oriented desktop transport for plain text and terminal download bars."""

import re
import threading
from dataclasses import dataclass

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS = re.compile(r"^(.+?):\s*(\d{1,3})%\|[^|]*\|\s*(.*)$")
_UNKNOWN_PROGRESS = re.compile(r"^(.+?):\s*(\d+(?:\.\d+)?\s*[kMGT]?B\s*\[.*\])$")
_TIMING = re.compile(r"^(.*?)\s*\[([^<,]+)<([^,]+),\s*(.*?)\]$")


@dataclass(frozen=True)
class DownloadProgress:
    name: str
    percent: int | None
    detail: str


def parse_progress(text: str):
    """Recognise tqdm byte progress without swallowing ordinary messages."""
    match = _PROGRESS.fullmatch(text.strip())
    if match and 0 <= int(match[2]) <= 100:
        detail = match[3]
        timing = _TIMING.fullmatch(detail)
        if timing:
            remaining = f" · {timing[3]} remaining" if timing[3] != '?' else ''
            detail = f"{timing[1]} · {timing[4]}{remaining}"
        return DownloadProgress(match[1].strip().removeprefix('📥 '), int(match[2]), detail)
    match = _UNKNOWN_PROGRESS.fullmatch(text.strip())
    if match:
        return DownloadProgress(match[1].strip().removeprefix('📥 '), None, match[2])
    return None


class LogStream:
    """Capture newline and carriage-return updates from concurrent workers."""

    encoding = "utf-8"

    def __init__(self, emit):
        self.emit = emit
        self.buffer = ""
        self.lock = threading.RLock()

    def isatty(self):
        return False

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        with self.lock:
            self.buffer += text
            parts = re.split(r"[\r\n]", self.buffer)
            self.buffer = parts.pop()
            for line in parts:
                self._emit(line)
        return len(text)

    def _emit(self, line):
        line = _ANSI.sub("", line)
        if line.strip():
            self.emit(line + "\n")

    def flush(self):
        with self.lock:
            self._emit(self.buffer)
            self.buffer = ""


def clean_log(text):
    return _ANSI.sub("", text)
