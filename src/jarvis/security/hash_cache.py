"""SHA-256 cache keyed by path + size + mtime (avoid rehash every cycle)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from jarvis.utils.atomic_write import atomic_write_json


class HashCache:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._mem: dict[str, dict[str, str]] = {}
        if path and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._mem = data
            except (OSError, json.JSONDecodeError):
                self._mem = {}

    def get_or_compute(self, file_path: Path, *, size: int, mtime: float) -> str:
        key = str(file_path).lower()
        token = f"{size}:{int(mtime)}"
        entry = self._mem.get(key)
        if entry and entry.get("token") == token and entry.get("sha256"):
            return entry["sha256"]
        try:
            h = hashlib.sha256()
            with file_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            return ""
        self._mem[key] = {"token": token, "sha256": digest}
        return digest

    def save(self) -> None:
        if self.path is None:
            return
        # Cap cache size
        items = list(self._mem.items())
        if len(items) > 5000:
            self._mem = dict(items[-5000:])
        atomic_write_json(self.path, self._mem)
