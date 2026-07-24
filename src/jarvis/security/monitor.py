"""Background READ-ONLY monitor (opt-in via config.enabled)."""

from __future__ import annotations

import threading
import time
from typing import Optional

from jarvis.debug import debug_log
from jarvis.security.service import SecurityCenterService


class SecurityMonitor:
    """Daemon thread — never blocks voice path; fail-open."""

    def __init__(self, service: SecurityCenterService) -> None:
        self.service = service
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not self.service.config.enabled:
            debug_log("security monitor not started (disabled)", "security")
            return
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cora-security-monitor", daemon=True)
        self._thread.start()
        debug_log("security monitor started", "security")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def _loop(self) -> None:
        cfg = self.service.config
        # Stagger: first full audit after short delay
        next_full = time.time() + 5
        next_hardening = time.time() + cfg.hardening_interval_sec
        while not self._stop.is_set():
            now = time.time()
            try:
                if now >= next_full:
                    self.service.run_audit(create_baseline_if_missing=True)
                    next_full = now + max(60, cfg.persistence_interval_sec)
                if now >= next_hardening:
                    # Hardening is included in full audit; just reschedule
                    next_hardening = now + max(3600, cfg.hardening_interval_sec)
            except Exception as exc:  # noqa: BLE001 — monitor must not die loudly
                debug_log(f"security monitor error: {type(exc).__name__}", "security")
            self._stop.wait(min(30, max(5, cfg.process_interval_sec)))
