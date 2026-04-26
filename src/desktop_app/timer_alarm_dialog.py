"""Timer-alarm dialogue: looping alarm sound + Dismiss button.

Opens when the daemon emits a ``__TIMER_ALARM__:`` start event over
stdout (see :mod:`jarvis.tools.builtin.timer`). Plays a short alarm
clip on loop via :class:`QSoundEffect` and shows a single Dismiss
button. The dialogue closes automatically on:

- The user clicking Dismiss (or the window's X button).
- A ``__TIMER_ALARM__:`` stop event arriving on stdout (timer
  cancelled, "stop" voice command, another timer set, auto-cap fired).
- Hitting the local 30-second auto-cap.

The audio is owned by the dialogue, not by the daemon, so the alarm is
silenced the moment the dialogue closes regardless of how it closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QObject
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

# QtMultimedia ships separately from QtWidgets in some PyQt6 builds (and
# isn't always present on headless test rigs). Fall back to a silent
# stub when unavailable: the dialogue still opens, the user still gets
# the visual cue, and Dismiss / IPC stop still work — there's just no
# looped audio. The daemon-side BEL fallback covers headless callers.
try:
    from PyQt6.QtMultimedia import QSoundEffect  # type: ignore
    _HAS_SOUND = True
except Exception:  # pragma: no cover - import-time guard
    QSoundEffect = None  # type: ignore[assignment]
    _HAS_SOUND = False

from .themes import JARVIS_THEME_STYLESHEET

# IPC protocol prefix; must match
# `jarvis.tools.builtin.timer.TIMER_ALARM_IPC_PREFIX`.
TIMER_ALARM_IPC_PREFIX = "__TIMER_ALARM__:"

# Local hard cap in case the daemon never emits a stop event (e.g. the
# daemon crashed). Mirrors the daemon-side `_ALARM_AUTO_STOP_SEC`.
_ALARM_AUTO_STOP_MS = 30 * 1000


class TimerAlarmSignals(QObject):
    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal(dict)


class TimerAlarmDialog(QDialog):
    """Single-instance alarm dialogue with looping audio.

    Reused for every fired timer; if a second alarm arrives while this
    dialogue is already showing, the title updates and the audio
    keeps looping (one dialogue, multiple ringing alarms).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._active_ids: set[str] = set()
        self._auto_stop = QTimer(self)
        self._auto_stop.setSingleShot(True)
        self._auto_stop.timeout.connect(self._dismiss)

        self._setup_ui()
        if _HAS_SOUND:
            self._sound = QSoundEffect(self)
            self._sound.setLoopCount(QSoundEffect.Loop.Infinite)
            self._sound.setVolume(0.85)
            wav_path = Path(__file__).parent / "desktop_assets" / "alarm.wav"
            self._sound.setSource(QUrl.fromLocalFile(str(wav_path)))
        else:
            self._sound = None

    def _setup_ui(self) -> None:
        self.setWindowTitle("Timer alarm")
        self.setMinimumWidth(380)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.title_label = QLabel("⏰ Timer done")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.body_label = QLabel("")
        self.body_label.setObjectName("subtitle")
        self.body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body_label.setWordWrap(True)
        layout.addWidget(self.body_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.dismiss_btn = QPushButton("Dismiss")
        self.dismiss_btn.setDefault(True)
        self.dismiss_btn.clicked.connect(self._dismiss)
        button_row.addWidget(self.dismiss_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

    def handle_start(self, payload: dict) -> None:
        timer_id = str(payload.get("id") or "")
        label = payload.get("label")
        duration_human = payload.get("duration_human") or ""
        if timer_id:
            self._active_ids.add(timer_id)

        title = "⏰ Timer done"
        if label:
            title = f"⏰ Timer '{label}' done"
        self.title_label.setText(title)
        body_bits = []
        if duration_human:
            body_bits.append(f"{duration_human} elapsed.")
        if len(self._active_ids) > 1:
            body_bits.append(f"{len(self._active_ids)} alarms ringing.")
        self.body_label.setText(" ".join(body_bits) or "Timer elapsed.")

        if self._sound is not None:
            try:
                if not self._sound.isPlaying():
                    self._sound.play()
            except Exception:
                pass

        # Restart the local auto-cap each time a fresh alarm arrives so
        # the dialogue stays open at least 30s after the latest one.
        self._auto_stop.start(_ALARM_AUTO_STOP_MS)

        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def handle_stop(self, payload: dict) -> None:
        if payload.get("all"):
            self._active_ids.clear()
        else:
            timer_id = str(payload.get("id") or "")
            if timer_id:
                self._active_ids.discard(timer_id)
        if not self._active_ids:
            self._dismiss()

    def _dismiss(self) -> None:
        self._active_ids.clear()
        if self._sound is not None:
            try:
                if self._sound.isPlaying():
                    self._sound.stop()
            except Exception:
                pass
        self._auto_stop.stop()
        if self.isVisible():
            self.hide()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._dismiss()
        super().closeEvent(event)

    def reject(self) -> None:  # Esc key
        self._dismiss()
        super().reject()


def parse_alarm_event(line: str) -> Optional[dict]:
    """Return the parsed event dict if ``line`` is a TIMER_ALARM IPC line."""
    line = line.strip()
    if not line.startswith(TIMER_ALARM_IPC_PREFIX):
        return None
    try:
        return json.loads(line[len(TIMER_ALARM_IPC_PREFIX):])
    except Exception:
        return None
