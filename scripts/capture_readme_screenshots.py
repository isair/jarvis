"""Capture the real desktop widgets with isolated, non-personal demo data.

Run with the project's Python environment:
    PYTHONPATH=src python scripts/capture_readme_screenshots.py

No daemon is launched. Configuration is isolated in a temporary directory;
network access and worker starts are blocked. Images are unretouched widget
captures, not mock-ups. Requires PyQt6 and the desktop dependencies.
"""

from contextlib import ExitStack
from datetime import datetime
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_SCALE_FACTOR', '2')

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest


OUTPUT = Path(__file__).resolve().parents[1] / 'docs' / 'img'


class DemoClock:
    @staticmethod
    def now():
        return datetime(2026, 1, 1, 9, 41)


def capture(window, name):
    window.show()
    QTest.qWait(100)
    assert window.grab().save(str(OUTPUT / name))
    window.hide()
    print(f'📸 Captured {name}')


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    with TemporaryDirectory(prefix='jarvis-screenshots-') as temporary, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'XDG_CONFIG_HOME': temporary,
            'XDG_DATA_HOME': temporary,
            'JARVIS_CONFIG_PATH': str(Path(temporary) / 'config.json'),
        }))
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Screenshots must be offline')))
        from desktop_app.qt_worker import KeepAliveWorker
        stack.enter_context(patch.object(KeepAliveWorker, 'start', side_effect=AssertionError('Screenshots must not start workers')))
        from desktop_app import chat_window, setup_wizard, settings_window
        from desktop_app import app as desktop_app
        stack.enter_context(patch.object(desktop_app.time, 'strftime', return_value='09:41:00'))

        stack.enter_context(patch.object(chat_window, 'get_hot_window_messages', return_value=[]))
        stack.enter_context(patch.object(chat_window, 'datetime', DemoClock))
        chat = chat_window.ChatWindow(submit_fn=lambda text: None)
        chat.resize(480, 720)
        chat._append_user('Help me make room for a slower morning.')
        chat._append_assistant(
            'Start with one small thing.\n\nA coffee without a screen, a short walk, '
            'or ten minutes with that book you keep meaning to read.\n\n'
            'What does your morning look like?'
        )
        chat._append_user('Coffee and a walk sounds perfect.')
        capture(chat, 'chat-window.png')

        wizard = setup_wizard.SetupWizard()
        wizard.setStartId(wizard.provider_choice_page_id)
        wizard.resize(960, 680)
        capture(wizard, 'setup-provider.png')

        logs = desktop_app.LogViewerWindow()
        logs.resize(900, 560)
        for line in (
            '🚀 Jarvis daemon started',
            '💾 Initialising dialogue memory…',
            '✓ Dialogue memory initialised',
            '⚠️ 📍 Optional location features unavailable. Add a GeoLite2 database in Setup → Location.',
            '🔊 Initialising TTS engine (piper)…',
            '✓ TTS engine started',
            '🎤 Preparing speech recognition…',
            'weights.npz:  48%|████▊     | 730M/1.52G [00:48<00:52,15.2MB/s]',
        ):
            logs.append_log(line)
        capture(logs, 'logs.png')

        settings = settings_window.SettingsWindow()
        settings.resize(960, 680)
        for index in range(settings._sidebar.count()):
            if 'Speech Recognition' in settings._sidebar.item(index).text():
                settings._sidebar.setCurrentRow(index)
                break
        capture(settings, 'settings-window.png')


if __name__ == '__main__':
    main()
