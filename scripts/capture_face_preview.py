"""Render the real face at desktop size, with isolated state and a fixed clock.

    PYTHONPATH=src python scripts/capture_face_preview.py --output-dir /tmp/jarvis-face

The contact sheet composites each native capture at its actual window opacity
on light and dark backgrounds. The optional animation shows the same widget
over time. No daemon, audio device, configuration or state file is accessed.
"""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_SCALE_FACTOR'] = '1'

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter
from PyQt6.QtWidgets import QApplication

from desktop_app import face_widget
from desktop_app.themes import COLORS


BACKGROUNDS = (COLORS['bg_secondary'], '#f2f0eb')


def composite(capture, opacity, label=''):
    """Keep the captured widget's pixels and scale; only supply its backdrop."""
    width, height = capture.width(), capture.height()
    board = QImage(width * 2, height + 32, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(board)
    for column, colour in enumerate(BACKGROUNDS):
        painter.fillRect(column * width, 0, width, height + 32, QColor(colour))
        painter.setOpacity(opacity)
        painter.drawImage(column * width, 32, capture)
        painter.setOpacity(1)
        painter.setPen(QColor(COLORS['text_secondary'] if column == 0 else '#52525b'))
        painter.setFont(QFont('Arial', 10))
        painter.drawText(QRectF(column * width, 4, width, 24), Qt.AlignmentFlag.AlignCenter, label)
    painter.end()
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--animate', action='store_true', help='Also render a short GIF of the live widget')
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    clock = SimpleNamespace(now=1000.0)
    state = SimpleNamespace(state=face_widget.JarvisState.ASLEEP)
    with ExitStack() as stack:
        stack.enter_context(patch.object(face_widget, 'get_jarvis_state', return_value=state))
        stack.enter_context(patch.object(face_widget._time, 'monotonic', side_effect=lambda: clock.now))
        stack.enter_context(patch.object(face_widget, 'debug_log'))
        stack.enter_context(patch.object(face_widget.random, 'uniform', side_effect=lambda low, high: (low + high) / 2))
        states = list(face_widget.JarvisState)
        sheet = QImage(220 * len(states), 624, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(sheet)
        for index, current in enumerate(states):
            clock.now = 1000.0
            state.state = face_widget.JarvisState.ASLEEP
            window = face_widget.FaceWindow()
            window.show()
            app.processEvents()
            window.face._animation_timer.stop()
            state.state = current
            window.face._animate()
            for frame in range(1, 76):
                clock.now = 1000.0 + frame / 30
                window.face._animate()
            capture = window.grab().toImage()
            capture.save(str(options.output_dir / f'{current.value}.png'))
            board = composite(capture, window.windowOpacity(), current.value.replace('_', ' ').upper())
            painter.drawImage(index * 220, 0, board.copy(0, 0, 220, 312))
            painter.drawImage(index * 220, 312, board.copy(220, 0, 220, 312))
            window.close()
        painter.end()
        sheet.save(str(options.output_dir / 'states-light-dark.png'))
        if options.animate:
            from PIL import Image

            clock.now = 1000.0
            state.state = face_widget.JarvisState.ASLEEP
            window = face_widget.FaceWindow()
            window.show()
            app.processEvents()
            window.face._animation_timer.stop()
            frames = []
            # A bounded glance makes its maximum idle movement visible in the preview.
            stack.enter_context(patch.object(face_widget.random, 'uniform', side_effect=lambda low, high: high))
            for current, duration in (
                (face_widget.JarvisState.IDLE, 16),
                (face_widget.JarvisState.LISTENING, 3),
                (face_widget.JarvisState.THINKING, 4),
                (face_widget.JarvisState.SPEAKING, 4),
                (face_widget.JarvisState.ASLEEP, 3),
            ):
                state.state = current
                window.face._animate()
                for _ in range(duration * 20):
                    clock.now += 1 / 20
                    window.face._animate()
                    board = composite(window.grab().toImage(), window.windowOpacity(), current.value.upper())
                    rgba = board.convertToFormat(QImage.Format.Format_RGBA8888)
                    frames.append(Image.frombytes('RGBA', (rgba.width(), rgba.height()),
                        rgba.constBits().asstring(rgba.sizeInBytes())).convert('RGB'))
            window.close()
            frames[0].save(options.output_dir / 'motion-light-dark.gif', save_all=True,
                           append_images=frames[1:], duration=50, loop=0, disposal=2)
    print(f'📸 Native face previews saved to {options.output_dir}')


if __name__ == '__main__':
    main()
