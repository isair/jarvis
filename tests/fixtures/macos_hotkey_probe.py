import threading
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from pynput import keyboard
from jarvis.dictation.dictation_engine import _create_keyboard_listener
import pynput.keyboard._darwin as backend
import Quartz

app = QApplication([])

failures = []
def run(listener):
    try:
        listener._run()
    except Exception as error:
        failures.append(str(error))

# Avoid monitoring or posting real keyboard events. Exercise the native
# decoder with synthetic events and the no-event-tap startup lifecycle.
for factory, expected_failure in [(keyboard.Listener, True), (_create_keyboard_listener, False)]:
    failures.clear()
    listener = factory(on_press=lambda key: None, on_release=lambda key: None)
    with patch.object(backend, 'keycode_context', side_effect=RuntimeError('TSM background access forbidden')), patch.object(listener, '_create_event_tap', return_value=None):
        worker = threading.Thread(target=run, args=(listener,))
        worker.start(); worker.join(3)
    assert not worker.is_alive()
    assert bool(failures) == expected_failure, failures
    print('✅', factory.__name__, 'fails safely under TSM sentinel' if expected_failure else 'starts without TSM access')

listener = _create_keyboard_listener(on_press=lambda key: None, on_release=lambda key: None)
for text in ['a', 'é', 'ж', '字']:
    event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
    Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
    assert listener._event_to_key(event).char == text
for code, flag, key in [(58, Quartz.kCGEventFlagMaskAlternate, keyboard.Key.alt), (61, Quartz.kCGEventFlagMaskAlternate, keyboard.Key.alt_r), (59, Quartz.kCGEventFlagMaskControl, keyboard.Key.ctrl)]:
    event = Quartz.CGEventCreateKeyboardEvent(None, code, True)
    Quartz.CGEventSetType(event, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(event, flag)
    assert listener._event_to_key(event) == key
    Quartz.CGEventSetFlags(event, 0)
    assert listener._event_to_key(event) == key
print('✅ Native Unicode and left/right modifier decoding survives without a Carbon context, no global key events captured or posted')
