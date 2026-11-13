"""Exercise native event creation/flags/release without posting desktop input."""
import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from jarvis.dictation.dictation_engine import _paste_cgevent

app = QApplication([])
cg = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
cf = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
cg.CGEventGetFlags.argtypes = [ctypes.c_void_p]
cg.CGEventGetFlags.restype = ctypes.c_uint64
cg.CGEventGetType.argtypes = [ctypes.c_void_p]
cg.CGEventGetType.restype = ctypes.c_uint32
posted = []


def intercept_post(tap, event):
    posted.append((cg.CGEventGetType(event), cg.CGEventGetFlags(event)))


proxy = SimpleNamespace(
    CGEventCreateKeyboardEvent=cg.CGEventCreateKeyboardEvent,
    CGEventSetFlags=cg.CGEventSetFlags,
    CGEventPost=intercept_post,
)
with patch('ctypes.cdll.LoadLibrary', side_effect=lambda path: proxy if 'CoreGraphics' in path else cf):
    results = []
    worker = threading.Thread(target=lambda: results.append(_paste_cgevent()), daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert results == [True]
assert posted == [(10, 0x100000), (11, 0x100000)], posted
print('✅ Native Cmd+V event pair prepared and released on a background worker; desktop input was not posted')
