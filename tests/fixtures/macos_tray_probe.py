import sys
import objc
from AppKit import NSApplication, NSEvent, NSEventTypeApplicationDefined, NSEventTypeLeftMouseDown, NSEventTypeRightMouseDown, NSEventTypeAppKitDefined, NSEventTypeSystemDefined
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt

app=QApplication([])
cls=objc.lookUpClass('QStatusItemDelegate')
original_init=cls.instanceMethodForSelector_(b'initWithSysTray:')
delegates=[]
def captured_init(self, pointer):
    result=original_init(self,pointer)
    delegates.append(result)
    return result
objc.classAddMethods(cls, [objc.selector(captured_init, selector=b'initWithSysTray:', signature=b'@@:^v')])
tray=QSystemTrayIcon()
pixmap=QPixmap(16,16);pixmap.fill(Qt.GlobalColor.blue)
tray.setIcon(QIcon(pixmap))
menu=QMenu()
menu.addAction("Tray probe")
tray.setContextMenu(menu)
if '--unguarded' not in sys.argv:
    from desktop_app.macos_tray import install_macos_tray_event_guard
    assert install_macos_tray_event_guard()
    assert install_macos_tray_event_guard(), 'Repeated installation must be safe'
tray.show()
assert delegates, 'No captured Qt delegate'
print('✅ Captured own Qt tray delegate', flush=True)
event=NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(NSEventTypeApplicationDefined,(0,0),0,0,0,None,0,0,0)
def current_event(self):
    return event
objc.classAddMethods(NSApplication,[objc.selector(current_event,selector=b'currentEvent',signature=b'@@:')])
assert NSApplication.sharedApplication().currentEvent().type()==NSEventTypeApplicationDefined
print('🔎 Invoke Qt tray tracking with an application-defined event',flush=True)
for event_type in [NSEventTypeApplicationDefined, NSEventTypeAppKitDefined, NSEventTypeSystemDefined]:
    event=NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(event_type,(0,0),0,0,0,None,0,0,0)
    delegates[0].statusItemMenuBeganTracking_(None)
    delegates[0].statusItemClicked()
print('✅ Tray tracking returned safely',flush=True)
delegates[0].statusItemClicked()
received=[]
tray.activated.connect(received.append)
event=NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(NSEventTypeLeftMouseDown,(0,0),0,0,0,None,1,1,0)
delegates[0].statusItemClicked()
assert received == [QSystemTrayIcon.ActivationReason.Trigger], received
event=NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(NSEventTypeLeftMouseDown,(0,0),0,0,0,None,1,2,0)
delegates[0].statusItemMenuBeganTracking_(None)
assert received[-1] == QSystemTrayIcon.ActivationReason.DoubleClick
event=NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(NSEventTypeRightMouseDown,(0,0),0,0,0,None,1,1,0)
delegates[0].statusItemClicked()
assert received[-1] == QSystemTrayIcon.ActivationReason.Context
event=None
delegates[0].statusItemClicked()
delegates[0].statusItemMenuBeganTracking_(None)
print('✅ Real mouse activation is preserved',flush=True)
tray.hide()
