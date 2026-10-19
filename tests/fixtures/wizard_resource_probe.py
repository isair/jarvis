"""Open setup when the optional macOS keyboard assistant is unavailable."""
import AppKit
import objc
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWizardPage

from desktop_app import setup_wizard as ui

original_lookup = AppKit.NSWorkspace.URLForApplicationWithBundleIdentifier_


def without_keyboard_assistant(self, identifier):
    if identifier == 'com.apple.KeyboardSetupAssistant':
        return None
    return original_lookup(self, identifier)


objc.classAddMethods(AppKit.NSWorkspace, [objc.selector(
    without_keyboard_assistant,
    selector=b'URLForApplicationWithBundleIdentifier:', signature=b'@@:@',
)])
assert AppKit.NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_(
    'com.apple.KeyboardSetupAssistant') is None

# Display real pages without downloads, provider discovery or configuration saves.
for value in vars(ui).values():
    if isinstance(value, type) and issubclass(value, QWizardPage) and value.__module__ == ui.__name__:
        value.initializePage = lambda self: None

app = QApplication([])
wizard = ui.SetupWizard()
wizard.show()
QTest.qWait(100)
assert wizard.isVisible()
assert wizard.currentPage().isVisible()
wizard.close()
