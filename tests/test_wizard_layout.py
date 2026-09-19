"""Wizard controls stay readable when the page exceeds the display height."""

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QComboBox, QLineEdit, QPushButton, QScrollArea, QWizard

from desktop_app import setup_wizard as ui


@pytest.mark.parametrize("page_type", [
    ui.OpenAICompatiblePage, ui.ModelsPage, ui.ProviderChoicePage,
    ui.DictationPage, ui.SearchProvidersPage,
    ui.WelcomePage, ui.OllamaInstallPage, ui.OllamaServerPage,
    ui.WhisperSetupPage, ui.LocationPage, ui.MCPPage, ui.CompletePage,
])
@pytest.mark.parametrize("height", [600, 800])
def test_controls_fit_on_short_display(qapp, monkeypatch, page_type, height):
    # Exercise real styled layouts without discovery, installs or config writes.
    monkeypatch.setattr(page_type, "initializePage", lambda self: None)
    wizard = QWizard()
    wizard.setWizardStyle(QWizard.WizardStyle.ModernStyle)
    wizard.setStyleSheet(ui.JARVIS_THEME_STYLESHEET)
    page = page_type()
    wizard.addPage(page)
    wizard.resize(700, height)
    wizard.show()
    QTest.qWait(100)
    try:
        assert wizard.height() <= height
        if isinstance(page, ui.OpenAICompatiblePage):
            page._connect_status.setText("Could not load models. " * 10)
            page._use_ollama_embed.show()
            page._openai_link_cb.setChecked(True)
            page._openai_link_cb.setChecked(False)
            QTest.qWait(100)
            assert wizard.height() <= height
        for control in page.findChildren((QComboBox, QLineEdit, QPushButton)):
            if control.isVisible() and not isinstance(control.parentWidget(), QComboBox):
                assert control.height() >= control.minimumSizeHint().height(), (
                    type(control).__name__, control.height(), control.minimumSizeHint().height()
                )
        scrolls = page.findChildren(QScrollArea)
        assert scrolls
        scroll = scrolls[0]
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        bottom = scroll.widget().mapTo(
            scroll.viewport(), QPoint(0, scroll.widget().height())
        )
        assert bottom.y() <= scroll.viewport().height()
        button = wizard.button(QWizard.WizardButton.FinishButton)
        assert button.isVisible()
        assert wizard.rect().contains(button.geometry())
    finally:
        wizard.close()


def test_wizard_initial_size_fits_available_screen(qapp, monkeypatch):
    monkeypatch.setattr(ui.WhisperSetupPage, "initializePage", lambda self: None)
    wizard = ui.SetupWizard()
    wizard.show()
    QTest.qWait(100)
    try:
        assert wizard.frameGeometry().height() <= wizard.screen().availableGeometry().height()
    finally:
        wizard.close()
