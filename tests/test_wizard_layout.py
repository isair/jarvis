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
@pytest.mark.parametrize("width,height", [(700, 600), (700, 800), (960, 780)])
def test_controls_fit_on_short_display(qapp, monkeypatch, page_type, width, height):
    # Exercise real styled layouts without discovery, installs or config writes.
    monkeypatch.setattr(page_type, "initializePage", lambda self: None)
    wizard = QWizard()
    wizard.setWizardStyle(QWizard.WizardStyle.ModernStyle)
    wizard.setStyleSheet(ui.JARVIS_THEME_STYLESHEET + ui.WIZARD_STYLESHEET)
    page = page_type()
    wizard.addPage(page)
    wizard.resize(width, height)
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
        assert scroll.horizontalScrollBar().maximum() == 0
        if width == 960:
            assert scroll.verticalScrollBar().maximum() == 0
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


@pytest.mark.parametrize("installed", [False, True])
def test_whisper_install_buttons_fit_status_text(qapp, monkeypatch, installed):
    from PyQt6.QtWidgets import QStyle, QStyleOptionButton

    monkeypatch.setattr(ui, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(ui.WhisperSetupPage, "initializePage", lambda self: None)
    monkeypatch.setattr(ui, "check_mlx_whisper_status", lambda: ui.MLXWhisperStatus(
        is_apple_silicon=True,
        is_ffmpeg_installed=installed,
        ffmpeg_path="/opt/homebrew/bin/ffmpeg" if installed else None,
        is_mlx_whisper_installed=installed,
    ))
    wizard = QWizard()
    wizard.setStyleSheet(ui.JARVIS_THEME_STYLESHEET + ui.WIZARD_STYLESHEET)
    page = ui.WhisperSetupPage()
    wizard.addPage(page)
    wizard.resize(700, 600)
    wizard.show()
    page._refresh_mlx_status()
    QTest.qWait(100)
    try:
        for button in (page.install_ffmpeg_btn, page.install_mlx_btn):
            assert button.isVisible()
            assert button.isEnabled() is not installed
            option = QStyleOptionButton()
            button.initStyleOption(option)
            contents = button.style().subElementRect(
                QStyle.SubElement.SE_PushButtonContents, option, button
            )
            assert contents.height() >= button.fontMetrics().height()
            assert contents.width() >= button.fontMetrics().horizontalAdvance(button.text())
    finally:
        wizard.close()
