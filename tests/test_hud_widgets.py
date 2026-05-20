"""Tests for reusable futuristic HUD desktop widgets."""

from PyQt6.QtWidgets import QLabel


def test_hud_panel_exposes_title_and_body(qapp):
    """HudPanel should provide a titled content area for desktop windows."""
    from desktop_app.hud_widgets import HudPanel

    panel = HudPanel("System Readiness", "Live prerequisite telemetry")

    assert panel.objectName() == "hudPanel"
    assert panel.title_label.text() == "System Readiness"
    assert panel.subtitle_label.text() == "Live prerequisite telemetry"
    assert panel.body_layout is not None

    child = QLabel("Nested content")
    panel.addWidget(child)
    assert panel.body_layout.count() == 1


def test_status_pill_tracks_status_object_name(qapp):
    """StatusPill should expose semantic status through stable object names."""
    from desktop_app.hud_widgets import StatusPill

    pill = StatusPill("Whisper", "warning")

    assert pill.objectName() == "statusPill-warning"
    assert pill.text() == "Whisper"

    pill.set_status("Online", "success")
    assert pill.objectName() == "statusPill-success"
    assert pill.text() == "Online"


def test_metric_tile_keeps_label_value_and_detail(qapp):
    """MetricTile should keep dashboard data readable and updatable."""
    from desktop_app.hud_widgets import MetricTile

    tile = MetricTile("Models", "3/3", "Ollama assets ready")

    assert tile.objectName() == "metricTile"
    assert tile.label_label.text() == "Models"
    assert tile.value_label.text() == "3/3"
    assert tile.detail_label.text() == "Ollama assets ready"

    tile.set_value("2/3", "Embedding model missing")
    assert tile.value_label.text() == "2/3"
    assert tile.detail_label.text() == "Embedding model missing"


def test_theme_contains_hud_selectors():
    """The shared stylesheet should define the HUD visual language centrally."""
    from desktop_app.themes import JARVIS_THEME_STYLESHEET

    required_selectors = [
        "QFrame#hudPanel",
        "QLabel#hudPanelTitle",
        "QLabel#statusPill-success",
        "QFrame#metricTile",
        "QFrame#telemetryRow",
        "QLabel#hudHeaderTitle",
        "QPushButton#modelOptionButton",
        "QTextEdit#wizardConsole",
    ]

    for selector in required_selectors:
        assert selector in JARVIS_THEME_STYLESHEET


def test_hud_header_exposes_title_and_status(qapp):
    """HudHeader should surface title, subtitle, and optional status pill."""
    from desktop_app.hud_widgets import HudHeader

    header = HudHeader("Jarvis", "Local AI", status_text="READY", status="success")

    assert header.title_label.text() == "Jarvis"
    assert header.subtitle_label.text() == "Local AI"
    assert header.status_pill is not None
    assert header.status_pill.text() == "READY"


def test_section_divider_has_stable_object_name(qapp):
    """SectionDivider should be themeable via object name."""
    from desktop_app.hud_widgets import SectionDivider

    divider = SectionDivider("Parameters")

    assert divider.objectName() == "sectionDivider"


def test_hud_grid_backdrop_advances_phase(qapp):
    """HudGridBackdrop should accept animation phase updates."""
    from desktop_app.hud_widgets import HudGridBackdrop

    backdrop = HudGridBackdrop()
    backdrop.resize(120, 80)
    backdrop.advance_phase(0.5)
    assert backdrop._phase == 0.5
