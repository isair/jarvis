"""
Tests for splash_screen.py functionality.

Tests the splash screen component used during application startup.
Note: These tests use headless mode where possible.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestSplashScreenImport:
    """Tests for splash screen module import."""

    def test_can_import_module(self):
        """splash_screen module should be importable."""
        from jarvis import splash_screen
        assert splash_screen is not None

    def test_splash_screen_class_exists(self):
        """SplashScreen class should be defined."""
        from jarvis.splash_screen import SplashScreen
        assert SplashScreen is not None

    def test_animated_orb_class_exists(self):
        """AnimatedOrb class should be defined."""
        from jarvis.splash_screen import AnimatedOrb
        assert AnimatedOrb is not None


class TestSplashScreenFunctionality:
    """Tests for splash screen functionality."""

    @pytest.fixture
    def qapp(self):
        """Create a QApplication for testing."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_splash_screen_instantiation(self, qapp):
        """SplashScreen should instantiate without error."""
        from jarvis.splash_screen import SplashScreen
        splash = SplashScreen()
        assert splash is not None
        splash.close()

    def test_splash_screen_set_status(self, qapp):
        """SplashScreen should allow setting status text."""
        from jarvis.splash_screen import SplashScreen
        splash = SplashScreen()
        splash.set_status("Test status message")
        assert splash._status_label.text() == "Test status message"
        splash.close()

    def test_splash_screen_close_splash(self, qapp):
        """SplashScreen close_splash should stop animation and close."""
        from jarvis.splash_screen import SplashScreen
        splash = SplashScreen()
        splash.show()
        splash.close_splash()
        # Orb animation should be stopped
        assert not splash._orb._timer.isActive()

    def test_animated_orb_instantiation(self, qapp):
        """AnimatedOrb should instantiate and start animation."""
        from jarvis.splash_screen import AnimatedOrb
        orb = AnimatedOrb()
        assert orb is not None
        assert orb._timer.isActive()
        orb.stop()
        assert not orb._timer.isActive()


class TestSplashScreenColors:
    """Tests for splash screen theme colors."""

    def test_uses_theme_colors(self):
        """SplashScreen should use colors from themes module."""
        from jarvis.splash_screen import COLORS
        from jarvis.themes import COLORS as THEME_COLORS

        # Should be using the same color constants
        assert COLORS == THEME_COLORS
