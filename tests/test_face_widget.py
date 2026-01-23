"""
Tests for the FaceWindow positioning logic.
"""

from unittest.mock import patch, MagicMock
import pytest


class TestFaceWindowPositioning:
    """Tests for FaceWindow positioning on the right side of screen."""

    def test_positions_on_right_side_of_screen(self):
        """FaceWindow should position itself on the right side of the screen."""
        # Mock screen geometry
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(
            right=lambda: 1920,
            top=lambda: 0,
            height=lambda: 1080,
        )

        # Mock QApplication.primaryScreen
        with patch(
            "jarvis.face_widget.QApplication.primaryScreen", return_value=mock_screen
        ):
            # Import after patching to avoid needing actual display
            from jarvis.face_widget import FaceWindow

            # Mock the parent class __init__ to avoid Qt initialization issues
            with patch.object(FaceWindow, "__init__", lambda self, parent=None: None):
                window = FaceWindow.__new__(FaceWindow)
                window._width = 350
                window._height = 450

                # Mock width() and height() methods
                window.width = lambda: 350
                window.height = lambda: 450

                # Track move calls
                move_calls = []
                window.move = lambda x, y: move_calls.append((x, y))

                # Call the positioning method
                window._position_on_right()

                # Verify positioning
                assert len(move_calls) == 1
                x, y = move_calls[0]

                # Should be on right side with 20px margin
                # x = 1920 - 350 - 20 = 1550
                assert x == 1550

                # Should be vertically centered
                # y = 0 + (1080 - 450) // 2 = 315
                assert y == 315

    def test_handles_none_screen_gracefully(self):
        """FaceWindow should handle missing screen gracefully."""
        with patch(
            "jarvis.face_widget.QApplication.primaryScreen", return_value=None
        ):
            from jarvis.face_widget import FaceWindow

            with patch.object(FaceWindow, "__init__", lambda self, parent=None: None):
                window = FaceWindow.__new__(FaceWindow)

                move_calls = []
                window.move = lambda x, y: move_calls.append((x, y))

                # Should not raise an exception
                window._position_on_right()

                # Should not move if no screen
                assert len(move_calls) == 0

    def test_adapts_to_different_screen_sizes(self):
        """FaceWindow should adapt to different screen sizes."""
        test_cases = [
            # (screen_right, screen_top, screen_height, expected_x, expected_y)
            (1920, 0, 1080, 1550, 315),  # Standard 1080p
            (2560, 0, 1440, 2190, 495),  # 1440p
            (3840, 0, 2160, 3470, 855),  # 4K
            (1366, 0, 768, 996, 159),  # Common laptop
        ]

        window_width = 350
        window_height = 450
        margin = 20

        for screen_right, screen_top, screen_height, expected_x, expected_y in test_cases:
            mock_screen = MagicMock()
            mock_screen.availableGeometry.return_value = MagicMock(
                right=lambda r=screen_right: r,
                top=lambda t=screen_top: t,
                height=lambda h=screen_height: h,
            )

            with patch(
                "jarvis.face_widget.QApplication.primaryScreen",
                return_value=mock_screen,
            ):
                from jarvis.face_widget import FaceWindow

                with patch.object(
                    FaceWindow, "__init__", lambda self, parent=None: None
                ):
                    window = FaceWindow.__new__(FaceWindow)
                    window.width = lambda: window_width
                    window.height = lambda: window_height

                    move_calls = []
                    window.move = lambda x, y: move_calls.append((x, y))

                    window._position_on_right()

                    assert len(move_calls) == 1
                    x, y = move_calls[0]
                    assert x == expected_x, f"For screen {screen_right}x{screen_height}"
                    assert y == expected_y, f"For screen {screen_right}x{screen_height}"
