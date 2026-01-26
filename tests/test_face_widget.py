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
            "desktop_app.face_widget.QApplication.primaryScreen", return_value=mock_screen
        ):
            # Import after patching to avoid needing actual display
            from desktop_app.face_widget import FaceWindow

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
            "desktop_app.face_widget.QApplication.primaryScreen", return_value=None
        ):
            from desktop_app.face_widget import FaceWindow

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
                "desktop_app.face_widget.QApplication.primaryScreen",
                return_value=mock_screen,
            ):
                from desktop_app.face_widget import FaceWindow

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


class TestFaceWidgetImports:
    """Tests that daemon modules can import face_widget from the correct location.

    These smoke tests catch broken imports after refactoring, which previously
    failed silently due to try/except ImportError blocks in daemon code.
    """

    @pytest.mark.unit
    def test_face_widget_importable_from_desktop_app(self):
        """face_widget should be importable from desktop_app package."""
        from desktop_app.face_widget import get_jarvis_state, JarvisState
        assert get_jarvis_state is not None
        assert JarvisState is not None

    @pytest.mark.unit
    def test_jarvis_state_enum_has_expected_values(self):
        """JarvisState enum should have all expected states."""
        from desktop_app.face_widget import JarvisState

        expected_states = ['ASLEEP', 'IDLE', 'LISTENING', 'THINKING', 'SPEAKING']
        for state in expected_states:
            assert hasattr(JarvisState, state), f"JarvisState missing {state}"

    @pytest.mark.unit
    def test_tts_module_face_widget_import(self):
        """TTS module's face_widget import should work.

        This tests the actual import path used in jarvis/output/tts.py
        """
        # Simulate the import done in tts.py
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            success = True
        except ImportError:
            success = False

        assert success, "TTS module cannot import face_widget - check import path"

    @pytest.mark.unit
    def test_listener_module_face_widget_import(self):
        """Listener module's face_widget import should work.

        This tests the actual import path used in jarvis/listening/listener.py
        """
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            success = True
        except ImportError:
            success = False

        assert success, "Listener module cannot import face_widget - check import path"

    @pytest.mark.unit
    def test_state_manager_module_face_widget_import(self):
        """State manager module's face_widget import should work.

        This tests the actual import path used in jarvis/listening/state_manager.py
        """
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            success = True
        except ImportError:
            success = False

        assert success, "State manager cannot import face_widget - check import path"

    @pytest.mark.unit
    def test_reply_engine_module_face_widget_import(self):
        """Reply engine module's face_widget import should work.

        This tests the actual import path used in jarvis/reply/engine.py
        """
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            success = True
        except ImportError:
            success = False

        assert success, "Reply engine cannot import face_widget - check import path"


class TestJarvisStateManager:
    """Tests for JarvisStateManager cross-process state sharing."""

    @pytest.fixture(autouse=True)
    def cleanup_state_file(self):
        """Clean up state file and singleton before/after each test."""
        import tempfile
        import os
        from desktop_app import face_widget

        state_file = os.path.join(tempfile.gettempdir(), "jarvis_state")

        # Reset singleton before test
        face_widget._jarvis_state_instance = None

        # Clean up state file
        if os.path.exists(state_file):
            os.remove(state_file)

        yield

        # Reset singleton after test
        face_widget._jarvis_state_instance = None

        # Clean up state file
        if os.path.exists(state_file):
            os.remove(state_file)

    @pytest.mark.unit
    def test_state_manager_creates_file_if_not_exists(self):
        """State manager should create state file if it doesn't exist."""
        import tempfile
        import os
        from desktop_app.face_widget import get_jarvis_state, JarvisState

        state_file = os.path.join(tempfile.gettempdir(), "jarvis_state")

        # File shouldn't exist before getting state manager
        assert not os.path.exists(state_file)

        # Get state manager (creates singleton)
        sm = get_jarvis_state()

        # File should now exist
        assert os.path.exists(state_file)

        # Default state should be ASLEEP
        assert sm.state == JarvisState.ASLEEP

    @pytest.mark.unit
    def test_state_manager_always_starts_asleep(self):
        """State manager should always start ASLEEP, ignoring stale file state.

        The state file is for cross-process communication during a session,
        not for persisting state across app restarts. A fresh launch should
        always start in ASLEEP state.
        """
        import tempfile
        import os
        from desktop_app import face_widget
        from desktop_app.face_widget import JarvisState

        state_file = os.path.join(tempfile.gettempdir(), "jarvis_state")

        # Create file with SPEAKING state (leftover from previous session)
        with open(state_file, 'w') as f:
            f.write("speaking")

        # Reset singleton to simulate a fresh app launch
        face_widget._jarvis_state_instance = None

        # Get state manager - should start ASLEEP, not read stale file
        sm = face_widget.get_jarvis_state()

        # State should be ASLEEP (fresh start), not SPEAKING (stale state)
        assert sm.state == JarvisState.ASLEEP

    @pytest.mark.unit
    def test_state_manager_file_based_sharing(self):
        """State changes should persist to file for cross-process sharing.

        During a session, the daemon (separate process) writes state to the file
        and the desktop app reads it via the state property. But on fresh launch,
        the state manager always resets to ASLEEP.
        """
        import tempfile
        import os
        from desktop_app import face_widget
        from desktop_app.face_widget import JarvisState

        state_file = os.path.join(tempfile.gettempdir(), "jarvis_state")

        # Get state manager and set state
        sm = face_widget.get_jarvis_state()
        sm.set_state(JarvisState.SPEAKING)

        # Verify file contains correct state (for cross-process sharing)
        with open(state_file, 'r') as f:
            content = f.read().strip()
        assert content == "speaking"

        # Verify the same instance reads updated state from file
        assert sm.state == JarvisState.SPEAKING

        # Simulate external process updating state (daemon writes to file)
        with open(state_file, 'w') as f:
            f.write("thinking")

        # Same instance should pick up change from file
        assert sm.state == JarvisState.THINKING

    @pytest.mark.unit
    def test_state_manager_handles_invalid_file_content(self):
        """State manager should handle invalid file content gracefully."""
        import tempfile
        import os
        from desktop_app import face_widget
        from desktop_app.face_widget import JarvisState

        state_file = os.path.join(tempfile.gettempdir(), "jarvis_state")

        # Create file with invalid content
        with open(state_file, 'w') as f:
            f.write("invalid_state")

        # Get state manager - should reinitialize with ASLEEP
        sm = face_widget.get_jarvis_state()

        # State should be ASLEEP (default) since file had invalid content
        assert sm.state == JarvisState.ASLEEP

        # File should be fixed
        with open(state_file, 'r') as f:
            content = f.read().strip()
        assert content == "asleep"
