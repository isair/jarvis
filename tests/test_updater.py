"""Tests for auto-update functionality."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from pathlib import Path

from desktop_app.updater import (
    check_for_updates,
    parse_version,
    get_platform_asset_name,
    get_last_installed_asset_id,
    save_installed_asset_id,
    UpdateChannel,
    UpdateStatus,
    ReleaseInfo,
    _escape_applescript_path,
    _escape_batch_path,
    _escape_shell_path,
)


class TestParseVersion:
    """Tests for version parsing."""

    @pytest.mark.unit
    def test_parses_semver_with_v_prefix(self):
        assert parse_version("v1.2.3") == (1, 2, 3)

    @pytest.mark.unit
    def test_parses_semver_without_prefix(self):
        assert parse_version("1.2.3") == (1, 2, 3)

    @pytest.mark.unit
    def test_handles_latest_tag(self):
        assert parse_version("latest") == (0, 0, 0)

    @pytest.mark.unit
    def test_compares_patch_versions(self):
        assert parse_version("v1.2.0") < parse_version("v1.2.1")

    @pytest.mark.unit
    def test_compares_major_versions(self):
        assert parse_version("v2.0.0") > parse_version("v1.9.9")

    @pytest.mark.unit
    def test_compares_minor_versions(self):
        assert parse_version("v1.3.0") > parse_version("v1.2.9")

    @pytest.mark.unit
    def test_handles_invalid_version(self):
        assert parse_version("invalid") == (0, 0, 0)


class TestGetPlatformAssetName:
    """Tests for platform asset name detection."""

    @pytest.mark.unit
    def test_macos_arm64(self):
        with patch("sys.platform", "darwin"):
            with patch("platform.machine", return_value="arm64"):
                assert get_platform_asset_name() == "Jarvis-macOS-arm64.zip"

    @pytest.mark.unit
    def test_macos_x64(self):
        with patch("sys.platform", "darwin"):
            with patch("platform.machine", return_value="x86_64"):
                assert get_platform_asset_name() == "Jarvis-macOS-x64.zip"

    @pytest.mark.unit
    def test_windows(self):
        with patch("sys.platform", "win32"):
            assert get_platform_asset_name() == "Jarvis-Windows-x64.zip"

    @pytest.mark.unit
    def test_linux(self):
        with patch("sys.platform", "linux"):
            assert get_platform_asset_name() == "Jarvis-Linux-x64.tar.gz"


class TestCheckForUpdates:
    """Tests for update checking."""

    @pytest.mark.unit
    def test_returns_no_update_when_current_version_matches(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "v1.0.0",
                "name": "v1.0.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.0.0",
                "body": "Release notes",
                "assets": [
                    {
                        "id": 100001,
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        assert status.update_available is False
                        assert status.current_version == "1.0.0"

    @pytest.mark.unit
    def test_returns_update_when_newer_version_available(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "v1.1.0",
                "name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.1.0",
                "body": "Release notes",
                "assets": [
                    {
                        "id": 100002,
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        assert status.update_available is True
                        assert status.latest_release is not None
                        assert status.latest_release.version == "1.1.0"

    @pytest.mark.unit
    def test_skips_prereleases_for_stable_channel(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "latest",
                "name": "Latest Development Build",
                "draft": False,
                "prerelease": True,
                "html_url": "https://github.com/isair/jarvis/releases/tag/latest",
                "body": "Dev release notes",
                "assets": [
                    {
                        "id": 100003,
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        # Should not find updates because only prerelease is available
                        # and we're on stable channel
                        assert status.update_available is False

    @pytest.mark.unit
    def test_skips_drafts(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "v2.0.0",
                "name": "v2.0.0",
                "draft": True,  # Draft release
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v2.0.0",
                "body": "Release notes",
                "assets": [
                    {
                        "id": 100004,
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        # Should not find updates because only draft is available
                        assert status.update_available is False

    @pytest.mark.unit
    def test_handles_network_error(self):
        import requests

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch(
                "requests.get", side_effect=requests.RequestException("Network error")
            ):
                status = check_for_updates()
                assert status.update_available is False
                assert status.error is not None
                assert "Network error" in status.error

    @pytest.mark.unit
    def test_handles_missing_platform_asset(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "v1.1.0",
                "name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.1.0",
                "body": "Release notes",
                "assets": [
                    {
                        "id": 100005,
                        "name": "Jarvis-Windows-x64.zip",  # Only Windows asset
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):  # On macOS
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        # No macOS asset available
                        assert status.update_available is False

    @pytest.mark.unit
    def test_develop_channel_shows_update_when_no_previous_install(self):
        """Develop channel should show update when no previous install is recorded."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "latest",
                "name": "Latest Development Build",
                "draft": False,
                "prerelease": True,
                "html_url": "https://github.com/isair/jarvis/releases/tag/latest",
                "body": "Dev release notes",
                "assets": [
                    {
                        "id": 200001,
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("dev-abc1234", "develop")):
            with patch("desktop_app.updater.get_last_installed_asset_id", return_value=None):
                with patch("requests.get", return_value=mock_response):
                    with patch("sys.platform", "darwin"):
                        with patch("platform.machine", return_value="arm64"):
                            status = check_for_updates()
                            assert status.update_available is True
                            assert status.latest_release.asset_id == 200001

    @pytest.mark.unit
    def test_develop_channel_shows_update_when_asset_id_differs(self):
        """Develop channel should show update when asset ID differs from last install."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "latest",
                "name": "Latest Development Build",
                "draft": False,
                "prerelease": True,
                "html_url": "https://github.com/isair/jarvis/releases/tag/latest",
                "body": "Dev release notes",
                "assets": [
                    {
                        "id": 200002,  # New asset ID
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("dev-abc1234", "develop")):
            with patch("desktop_app.updater.get_last_installed_asset_id", return_value=200001):  # Old ID
                with patch("requests.get", return_value=mock_response):
                    with patch("sys.platform", "darwin"):
                        with patch("platform.machine", return_value="arm64"):
                            status = check_for_updates()
                            assert status.update_available is True

    @pytest.mark.unit
    def test_develop_channel_no_update_when_asset_id_matches(self):
        """Develop channel should NOT show update when asset ID matches last install."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 12345,
                "tag_name": "latest",
                "name": "Latest Development Build",
                "draft": False,
                "prerelease": True,
                "html_url": "https://github.com/isair/jarvis/releases/tag/latest",
                "body": "Dev release notes",
                "assets": [
                    {
                        "id": 200001,  # Same asset ID as last install
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("desktop_app.updater.get_version", return_value=("dev-abc1234", "develop")):
            with patch("desktop_app.updater.get_last_installed_asset_id", return_value=200001):  # Same ID
                with patch("requests.get", return_value=mock_response):
                    with patch("sys.platform", "darwin"):
                        with patch("platform.machine", return_value="arm64"):
                            status = check_for_updates()
                            assert status.update_available is False


class TestUpdateStatus:
    """Tests for UpdateStatus dataclass."""

    @pytest.mark.unit
    def test_update_status_fields(self):
        release = ReleaseInfo(
            asset_id=100001,
            tag_name="v1.0.0",
            version="1.0.0",
            name="Version 1.0.0",
            prerelease=False,
            html_url="https://example.com",
            download_url="https://example.com/download",
            asset_name="Jarvis-macOS-arm64.zip",
            asset_size=1000000,
            release_notes="Test notes",
        )
        status = UpdateStatus(
            update_available=True,
            current_version="0.9.0",
            current_channel="stable",
            latest_release=release,
        )
        assert status.update_available is True
        assert status.current_version == "0.9.0"
        assert status.latest_release.version == "1.0.0"


class TestReleaseInfo:
    """Tests for ReleaseInfo dataclass."""

    @pytest.mark.unit
    def test_release_info_fields(self):
        release = ReleaseInfo(
            asset_id=100002,
            tag_name="v1.2.3",
            version="1.2.3",
            name="Version 1.2.3",
            prerelease=False,
            html_url="https://github.com/isair/jarvis/releases/tag/v1.2.3",
            download_url="https://github.com/isair/jarvis/releases/download/v1.2.3/Jarvis.zip",
            asset_name="Jarvis-macOS-arm64.zip",
            asset_size=52428800,
            release_notes="## Changes\n- Bug fixes",
        )
        assert release.tag_name == "v1.2.3"
        assert release.version == "1.2.3"
        assert release.prerelease is False
        assert release.asset_size == 52428800
        assert release.asset_id == 100002


class TestInstallUpdateWindows:
    """Tests for Windows update installation."""

    @pytest.mark.unit
    def test_batch_script_waits_for_pid(self, tmp_path):
        """Verify the Windows batch script waits for the current process to exit."""
        import os
        import subprocess
        import zipfile
        from unittest.mock import patch, MagicMock, call

        # Create a mock zip file with Jarvis.exe
        zip_path = tmp_path / "update.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Jarvis.exe", b"mock executable content")

        # Mock get_app_path to return a fake path
        mock_app_path = tmp_path / "Jarvis.exe"
        mock_app_path.write_bytes(b"old executable")

        # Import here to avoid issues with platform checks
        from desktop_app.updater import install_update_windows

        # Capture the batch script content via the Popen call
        batch_content_captured = []

        def capture_popen(args, **kwargs):
            if args[0] == "cmd" and args[1] == "/c":
                # Read the batch script content
                batch_path = Path(args[2])
                if batch_path.exists():
                    batch_content_captured.append(batch_path.read_text())
            return MagicMock()

        with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
            # Mock CREATE_NO_WINDOW for non-Windows platforms
            if not hasattr(subprocess, 'CREATE_NO_WINDOW'):
                with patch.object(subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True):
                    with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                        result = install_update_windows(zip_path)
            else:
                with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                    result = install_update_windows(zip_path)

            assert result is True
            assert len(batch_content_captured) == 1
            batch_content = batch_content_captured[0]

            # Verify key elements of the PID-waiting batch script
            current_pid = os.getpid()
            assert f"pid eq {current_pid}" in batch_content
            assert ":wait_loop" in batch_content
            assert "goto wait_loop" in batch_content
            assert "tasklist" in batch_content
            assert "Process exited" in batch_content


class TestInstallUpdateLinux:
    """Tests for Linux update installation."""

    @pytest.mark.unit
    def test_shell_script_waits_for_pid(self, tmp_path):
        """Verify the Linux shell script waits for the current process to exit."""
        import os
        import tarfile
        from unittest.mock import patch, MagicMock

        # Create a mock tar.gz file with Jarvis directory
        tar_path = tmp_path / "update.tar.gz"
        jarvis_dir = tmp_path / "jarvis_content" / "Jarvis"
        jarvis_dir.mkdir(parents=True)
        (jarvis_dir / "Jarvis").write_bytes(b"mock executable content")

        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(jarvis_dir, arcname="Jarvis")

        # Mock get_app_path to return a fake path
        mock_app_dir = tmp_path / "installed" / "Jarvis"
        mock_app_dir.mkdir(parents=True)
        (mock_app_dir / "Jarvis").write_bytes(b"old executable")

        # Import here to avoid issues with platform checks
        from desktop_app.updater import install_update_linux

        # Capture the shell script content via the Popen call
        script_content_captured = []

        def capture_popen(args, **kwargs):
            if len(args) == 1 and args[0].endswith("update.sh"):
                # Read the shell script content
                script_path = Path(args[0])
                if script_path.exists():
                    script_content_captured.append(script_path.read_text())
            return MagicMock()

        with patch("desktop_app.updater.get_app_path", return_value=mock_app_dir):
            with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                result = install_update_linux(tar_path)

                assert result is True
                assert len(script_content_captured) == 1
                script_content = script_content_captured[0]

                # Verify key elements of the PID-waiting shell script
                current_pid = os.getpid()
                assert f"kill -0 {current_pid}" in script_content
                assert "while" in script_content
                assert "sleep 1" in script_content
                assert "Process exited" in script_content


class TestPathEscaping:
    """Tests for path escaping functions to prevent script injection."""

    @pytest.mark.unit
    def test_applescript_escapes_quotes(self):
        path = Path('/Users/test/"quoted"/app')
        escaped = _escape_applescript_path(path)
        assert '\\"' in escaped
        assert '"quoted"' not in escaped

    @pytest.mark.unit
    def test_applescript_escapes_backslashes(self):
        path = Path('/Users/test\\backslash/app')
        escaped = _escape_applescript_path(path)
        assert '\\\\' in escaped

    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix path test")
    def test_applescript_normal_path_unchanged(self):
        path = Path('/Applications/Jarvis.app')
        escaped = _escape_applescript_path(path)
        assert escaped == '/Applications/Jarvis.app'

    @pytest.mark.unit
    def test_batch_rejects_percent_sign(self):
        path = Path('C:\\Users\\test%USERPROFILE%\\app')
        with pytest.raises(ValueError, match="unsafe character"):
            _escape_batch_path(path)

    @pytest.mark.unit
    def test_batch_rejects_ampersand(self):
        path = Path('C:\\Users\\test&echo bad\\app')
        with pytest.raises(ValueError, match="unsafe character"):
            _escape_batch_path(path)

    @pytest.mark.unit
    def test_batch_rejects_pipe(self):
        path = Path('C:\\Users\\test|dir\\app')
        with pytest.raises(ValueError, match="unsafe character"):
            _escape_batch_path(path)

    @pytest.mark.unit
    def test_batch_normal_path_unchanged(self):
        path = Path('C:\\Program Files\\Jarvis\\Jarvis.exe')
        escaped = _escape_batch_path(path)
        assert escaped == 'C:\\Program Files\\Jarvis\\Jarvis.exe'

    @pytest.mark.unit
    def test_shell_escapes_single_quotes(self):
        path = Path("/Users/test's folder/app")
        escaped = _escape_shell_path(path)
        # Single quotes should be escaped by ending quote, adding escaped quote, starting new quote
        assert "'" in escaped
        assert escaped.startswith("'")
        assert escaped.endswith("'")

    @pytest.mark.unit
    def test_shell_handles_special_chars(self):
        path = Path('/Users/test $HOME `whoami`/app')
        escaped = _escape_shell_path(path)
        # In single quotes, $ and backticks are literal
        assert escaped.startswith("'")
        assert escaped.endswith("'")
        # The content should be preserved (not interpreted)
        assert '$HOME' in escaped
        assert '`whoami`' in escaped

    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix path test")
    def test_shell_normal_path_wrapped(self):
        path = Path('/opt/Jarvis/Jarvis')
        escaped = _escape_shell_path(path)
        assert escaped == "'/opt/Jarvis/Jarvis'"
