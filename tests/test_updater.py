"""Tests for auto-update functionality."""

import os
import subprocess
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


def _zipfile_extract_for_tests(zip_path: Path, dest_dir: Path) -> None:
    """Stand-in for ``_extract_macos_bundle`` used by existing unit tests.

    Production code uses ``ditto`` (a subprocess call), but tests mock
    ``subprocess.Popen`` which also breaks ``subprocess.run``. Swapping in a
    direct zipfile extraction lets the existing tests run their assertions
    on the generated shell script without the ditto invocation.
    """
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


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

            # Verify the installer is run silently (not the old move/replace approach).
            # We use /SILENT rather than /VERYSILENT so Inno Setup shows its own
            # progress window during install — otherwise the user sees nothing
            # between the download dialog closing and the new app launching.
            assert "/SILENT" in batch_content
            assert "/VERYSILENT" not in batch_content
            assert "/SUPPRESSMSGBOXES" in batch_content
            assert "move /y" not in batch_content

    @pytest.mark.unit
    def test_batch_script_launches_updated_exe(self, tmp_path):
        """After silent install, the batch script must relaunch the upgraded exe.

        Inno Setup's postinstall launch is skipped under /VERYSILENT, so the
        updater itself has to start the new version — otherwise the user is
        left with a stopped app after a successful update.
        """
        import subprocess
        import zipfile
        from unittest.mock import patch, MagicMock

        zip_path = tmp_path / "update.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Jarvis.exe", b"mock executable content")

        mock_app_path = tmp_path / "Program Files" / "Jarvis" / "Jarvis.exe"
        mock_app_path.parent.mkdir(parents=True)
        mock_app_path.write_bytes(b"old executable")

        from desktop_app.updater import install_update_windows

        batch_content_captured = []

        def capture_popen(args, **kwargs):
            if args[0] == "cmd" and args[1] == "/c":
                batch_path = Path(args[2])
                if batch_path.exists():
                    batch_content_captured.append(batch_path.read_text())
            return MagicMock()

        with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
            if not hasattr(subprocess, 'CREATE_NO_WINDOW'):
                with patch.object(subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True):
                    with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                        install_update_windows(zip_path)
            else:
                with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                    install_update_windows(zip_path)

        assert len(batch_content_captured) == 1
        batch_content = batch_content_captured[0]

        # The launch must come after the installer line so the new binary is
        # in place when it runs.
        installer_idx = batch_content.find("/SILENT")
        launch_idx = batch_content.find(f'start "" "{mock_app_path}"')
        assert installer_idx != -1, "installer line missing"
        assert launch_idx != -1, "start line for upgraded exe missing"
        assert launch_idx > installer_idx, "launch must follow install"


class TestInstallUpdateMacos:
    """Tests for macOS update installation."""

    @pytest.mark.unit
    def test_shell_script_waits_for_pid_and_relaunches(self, tmp_path):
        """macOS installer must wait for the current PID to exit, replace the
        bundle with plain file ops (no Finder automation), and relaunch.

        The previous AppleScript/Finder approach was failing mid-install on
        some machines — it would trash the old app, prompt for file-editing
        permission, then error out, leaving the user with no app. The shell
        script approach matches Linux and avoids Finder entirely.
        """
        import os
        import zipfile
        from unittest.mock import patch, MagicMock

        zip_path = tmp_path / "update.zip"
        app_source = tmp_path / "zip_content" / "Jarvis.app"
        app_source.mkdir(parents=True)
        (app_source / "Contents").mkdir()
        (app_source / "Contents" / "Info.plist").write_bytes(b"mock plist")

        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in app_source.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(tmp_path / "zip_content")))

        mock_app_path = tmp_path / "Applications" / "Jarvis.app"
        mock_app_path.mkdir(parents=True)
        (mock_app_path / "existing").write_bytes(b"old bundle")

        from desktop_app.updater import install_update_macos

        script_content_captured = []

        def capture_popen(args, **kwargs):
            if len(args) == 1 and args[0].endswith("update.sh"):
                script_path = Path(args[0])
                if script_path.exists():
                    script_content_captured.append(script_path.read_text())
            return MagicMock()

        with patch("desktop_app.updater._extract_macos_bundle", side_effect=_zipfile_extract_for_tests):
            with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
                with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                    result = install_update_macos(zip_path)

        assert result is True
        assert len(script_content_captured) == 1
        script_content = script_content_captured[0]

        current_pid = os.getpid()
        assert f"kill -0 {current_pid}" in script_content
        assert "sleep 1" in script_content
        # No Finder automation
        assert "osascript" not in script_content
        assert "Finder" not in script_content
        # Bundle is replaced and relaunched
        assert "mv " in script_content
        assert "open " in script_content

        # Previous bundle is preserved as a .backup for rollback, not deleted.
        # This is important: if the new version fails to launch, the user can
        # restore the backup manually.
        backup_path = str(mock_app_path) + ".backup"
        assert backup_path in script_content
        assert f"mv '{mock_app_path}' '{backup_path}'" in script_content
        # The old .backup from the previous update is cleared first.
        assert f"rm -rf '{backup_path}'" in script_content

        # Quarantine xattr is stripped so Gatekeeper doesn't re-prompt on every
        # update for unsigned builds.
        assert "xattr -dr com.apple.quarantine" in script_content

        clear_backup_idx = script_content.find(f"rm -rf '{backup_path}'")
        move_to_backup_idx = script_content.find(f"mv '{mock_app_path}' '{backup_path}'")
        install_idx = script_content.find(f"mv '") # first mv is to backup, find install
        xattr_idx = script_content.find("xattr -dr com.apple.quarantine")
        open_idx = script_content.find("open ")
        assert clear_backup_idx < move_to_backup_idx, "must clear old backup before creating new one"
        assert move_to_backup_idx < xattr_idx, "backup happens before xattr strip"
        assert xattr_idx < open_idx, "xattr strip must precede launch"

        # LaunchServices caches the old bundle inode across the mv swap, so a
        # bare `open` silently no-ops. Re-register the bundle and force a new
        # instance, and fall back to execing the inner binary if `open` fails
        # — otherwise the update "installs" but never relaunches.
        from desktop_app.updater import UPDATER_LOG_NAME
        from desktop_app.paths import get_log_dir
        assert "lsregister" in script_content
        assert "open -n" in script_content
        binary_path = str(mock_app_path / "Contents" / "MacOS" / "Jarvis")
        assert binary_path in script_content, "fallback must exec the bundle's inner binary"
        lsregister_idx = script_content.find("lsregister")
        assert xattr_idx < lsregister_idx < open_idx, "lsregister must run after xattr and before open"

        # Script output must be captured to a log file — otherwise detached
        # failures leave no trace and we can't diagnose future relaunch bugs.
        expected_log_path = str(get_log_dir() / UPDATER_LOG_NAME)
        assert expected_log_path in script_content

    @pytest.mark.unit
    def test_binary_name_read_from_bundle_info_plist(self, tmp_path):
        """The fallback exec must target the actual CFBundleExecutable, not a
        hardcoded "Jarvis" — so a future bundle rename doesn't silently break
        the fallback relaunch."""
        import plistlib
        import zipfile
        from unittest.mock import patch, MagicMock

        custom_binary_name = "JarvisNext"
        zip_path = tmp_path / "update.zip"
        app_source = tmp_path / "zip_content" / "Jarvis.app"
        (app_source / "Contents").mkdir(parents=True)
        plist_bytes = plistlib.dumps({"CFBundleExecutable": custom_binary_name})
        (app_source / "Contents" / "Info.plist").write_bytes(plist_bytes)

        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in app_source.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(tmp_path / "zip_content")))

        mock_app_path = tmp_path / "Applications" / "Jarvis.app"
        mock_app_path.mkdir(parents=True)

        from desktop_app.updater import install_update_macos

        script_content_captured = []

        def capture_popen(args, **kwargs):
            if len(args) == 1 and args[0].endswith("update.sh"):
                script_content_captured.append(Path(args[0]).read_text())
            return MagicMock()

        with patch("desktop_app.updater._extract_macos_bundle", side_effect=_zipfile_extract_for_tests):
            with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
                with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                    assert install_update_macos(zip_path) is True

        script_content = script_content_captured[0]
        expected_binary = str(mock_app_path / "Contents" / "MacOS" / custom_binary_name)
        assert expected_binary in script_content, (
            "fallback exec must use CFBundleExecutable from the new bundle"
        )
        # Shell-quoted; a bare 'Jarvis' occurrence would end with a single
        # quote, whereas 'JarvisNext' does not.
        hardcoded_binary = f"{mock_app_path / 'Contents' / 'MacOS' / 'Jarvis'}'"
        assert hardcoded_binary not in script_content, (
            "must not fall back to hardcoded 'Jarvis' when the bundle reports a different name"
        )

    @pytest.mark.unit
    def test_shell_script_fallback_execs_binary_when_open_fails(self, tmp_path):
        """When `open -n` fails (the real-world failure mode we're fixing),
        the generated script must actually exec the bundle's inner binary.
        Structural assertions that the text is present are not enough —
        quoting bugs or `$?` semantics could break the runtime path.

        This test executes the generated script in a sandbox where `open` is
        stubbed to exit non-zero, and asserts the fallback binary runs.
        """
        import plistlib
        import re
        import time
        import zipfile
        from unittest.mock import patch, MagicMock

        zip_path = tmp_path / "update.zip"
        app_source = tmp_path / "zip_content" / "Jarvis.app"
        (app_source / "Contents" / "MacOS").mkdir(parents=True)
        (app_source / "Contents" / "Info.plist").write_bytes(
            plistlib.dumps({"CFBundleExecutable": "Jarvis"})
        )
        # The fallback execs Contents/MacOS/<binary_name>; stub it with a
        # shell script that writes a marker file we can check for.
        marker_path = tmp_path / "fallback_fired.marker"
        stub_binary = app_source / "Contents" / "MacOS" / "Jarvis"
        stub_binary.write_text(f'#!/bin/bash\necho fired > {marker_path}\n')
        stub_binary.chmod(0o755)

        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in app_source.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(tmp_path / "zip_content")))

        mock_app_path = tmp_path / "Applications" / "Jarvis.app"
        mock_app_path.mkdir(parents=True)

        # PATH-shadowed stubs: `open` always fails, `xattr` no-ops. The real
        # /System lsregister path won't exist in tests, so the script's
        # `if [ -x "$LSREGISTER" ]` guard skips it cleanly.
        stub_dir = tmp_path / "path_stubs"
        stub_dir.mkdir()
        (stub_dir / "open").write_text("#!/bin/bash\nexit 1\n")
        (stub_dir / "open").chmod(0o755)
        (stub_dir / "xattr").write_text("#!/bin/bash\nexit 0\n")
        (stub_dir / "xattr").chmod(0o755)

        from desktop_app.updater import install_update_macos

        captured = {}

        def capture_popen(args, **kwargs):
            if len(args) == 1 and args[0].endswith("update.sh"):
                captured["script"] = Path(args[0])
                captured["text"] = captured["script"].read_text()
            return MagicMock()

        with patch("desktop_app.updater._extract_macos_bundle", side_effect=_zipfile_extract_for_tests):
            with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
                with patch("desktop_app.updater.subprocess.Popen", side_effect=capture_popen):
                    assert install_update_macos(zip_path) is True

        # Python's zipfile.extractall doesn't restore the Unix exec bit, so
        # the stub binary inside the extracted new bundle comes out without
        # +x — the nohup fallback would then fail with EACCES, which would
        # hide real exec failures behind a test-infrastructure bug. Walk the
        # new bundle (located from the `mv <new>` line in the script) and
        # restore the exec bit before running.
        new_app_match = re.search(r"mv '([^']+\.app)' '" + re.escape(str(mock_app_path)) + "'",
                                  captured["text"])
        assert new_app_match, "could not find extracted new_app path in script"
        new_binary = Path(new_app_match.group(1)) / "Contents" / "MacOS" / "Jarvis"
        new_binary.chmod(0o755)

        # Strip the PID-wait loop so the test doesn't hang on the parent PID,
        # and swap the log redirect for stdout so any script errors surface in
        # the pytest output rather than being hidden.
        script_text = captured["text"]
        script_text = re.sub(
            r"while kill -0 \d+ 2>/dev/null; do\s*\n\s*sleep 1\s*\ndone",
            ":",
            script_text,
        )
        script_text = re.sub(r'^exec >> .*$', 'true', script_text, count=1, flags=re.MULTILINE)
        # Drop the log-rotation preamble — it references the same log file
        # we've just neutered.
        script_text = re.sub(
            r'LOG_FILE=.*?\nif \[ -f "\$LOG_FILE".*?fi\n',
            '',
            script_text,
            count=1,
            flags=re.DOTALL,
        )
        # Fallback nohup also redirects to $LOG_FILE; neutralise it.
        script_text = script_text.replace('>> "$LOG_FILE" 2>&1', '>/dev/null 2>&1')
        runnable = tmp_path / "run.sh"
        runnable.write_text(script_text)
        runnable.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
        result = subprocess.run(
            ["bash", str(runnable)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"script failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )

        # The fallback is backgrounded via nohup, give it a moment to run.
        for _ in range(20):
            if marker_path.exists():
                break
            time.sleep(0.1)

        assert marker_path.exists(), (
            "fallback binary did not execute when `open` failed — "
            "the user would be left without a running app after update"
        )


    @pytest.mark.unit
    def test_uses_ditto_to_preserve_bundle_symlinks(self, tmp_path):
        """PyInstaller's Qt bundle contains symlinks (framework
        Versions/Current, etc.) that Python's zipfile silently flattens into
        regular files — the extracted bundle then fails to launch with
        "Jarvis.app can't be opened". The updater must extract with
        `/usr/bin/ditto` when it is available, not zipfile."""
        import plistlib
        import zipfile
        from unittest.mock import patch, MagicMock

        zip_path = tmp_path / "update.zip"
        app_source = tmp_path / "zip_content" / "Jarvis.app"
        (app_source / "Contents").mkdir(parents=True)
        (app_source / "Contents" / "Info.plist").write_bytes(
            plistlib.dumps({"CFBundleExecutable": "Jarvis"})
        )
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in app_source.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(tmp_path / "zip_content")))

        mock_app_path = tmp_path / "Applications" / "Jarvis.app"
        mock_app_path.mkdir(parents=True)

        # Stand in for /usr/bin/ditto with a real file that the updater's
        # existence check will see; subprocess.run is mocked so we never
        # actually execute it. The fake "runs" the command by extracting
        # the zip so the rest of the installer sees the expected bundle.
        fake_ditto = tmp_path / "fake_ditto"
        fake_ditto.write_text("")

        run_calls = []

        def fake_run(args, **kwargs):
            run_calls.append(args)
            if isinstance(args, list) and len(args) >= 4 and args[0] == str(fake_ditto):
                dest = Path(args[-1])
                with zipfile.ZipFile(args[-2], "r") as zf:
                    zf.extractall(dest)
            return MagicMock(returncode=0)

        from desktop_app.updater import install_update_macos

        with patch("desktop_app.updater.DITTO_PATH", str(fake_ditto)):
            with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
                with patch("desktop_app.updater.subprocess.run", side_effect=fake_run):
                    with patch("desktop_app.updater.subprocess.Popen", return_value=MagicMock()):
                        assert install_update_macos(zip_path) is True

        ditto_calls = [c for c in run_calls if isinstance(c, list) and c and c[0] == str(fake_ditto)]
        assert ditto_calls, (
            "updater must invoke ditto to extract the macOS bundle — "
            "Python's zipfile drops symlinks and produces an unlaunchable bundle"
        )
        assert ditto_calls[0][1:3] == ["-x", "-k"], (
            f"expected `ditto -x -k <src> <dest>`, got {ditto_calls[0]}"
        )

    @pytest.mark.unit
    def test_falls_back_to_zipfile_when_ditto_missing(self, tmp_path):
        """When ditto is absent (non-macOS CI), extraction must fall back to
        zipfile rather than raising FileNotFoundError. Non-macOS hosts never
        hit this path in production, but the safety net keeps the unit suite
        runnable off-macOS — regressing that would silently break CI."""
        import zipfile
        from desktop_app.updater import _extract_macos_bundle

        zip_path = tmp_path / "bundle.zip"
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        (payload_dir / "hello.txt").write_text("hi")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(payload_dir / "hello.txt", arcname="hello.txt")

        dest = tmp_path / "dest"
        dest.mkdir()

        missing_ditto = tmp_path / "does_not_exist"
        assert not missing_ditto.exists()

        with patch("desktop_app.updater.DITTO_PATH", str(missing_ditto)):
            _extract_macos_bundle(zip_path, dest)

        assert (dest / "hello.txt").read_text() == "hi", (
            "fallback must still extract the zip when ditto is unavailable"
        )

    @pytest.mark.unit
    def test_ditto_extraction_failure_surfaces_as_install_failure(self, tmp_path):
        """If ditto exits non-zero, install_update_macos must catch the
        CalledProcessError and return False so the UI shows the generic
        update-failed dialog — never crash the app or leave a half-applied
        bundle behind."""
        import zipfile

        zip_path = tmp_path / "update.zip"
        app_source = tmp_path / "zip_content" / "Jarvis.app" / "Contents"
        app_source.mkdir(parents=True)
        (app_source / "Info.plist").write_bytes(b"mock")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in (tmp_path / "zip_content").rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(tmp_path / "zip_content")))

        mock_app_path = tmp_path / "Applications" / "Jarvis.app"
        mock_app_path.mkdir(parents=True)

        fake_ditto = tmp_path / "fake_ditto"
        fake_ditto.write_text("")

        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args)

        from desktop_app.updater import install_update_macos

        with patch("desktop_app.updater.DITTO_PATH", str(fake_ditto)):
            with patch("desktop_app.updater.get_app_path", return_value=mock_app_path):
                with patch("desktop_app.updater.subprocess.run", side_effect=fake_run):
                    with patch("desktop_app.updater.subprocess.Popen", return_value=MagicMock()) as popen:
                        result = install_update_macos(zip_path)

        assert result is False, "ditto failure must surface as install-failed"
        assert not popen.called, (
            "must not launch the relaunch script after extraction failed"
        )


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

                # Previous directory is kept as .backup for rollback.
                backup_path = str(mock_app_dir) + ".backup"
                assert backup_path in script_content
                assert f"mv '{mock_app_dir}' '{backup_path}'" in script_content


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
