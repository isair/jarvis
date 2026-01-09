"""Tests for auto-update functionality."""

import pytest
from unittest.mock import patch, MagicMock

from jarvis.updater import (
    check_for_updates,
    parse_version,
    get_platform_asset_name,
    UpdateChannel,
    UpdateStatus,
    ReleaseInfo,
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
                "tag_name": "v1.0.0",
                "name": "v1.0.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.0.0",
                "body": "Release notes",
                "assets": [
                    {
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
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
                "tag_name": "v1.1.0",
                "name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.1.0",
                "body": "Release notes",
                "assets": [
                    {
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
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
                "tag_name": "latest",
                "name": "Latest Development Build",
                "draft": False,
                "prerelease": True,
                "html_url": "https://github.com/isair/jarvis/releases/tag/latest",
                "body": "Dev release notes",
                "assets": [
                    {
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
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
                "tag_name": "v2.0.0",
                "name": "v2.0.0",
                "draft": True,  # Draft release
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v2.0.0",
                "body": "Release notes",
                "assets": [
                    {
                        "name": "Jarvis-macOS-arm64.zip",
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        # Should not find updates because only draft is available
                        assert status.update_available is False

    @pytest.mark.unit
    def test_handles_network_error(self):
        import requests

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
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
                "tag_name": "v1.1.0",
                "name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "html_url": "https://github.com/isair/jarvis/releases/tag/v1.1.0",
                "body": "Release notes",
                "assets": [
                    {
                        "name": "Jarvis-Windows-x64.zip",  # Only Windows asset
                        "browser_download_url": "https://example.com/download",
                        "size": 1000,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("jarvis.updater.get_version", return_value=("1.0.0", "stable")):
            with patch("requests.get", return_value=mock_response):
                with patch("sys.platform", "darwin"):  # On macOS
                    with patch("platform.machine", return_value="arm64"):
                        status = check_for_updates()
                        # No macOS asset available
                        assert status.update_available is False


class TestUpdateStatus:
    """Tests for UpdateStatus dataclass."""

    @pytest.mark.unit
    def test_update_status_fields(self):
        release = ReleaseInfo(
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
