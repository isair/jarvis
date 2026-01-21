"""
Auto-update functionality for Jarvis Desktop App.

Checks GitHub Releases for new versions and handles the update process.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from jarvis import get_version
from jarvis.debug import debug_log

GITHUB_REPO = "isair/jarvis"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"


def _get_update_state_path() -> Path:
    """Get path to update state file."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        config_dir = Path(xdg) / "jarvis"
    else:
        config_dir = Path.home() / ".config" / "jarvis"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "update_state.json"


def get_last_installed_asset_id() -> Optional[int]:
    """Get the asset ID of the last installed update.

    We track the asset ID rather than release ID because for the "latest"
    prerelease tag, the release ID stays the same when updated, but each
    uploaded asset gets a new unique ID.
    """
    try:
        state_path = _get_update_state_path()
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_installed_asset_id")
    except Exception as e:
        debug_log(f"Failed to read update state: {e}", "updater")
    return None


def save_installed_asset_id(asset_id: int) -> None:
    """Save the asset ID after a successful update."""
    try:
        state_path = _get_update_state_path()
        data = {}
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data["last_installed_asset_id"] = asset_id
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        debug_log(f"Saved installed asset ID: {asset_id}", "updater")
    except Exception as e:
        debug_log(f"Failed to save update state: {e}", "updater")


class UpdateChannel(Enum):
    """Update channel for the application."""

    STABLE = "stable"
    DEVELOP = "develop"


@dataclass
class ReleaseInfo:
    """Information about a GitHub release."""

    asset_id: int  # Unique GitHub asset ID for tracking updates (changes on each upload)
    tag_name: str
    version: str
    name: str
    prerelease: bool
    html_url: str
    download_url: str
    asset_name: str
    asset_size: int
    release_notes: str


@dataclass
class UpdateStatus:
    """Result of checking for updates."""

    update_available: bool
    current_version: str
    current_channel: str
    latest_release: Optional[ReleaseInfo]
    error: Optional[str] = None


def get_platform_asset_name() -> str:
    """Get the expected asset name for the current platform."""
    if sys.platform == "darwin":
        arch = platform.machine()
        if arch == "arm64":
            return "Jarvis-macOS-arm64.zip"
        return "Jarvis-macOS-x64.zip"
    elif sys.platform == "win32":
        return "Jarvis-Windows-x64.zip"
    else:
        return "Jarvis-Linux-x64.tar.gz"


def parse_version(tag: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison.

    Handles both 'v1.2.3' and 'latest' (develop) formats.
    """
    if tag == "latest":
        return (0, 0, 0)

    version_str = tag.lstrip("v")

    try:
        parts = version_str.split(".")
        return tuple(int(p) for p in parts)
    except ValueError:
        return (0, 0, 0)


def check_for_updates(channel: Optional[UpdateChannel] = None) -> UpdateStatus:
    """Check GitHub Releases for available updates.

    Args:
        channel: Update channel to check. If None, uses current app's channel.

    Returns:
        UpdateStatus with update information.
    """
    current_version, current_channel = get_version()

    if channel is None:
        channel = (
            UpdateChannel.DEVELOP
            if current_channel == "develop"
            else UpdateChannel.STABLE
        )

    try:
        response = requests.get(
            GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        response.raise_for_status()
        releases = response.json()

        target_release = None
        platform_asset_name = get_platform_asset_name()

        for release in releases:
            if release.get("draft", False):
                continue

            if channel == UpdateChannel.STABLE and release.get("prerelease", False):
                continue

            if channel == UpdateChannel.DEVELOP:
                if release.get("tag_name") != "latest":
                    continue

            assets = release.get("assets", [])
            for asset in assets:
                if asset["name"] == platform_asset_name:
                    target_release = ReleaseInfo(
                        asset_id=asset["id"],
                        tag_name=release["tag_name"],
                        version=release["tag_name"].lstrip("v"),
                        name=release.get("name", release["tag_name"]),
                        prerelease=release.get("prerelease", False),
                        html_url=release["html_url"],
                        download_url=asset["browser_download_url"],
                        asset_name=asset["name"],
                        asset_size=asset["size"],
                        release_notes=release.get("body", ""),
                    )
                    break

            if target_release:
                break

        if not target_release:
            return UpdateStatus(
                update_available=False,
                current_version=current_version,
                current_channel=current_channel,
                latest_release=None,
            )

        if channel == UpdateChannel.DEVELOP:
            # For develop channel, compare asset IDs to detect new builds
            # (release ID stays the same when "latest" is updated, but asset IDs change)
            last_installed_id = get_last_installed_asset_id()
            update_available = (
                last_installed_id is None
                or target_release.asset_id != last_installed_id
            )
        else:
            current_tuple = parse_version(current_version)
            latest_tuple = parse_version(target_release.tag_name)
            update_available = latest_tuple > current_tuple

        return UpdateStatus(
            update_available=update_available,
            current_version=current_version,
            current_channel=current_channel,
            latest_release=target_release,
        )

    except requests.RequestException as e:
        debug_log(f"Failed to check for updates: {e}", "updater")
        return UpdateStatus(
            update_available=False,
            current_version=current_version,
            current_channel=current_channel,
            latest_release=None,
            error=str(e),
        )


class DownloadSignals(QObject):
    """Signals for download progress updates."""

    progress = pyqtSignal(int, int)  # downloaded_bytes, total_bytes
    completed = pyqtSignal(str)  # path to downloaded file
    error = pyqtSignal(str)  # error message


class DownloadWorker(QThread):
    """Background worker for downloading updates."""

    def __init__(self, url: str, dest_path: Path, signals: DownloadSignals):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
        self.signals = signals
        self._cancelled = False

    def run(self):
        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(self.dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self._cancelled:
                        return
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.signals.progress.emit(downloaded, total_size)

            self.signals.completed.emit(str(self.dest_path))

        except Exception as e:
            self.signals.error.emit(str(e))

    def cancel(self):
        self._cancelled = True


def get_app_path() -> Path:
    """Get the path to the current application."""
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            # Jarvis.app/Contents/MacOS/Jarvis -> Jarvis.app
            return Path(sys.executable).parent.parent.parent
        elif sys.platform == "win32":
            return Path(sys.executable)
        else:
            return Path(sys.executable).parent
    else:
        raise RuntimeError("Cannot update when running from source")


def is_frozen() -> bool:
    """Check if running as a bundled/frozen application."""
    return getattr(sys, "frozen", False)


def install_update_macos(download_path: Path) -> bool:
    """Install update on macOS.

    Strategy:
    1. Extract zip to temp location
    2. Move current app to trash
    3. Move new app to original location
    4. Launch new app
    5. Exit current app
    """
    import zipfile

    app_path = get_app_path()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(temp_dir)

        new_app_path = temp_dir / "Jarvis.app"

        if not new_app_path.exists():
            raise FileNotFoundError("Jarvis.app not found in download")

        # Use AppleScript to move to trash and replace
        script = f'''
        tell application "Finder"
            move POSIX file "{app_path}" to trash
            move POSIX file "{new_app_path}" to folder "{app_path.parent}"
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True)

        # Launch new app
        subprocess.Popen(["open", str(app_path.parent / "Jarvis.app")])

        return True

    except Exception as e:
        debug_log(f"macOS update failed: {e}", "updater")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def install_update_windows(download_path: Path) -> bool:
    """Install update on Windows.

    Strategy:
    1. Extract zip to temp location
    2. Create batch script to:
       - Wait for current process to exit
       - Replace executable
       - Launch new app
    3. Execute batch script and exit
    """
    import zipfile

    app_path = get_app_path()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(temp_dir)

        new_exe_path = temp_dir / "Jarvis.exe"

        if not new_exe_path.exists():
            raise FileNotFoundError("Jarvis.exe not found in download")

        batch_script = temp_dir / "update.bat"
        batch_content = f'''@echo off
echo Updating Jarvis...
timeout /t 2 /nobreak > nul
del /f "{app_path}"
move /y "{new_exe_path}" "{app_path}"
start "" "{app_path}"
del "%~f0"
'''
        batch_script.write_text(batch_content)

        subprocess.Popen(
            ["cmd", "/c", str(batch_script)],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        return True

    except Exception as e:
        debug_log(f"Windows update failed: {e}", "updater")
        return False


def install_update_linux(download_path: Path) -> bool:
    """Install update on Linux.

    Strategy:
    1. Extract tar.gz to temp location
    2. Create shell script to:
       - Wait for current process to exit
       - Replace directory
       - Launch new app
    3. Execute script and exit
    """
    import tarfile

    app_dir = get_app_path()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        with tarfile.open(download_path, "r:gz") as tf:
            tf.extractall(temp_dir)

        new_app_dir = temp_dir / "Jarvis"

        if not new_app_dir.exists():
            raise FileNotFoundError("Jarvis directory not found in download")

        script_path = temp_dir / "update.sh"
        script_content = f'''#!/bin/bash
sleep 2
rm -rf "{app_dir}"
mv "{new_app_dir}" "{app_dir}"
"{app_dir}/Jarvis" &
rm -f "$0"
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        subprocess.Popen([str(script_path)], start_new_session=True)

        return True

    except Exception as e:
        debug_log(f"Linux update failed: {e}", "updater")
        return False


def install_update(download_path: Path) -> bool:
    """Install update for current platform."""
    if sys.platform == "darwin":
        return install_update_macos(download_path)
    elif sys.platform == "win32":
        return install_update_windows(download_path)
    else:
        return install_update_linux(download_path)
