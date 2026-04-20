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

from .paths import get_log_dir

GITHUB_REPO = "isair/jarvis"
UPDATER_LOG_NAME = "updater.log"
# Truncate the updater log above this size before appending a new run. Each
# run writes ~10 lines, so 1 MiB keeps hundreds of update histories without
# unbounded growth.
UPDATER_LOG_MAX_BYTES = 1024 * 1024


def _escape_applescript_path(path: Path) -> str:
    """Escape a path for use in AppleScript POSIX file strings.

    AppleScript POSIX file paths are enclosed in double quotes, so we need to
    escape backslashes and double quotes.
    """
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _escape_batch_path(path: Path) -> str:
    """Escape a path for use in Windows batch scripts.

    Batch scripts handle paths in double quotes, but certain characters
    like % need to be escaped. For safety, we reject paths with problematic
    characters since they're unusual for app installation paths.
    """
    path_str = str(path)
    # Reject paths with characters that are hard to safely escape in batch
    dangerous_chars = ['%', '!', '^', '&', '<', '>', '|']
    for char in dangerous_chars:
        if char in path_str:
            raise ValueError(f"Path contains unsafe character for batch script: {char}")
    return path_str


def _escape_shell_path(path: Path) -> str:
    """Escape a path for use in shell scripts.

    Uses single quotes which prevent all interpretation except for single quotes
    themselves, which we escape by ending the string, adding escaped quote, and
    starting a new string.
    """
    # Single quotes prevent interpretation, escape embedded single quotes
    return "'" + str(path).replace("'", "'\"'\"'") + "'"
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

    Strategy mirrors Linux: write a shell script that waits for the current
    process to exit, replaces the .app bundle with `rm -rf` + `mv`, relaunches
    via `open`, and cleans up temp. Using plain Unix file operations avoids
    the Finder/AppleScript automation prompts that were failing mid-install
    and leaving users with a trashed app and no replacement.
    """
    import plistlib
    import zipfile

    app_path = get_app_path()
    temp_dir = Path(tempfile.mkdtemp())
    current_pid = os.getpid()

    try:
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(temp_dir)

        new_app_path = temp_dir / "Jarvis.app"

        if not new_app_path.exists():
            raise FileNotFoundError("Jarvis.app not found in download")

        # Read the executable name from the new bundle's Info.plist rather
        # than hardcoding "Jarvis" — if the bundle ever renames its
        # CFBundleExecutable, the fallback relaunch still targets the right
        # binary.
        binary_name = "Jarvis"
        info_plist = new_app_path / "Contents" / "Info.plist"
        if info_plist.is_file():
            try:
                with info_plist.open("rb") as fp:
                    binary_name = plistlib.load(fp).get("CFBundleExecutable", binary_name)
            except Exception as e:
                debug_log(f"Could not read CFBundleExecutable, defaulting to {binary_name}: {e}", "updater")

        escaped_app = _escape_shell_path(app_path)
        escaped_backup = _escape_shell_path(app_path.with_suffix(app_path.suffix + ".backup"))
        escaped_new_app = _escape_shell_path(new_app_path)
        escaped_temp = _escape_shell_path(temp_dir)
        escaped_binary = _escape_shell_path(app_path / "Contents" / "MacOS" / binary_name)
        log_path = get_log_dir() / UPDATER_LOG_NAME
        escaped_log = _escape_shell_path(log_path)
        log_max = UPDATER_LOG_MAX_BYTES

        # The quarantine strip is essential for unsigned builds: without it,
        # Gatekeeper may re-prompt with "unidentified developer" on every
        # update. Keeping the previous bundle as .backup provides a one-step
        # rollback if the new version fails to launch.
        #
        # After the mv swap, LaunchServices still has the old bundle's inode
        # cached, so a bare `open` can silently no-op. `lsregister -f` forces
        # a re-scan, `open -n` forces a fresh instance, and if that still
        # fails we exec the bundle's inner binary directly. Script output is
        # appended to ~/Library/Logs/Jarvis/updater.log so future failures
        # leave a trace — the script runs detached with no terminal.
        script_path = temp_dir / "update.sh"
        script_content = f'''#!/bin/bash
LOG_FILE={escaped_log}
if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)" -gt {log_max} ]; then
    : > "$LOG_FILE"
fi
exec >> "$LOG_FILE" 2>&1
echo "=== Jarvis update $(date) ==="
echo "Waiting for process {current_pid} to exit..."
while kill -0 {current_pid} 2>/dev/null; do
    sleep 1
done
echo "Process exited, applying update..."
rm -rf {escaped_backup}
if [ -e {escaped_app} ]; then
    mv {escaped_app} {escaped_backup}
fi
mv {escaped_new_app} {escaped_app}
xattr -dr com.apple.quarantine {escaped_app} 2>/dev/null || true
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
if [ -x "$LSREGISTER" ]; then
    "$LSREGISTER" -f {escaped_app} || true
fi
echo "Relaunching..."
open -n {escaped_app}
open_rc=$?
if [ $open_rc -ne 0 ]; then
    echo "open failed (exit $open_rc), execing binary directly"
    nohup {escaped_binary} >> "$LOG_FILE" 2>&1 &
fi
rm -rf {escaped_temp}
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        subprocess.Popen([str(script_path)], start_new_session=True)

        return True

    except Exception as e:
        debug_log(f"macOS update failed: {e}", "updater")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def install_update_windows(download_path: Path) -> bool:
    """Install update on Windows.

    Strategy:
    1. Extract zip to temp location (contains Inno Setup installer as Jarvis.exe)
    2. Create batch script to:
       - Wait for current process to actually exit (by PID)
       - Run the installer silently (upgrades in place to Program Files)
       - Clean up temp directory
    3. Execute batch script and exit
    """
    import zipfile

    temp_dir = Path(tempfile.mkdtemp())
    current_pid = os.getpid()
    installed_exe_path = get_app_path()

    try:
        escaped_temp = _escape_batch_path(temp_dir)
        escaped_installed_exe = _escape_batch_path(installed_exe_path)

        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(temp_dir)

        new_exe_path = temp_dir / "Jarvis.exe"

        if not new_exe_path.exists():
            raise FileNotFoundError("Jarvis.exe not found in download")

        escaped_new_exe = _escape_batch_path(new_exe_path)

        batch_script = temp_dir / "update.bat"
        # Wait for the current process to exit by checking if PID still exists.
        # tasklist returns errorlevel 0 if process found, 1 if not found.
        # We use /SILENT (not /VERYSILENT) so Inno Setup shows its own progress
        # window during install — otherwise the user sees nothing between the
        # download dialog closing and the new app launching, which can take
        # long enough to feel like a hang. The installer's own [Run] launch
        # step is still skipped under /SILENT (skipifsilent), so we relaunch
        # the upgraded exe ourselves.
        batch_content = f'''@echo off
echo Updating Jarvis...
echo Waiting for process {current_pid} to exit...
:wait_loop
tasklist /fi "pid eq {current_pid}" 2>nul | find "{current_pid}" >nul
if not errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)
echo Process exited, running installer...
"{escaped_new_exe}" /SILENT /SUPPRESSMSGBOXES /NORESTART
echo Launching updated Jarvis...
start "" "{escaped_installed_exe}"
rmdir /s /q "{escaped_temp}"
'''
        batch_script.write_text(batch_content)

        subprocess.Popen(
            ["cmd", "/c", str(batch_script)],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        return True

    except Exception as e:
        debug_log(f"Windows update failed: {e}", "updater")
        # Clean up temp dir on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def install_update_linux(download_path: Path) -> bool:
    """Install update on Linux.

    Strategy:
    1. Extract tar.gz to temp location
    2. Create shell script to:
       - Wait for current process to actually exit (by PID)
       - Replace directory
       - Launch new app
       - Clean up temp directory
    3. Execute script and exit
    """
    import tarfile

    app_dir = get_app_path()
    temp_dir = Path(tempfile.mkdtemp())
    current_pid = os.getpid()

    try:
        with tarfile.open(download_path, "r:gz") as tf:
            tf.extractall(temp_dir)

        new_app_dir = temp_dir / "Jarvis"

        if not new_app_dir.exists():
            raise FileNotFoundError("Jarvis directory not found in download")

        # Escape paths using single quotes to prevent shell injection
        escaped_app_dir = _escape_shell_path(app_dir)
        escaped_backup = _escape_shell_path(app_dir.with_name(app_dir.name + ".backup"))
        escaped_new_app = _escape_shell_path(new_app_dir)
        escaped_temp = _escape_shell_path(temp_dir)
        escaped_jarvis = _escape_shell_path(app_dir / "Jarvis")

        script_path = temp_dir / "update.sh"
        # Keeping the previous directory as .backup gives the user a one-step
        # rollback if the new version fails to launch.
        script_content = f'''#!/bin/bash
echo "Updating Jarvis..."
echo "Waiting for process {current_pid} to exit..."
while kill -0 {current_pid} 2>/dev/null; do
    sleep 1
done
echo "Process exited, applying update..."
rm -rf {escaped_backup}
if [ -e {escaped_app_dir} ]; then
    mv {escaped_app_dir} {escaped_backup}
fi
mv {escaped_new_app} {escaped_app_dir}
{escaped_jarvis} &
rm -rf {escaped_temp}
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
