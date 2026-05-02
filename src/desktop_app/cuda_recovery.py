"""Recovery action for the GPU acceleration libraries on Windows.

The Inno Setup installer ships a PowerShell script (`install_cuda.ps1`) that
downloads cuBLAS and cuDNN into `{app}\\cuda`. That step runs once during
install and may fail silently — slow connections truncate the 643 MB cuDNN
wheel, AV quarantines the unsigned engines DLL, the user dismisses a UAC
prompt. When that happens the runtime probe in `jarvis.listening.listener`
falls back to CPU and the only documented fix used to be "reinstall the app",
which doesn't help because the `.cuda_installed` marker tricks the installer
into skipping the CUDA step.

This module exposes a tray menu action that re-runs the installer script
directly, with UAC elevation, so users can recover without touching the
installer at all.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CudaRecoveryAction:
    label: str
    script_path: Path
    target_dir: Path
    executable: str
    arguments: list[str]


@functools.lru_cache(maxsize=None)
def _has_nvidia_driver() -> bool:
    """Match the Inno Setup HasNvidiaGPU check: nvcuda.dll in System32.

    Cached because drivers don't appear or disappear during a process run.
    """
    if sys.platform != "win32":
        return False
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return Path(system_root, "System32", "nvcuda.dll").exists()


def _powershell_executable() -> str:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return str(
        Path(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    )


def cuda_recovery_action(install_root: Path) -> Optional[CudaRecoveryAction]:
    """Return a recovery action if the host platform supports it.

    `install_root` is the directory containing `install_cuda.ps1` (in
    bundled mode this is the directory next to the frozen executable).
    Returns `None` when:

    - The platform isn't Windows.
    - No NVIDIA driver is detected (nothing to recover to).
    - The installer-bundled script is missing (dev runs from source).
    """
    if sys.platform != "win32":
        return None
    if not _has_nvidia_driver():
        return None

    script_path = Path(install_root) / "install_cuda.ps1"
    if not script_path.exists():
        return None

    target_dir = Path(install_root) / "cuda"
    log_path = target_dir / "install.log"

    arguments = [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-TargetDir",
        str(target_dir),
        "-LogPath",
        str(log_path),
    ]

    return CudaRecoveryAction(
        label="🎮 Reinstall GPU libraries",
        script_path=script_path,
        target_dir=target_dir,
        executable=_powershell_executable(),
        arguments=arguments,
    )


def _shell_execute(hwnd: int, verb: str, file: str, params: str, directory: str, show: int) -> int:
    """Thin wrapper over ShellExecuteW so tests can patch it without dragging in ctypes."""
    import ctypes

    return int(
        ctypes.windll.shell32.ShellExecuteW(hwnd, verb, file, params, directory, show)
    )


def _quote_arg(arg: str) -> str:
    """Quote a single argument for ShellExecuteW's lpParameters string.

    Windows argv parsing (CommandLineToArgvW) treats a backslash run only as
    an escape when it precedes a quote: 2n backslashes + " emits n
    backslashes and ends the quoted string; 2n+1 emits n + a literal ".
    A trailing backslash inside `"..."` therefore swallows the closing
    quote unless it is doubled. Doubling every trailing backslash is the
    canonical fix and is what argv parsers expect.
    """
    if not arg:
        return '""'
    if not any(ch in arg for ch in (" ", "\t", '"')):
        return arg

    out: list[str] = ['"']
    i = 0
    while i < len(arg):
        bs = 0
        while i < len(arg) and arg[i] == "\\":
            bs += 1
            i += 1
        if i == len(arg):
            out.append("\\" * (bs * 2))
            break
        if arg[i] == '"':
            out.append("\\" * (bs * 2 + 1))
            out.append('"')
        else:
            out.append("\\" * bs)
            out.append(arg[i])
        i += 1
    out.append('"')
    return "".join(out)


def run_action(action: CudaRecoveryAction) -> bool:
    """Launch the recovery script with UAC elevation.

    `install_cuda.ps1` writes into `Program Files\\Jarvis\\cuda`, which a
    standard user account cannot write to. ShellExecuteW with the `runas`
    verb triggers the UAC prompt; without it the script silently fails
    its first file write and the user is no better off than before.
    """
    if sys.platform != "win32":
        return False

    params = " ".join(_quote_arg(a) for a in action.arguments)
    rc = _shell_execute(0, "runas", action.executable, params, str(action.target_dir.parent), 1)
    # ShellExecuteW returns >32 on success; <=32 means an error code (e.g.
    # SE_ERR_ACCESSDENIED 5 when the user dismisses the UAC prompt).
    return rc > 32
