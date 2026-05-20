"""Install and run the lharries WhatsApp Go bridge with Jarvis-friendly QR output."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Optional
from urllib.request import urlopen

from jarvis.config import default_config_path
from jarvis.debug import debug_log

REPO_ZIP_URL = "https://github.com/lharries/whatsapp-mcp/archive/refs/heads/main.zip"
PATCH_MARKER = "JARVIS_QR_CODE"
_QR_LINE = re.compile(r"^JARVIS_QR_CODE:(.+)$")
_AUTH_OK = "JARVIS_AUTH_OK"


def whatsapp_install_dir() -> Path:
    return default_config_path().parent / "whatsapp-mcp"


def bridge_dir(install: Path | None = None) -> Path:
    root = install or whatsapp_install_dir()
    return root / "whatsapp-mcp-main" / "whatsapp-bridge"


def mcp_server_dir(install: Path | None = None) -> Path:
    root = install or whatsapp_install_dir()
    return root / "whatsapp-mcp-main" / "whatsapp-mcp-server"


def _repo_root(install: Path) -> Path:
    return install / "whatsapp-mcp-main"


def ensure_whatsapp_mcp_repo(install_dir: Path | None = None) -> Path:
    """Download lharries/whatsapp-mcp zip into ``install_dir`` if not present."""
    install = install_dir or whatsapp_install_dir()
    install.mkdir(parents=True, exist_ok=True)
    root = _repo_root(install)
    if (root / "whatsapp-bridge" / "main.go").is_file():
        return install

    zip_path = install / "whatsapp-mcp-main.zip"
    debug_log(f"downloading WhatsApp MCP repo to {install}", "whatsapp")
    with urlopen(REPO_ZIP_URL, timeout=120) as resp:
        zip_path.write_bytes(resp.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(install)
    try:
        zip_path.unlink()
    except OSError:
        pass

    if not (root / "whatsapp-bridge" / "main.go").is_file():
        raise RuntimeError("WhatsApp MCP download failed: whatsapp-bridge/main.go missing")
    return install


def apply_bridge_patch(main_go: Path | None = None) -> None:
    """Emit machine-readable QR + auth lines on stdout for the desktop UI."""
    path = main_go or (bridge_dir() / "main.go")
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return

    needle = 'if evt.Event == "code" {'
    if needle not in text:
        raise RuntimeError("Could not patch whatsapp-bridge: QR loop not found")

    patched = text.replace(
        needle,
        'if evt.Event == "code" {\n'
        '\t\t\t\tfmt.Printf("JARVIS_QR_CODE:%s\\n", evt.Code)',
        1,
    )
    patched = patched.replace(
        '} else if evt.Event == "success" {',
        '} else if evt.Event == "success" {\n'
        '\t\t\t\tfmt.Println("JARVIS_AUTH_OK")',
        1,
    )
    success_msg = 'fmt.Println("\\nSuccessfully connected and authenticated!")'
    if success_msg in patched:
        patched = patched.replace(
            success_msg,
            success_msg + '\n\t\t\tfmt.Println("JARVIS_AUTH_OK")',
            1,
        )
    already_logged = "connected <- true\n\t}"
    if already_logged in patched and 'Already logged in' in patched:
        patched = patched.replace(
            "connected <- true\n\t}",
            "connected <- true\n\t\tfmt.Println(\"JARVIS_AUTH_OK\")\n\t}",
            1,
        )
    path.write_text(patched, encoding="utf-8")
    debug_log(f"patched {path} for Jarvis QR output", "whatsapp")


_WHATSMEOW_CONTEXT_REPLACEMENTS = (
    ("client.Download(downloader)", "client.Download(context.Background(), downloader)"),
    ('sqlstore.New("sqlite3"', 'sqlstore.New(context.Background(), "sqlite3"'),
    ("container.GetFirstDevice()", "container.GetFirstDevice(context.Background())"),
    ("client.GetGroupInfo(jid)", "client.GetGroupInfo(context.Background(), jid)"),
    (
        "client.Store.Contacts.GetContact(jid)",
        "client.Store.Contacts.GetContact(context.Background(), jid)",
    ),
)


def apply_whatsmeow_context_patch(main_go: Path | None = None) -> bool:
    """Patch lharries bridge for whatsmeow APIs that require context.Context."""
    path = main_go or (bridge_dir() / "main.go")
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    modified = False
    for old, new in _WHATSMEOW_CONTEXT_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new, 1)
            modified = True
    if modified:
        path.write_text(text, encoding="utf-8")
        debug_log(f"patched {path} for whatsmeow context API", "whatsapp")
    return modified


def upgrade_whatsapp_bridge_deps(bdir: Path | None = None) -> None:
    """Update whatsmeow so WhatsApp servers do not reject the client (405 outdated)."""
    directory = bdir or bridge_dir()
    if not (directory / "go.mod").is_file():
        return
    env = augmented_path_env()
    env["CGO_ENABLED"] = "1"
    go_cmd = resolve_go_command()
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    for args in (["get", "-u", "go.mau.fi/whatsmeow@latest"], ["mod", "tidy"]):
        try:
            subprocess.run(
                [go_cmd] + args,
                cwd=str(directory),
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                creationflags=creationflags,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            debug_log(f"whatsapp bridge go {args[0]} failed: {exc}", "whatsapp")
    apply_whatsmeow_context_patch(directory / "main.go")


def _go_search_dirs() -> list[str]:
    """Common Go install locations (GUI apps may not inherit updated PATH)."""
    dirs: list[str] = []
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(key)
        if base:
            dirs.append(str(Path(base) / "Go" / "bin"))
    local = Path.home() / "go" / "bin"
    dirs.append(str(local))
    return dirs


def go_available() -> bool:
    if shutil.which("go"):
        return True
    for d in _go_search_dirs():
        candidate = Path(d) / "go.exe"
        if candidate.is_file():
            return True
    return False


def resolve_go_command() -> str:
    found = shutil.which("go")
    if found:
        return found
    for d in _go_search_dirs():
        candidate = Path(d) / "go.exe"
        if candidate.is_file():
            return str(candidate)
    return "go"


def _gcc_search_dirs() -> list[str]:
    """Common MinGW / WinLibs locations (GUI apps may not inherit updated PATH)."""
    dirs: list[str] = []
    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "mingw64" / "bin",
        Path(r"C:\msys64\mingw64\bin"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "mingw64" / "bin",
    ):
        if candidate.is_dir():
            dirs.append(str(candidate))

    winget_packages = (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    )
    if winget_packages.is_dir():
        for pkg in winget_packages.iterdir():
            name_lower = pkg.name.lower()
            if "winlibs" in name_lower or "mingw" in name_lower:
                mingw_bin = pkg / "mingw64" / "bin"
                if mingw_bin.is_dir():
                    dirs.append(str(mingw_bin))
    return dirs


def resolve_gcc_command(env: dict[str, str] | None = None) -> str | None:
    """Return path to gcc.exe if found on PATH or in known install dirs."""
    check_env = env if env is not None else augmented_path_env()
    path_val = check_env.get("PATH", os.environ.get("PATH", ""))
    for folder in path_val.split(os.pathsep):
        if not folder:
            continue
        for name in ("gcc", "cc", "x86_64-w64-mingw32-gcc"):
            candidate = Path(folder) / f"{name}.exe"
            if candidate.is_file():
                return str(candidate)
            candidate_plain = Path(folder) / name
            if candidate_plain.is_file():
                return str(candidate_plain)

    for folder in _gcc_search_dirs():
        candidate = Path(folder) / "gcc.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def gcc_available() -> bool:
    """WhatsApp bridge uses mattn/go-sqlite3 and needs a C compiler on Windows."""
    if resolve_gcc_command() is not None:
        return True
    for name in ("gcc", "cc", "x86_64-w64-mingw32-gcc"):
        if shutil.which(name):
            return True
    return False


def windows_gcc_install_hint() -> str:
    return (
        "On Windows the WhatsApp bridge needs gcc (C compiler) for SQLite.\n"
        "Install: winget install BrechtSanders.WinLibs.POSIX.UCRT\n"
        "Then open a new terminal and run Connect WhatsApp again.\n"
        "Alternative: MSYS2 + pacman -S mingw-w64-x86_64-gcc (add mingw64\\bin to PATH)."
    )


def augmented_path_env() -> dict[str, str]:
    """PATH with Go, uv, gcc, and npm dirs for subprocesses launched from the GUI."""
    env = os.environ.copy()
    extra: list[str] = []
    extra.extend(_go_search_dirs())
    extra.extend(_gcc_search_dirs())
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.is_dir():
        extra.append(str(local_bin))
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(key)
        if base:
            extra.append(str(Path(base) / "nodejs"))
    combined = os.pathsep.join(extra + [env.get("PATH", "")])
    env["PATH"] = combined
    return env


def uv_available() -> bool:
    return bool(shutil.which("uv") or shutil.which("uvx"))


def lharries_whatsapp_mcp_config(install_dir: Path | None = None) -> dict:
    """MCP entry for the cloned lharries Python server."""
    install = install_dir or whatsapp_install_dir()
    server = mcp_server_dir(install)
    if not server.is_dir():
        raise RuntimeError(f"WhatsApp MCP server not found at {server}")
    uv_cmd = shutil.which("uv") or "uv"
    return {
        "transport": "stdio",
        "command": uv_cmd,
        "args": ["--directory", str(server), "run", "main.py"],
    }


def render_qr_png(pairing_code: str) -> bytes:
    import qrcode

    img = qrcode.make(pairing_code)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bridge_api_ready() -> bool:
    """True when the lharries Go bridge REST API is listening on port 8080."""
    try:
        import requests

        # ``/api/send`` only allows POST; GET → 405 proves the bridge is up.
        r = requests.get("http://localhost:8080/api/send", timeout=2)
        return r.status_code == 405
    except Exception:
        return False


def migrate_legacy_whatsapp_mcp_config() -> bool:
    """Replace uvx ``whatsapp-mcp-server`` (GreenAPI) with lharries when the bridge exists."""
    server = mcp_server_dir(whatsapp_install_dir())
    if not (server.is_dir() and (server / "main.py").is_file()):
        return False
    from jarvis.config import _load_json, _save_json

    path = default_config_path()
    cfg = _load_json(path) or {}
    mcps = cfg.get("mcps")
    if not isinstance(mcps, dict):
        return False
    entry = mcps.get("whatsapp")
    if not isinstance(entry, dict):
        return False
    if entry.get("command") == "uvx" and "whatsapp-mcp-server" in (entry.get("args") or []):
        mcps["whatsapp"] = lharries_whatsapp_mcp_config()
        cfg["mcps"] = mcps
        _save_json(path, cfg)
        debug_log("migrated mcps.whatsapp from uvx to lharries bridge", "whatsapp")
        return True
    return False


def save_whatsapp_mcp_to_config(install_dir: Path | None = None) -> None:
    from jarvis.config import _load_json, _save_json

    path = default_config_path()
    cfg = _load_json(path) or {}
    mcps = cfg.get("mcps")
    if not isinstance(mcps, dict):
        mcps = {}
    mcps["whatsapp"] = lharries_whatsapp_mcp_config(install_dir)
    cfg["mcps"] = mcps
    cfg["whatsapp_bridge_enabled"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(path, cfg)


class WhatsAppBridgeController:
    """Start/stop the Go bridge and surface QR + auth events."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        *,
        on_qr: Callable[[str], None],
        on_authenticated: Callable[[], None],
        on_log: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        with self._lock:
            if self.running:
                on_log("Bridge already running.")
                return

            if not go_available():
                on_error(
                    "Go is not installed. Install from https://go.dev/dl/ "
                    "then restart Jarvis."
                )
                return
            if not uv_available():
                on_error(
                    "uv is not on PATH. Install from https://docs.astral.sh/uv/ "
                    "then restart Jarvis."
                )
                return
            if sys.platform == "win32" and not gcc_available():
                on_error(windows_gcc_install_hint())
                return

            migrate_legacy_whatsapp_mcp_config()

            try:
                install = ensure_whatsapp_mcp_repo()
                bdir_prep = bridge_dir(install)
                upgrade_whatsapp_bridge_deps(bdir_prep)
                apply_bridge_patch(bdir_prep / "main.go")
            except Exception as exc:
                on_error(f"Could not prepare WhatsApp files: {exc}")
                return

            bdir = bridge_dir(install)
            env = augmented_path_env()
            env["CGO_ENABLED"] = "1"
            go_cmd = resolve_go_command()

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

            try:
                self._proc = subprocess.Popen(
                    [go_cmd, "run", "main.go"],
                    cwd=str(bdir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=creationflags,
                )
            except Exception as exc:
                on_error(f"Failed to start bridge: {exc}")
                self._proc = None
                return

            on_log("Starting WhatsApp bridge (first run may compile Go for 1–3 min)…")
            recent: Deque[str] = deque(maxlen=24)

            def _read_stdout() -> None:
                assert self._proc and self._proc.stdout
                for line in self._proc.stdout:
                    line = line.rstrip("\n\r")
                    if not line:
                        continue
                    recent.append(line)
                    m = _QR_LINE.match(line)
                    if m:
                        on_qr(m.group(1))
                        continue
                    if _AUTH_OK in line:
                        on_authenticated()
                        continue
                    low = line.lower()
                    if any(
                        tok in low
                        for tok in (
                            "error",
                            "failed",
                            "gcc",
                            "cgo",
                            "sqlite",
                            "scan this qr",
                            "successfully connected",
                        )
                    ) or line.startswith("#") or "go:" in line or "building" in low:
                        on_log(line)

            def _watch_exit() -> None:
                assert self._proc is not None
                code = self._proc.wait()
                if code is None or code == 0:
                    return
                tail = "\n".join(recent)
                hint = windows_gcc_install_hint() if "gcc" in tail.lower() or "cgo" in tail.lower() else ""
                msg = f"Bridge exited (code {code})."
                if tail:
                    msg += f"\n\nLast output:\n{tail}"
                if hint:
                    msg += f"\n\n{hint}"
                on_error(msg)

            self._reader = threading.Thread(target=_read_stdout, daemon=True)
            self._reader.start()
            threading.Thread(target=_watch_exit, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
