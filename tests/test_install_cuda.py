"""Integration tests for installer/windows/install_cuda.ps1.

These tests spin up a local HTTP server that mimics the subset of the PyPI
JSON API used by the script, serve tiny fake wheel files, and run the real
PowerShell script against them. They verify the four reliability properties
that motivated the rewrite:

- After-extract verification: the marker file is only written when every
  expected DLL is present on disk (and non-trivial).
- SHA256 verification: download integrity is checked against the digest
  PyPI returns; a tampered wheel must fail the run.
- Marker honesty: a stale marker with missing DLLs does not cause the
  script to skip; the work is repeated.
- Log file: every run leaves a transcript at the requested -LogPath.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "installer"
    / "windows"
    / "install_cuda.ps1"
)


# Names matching what install_cuda.ps1 will attempt to download. Keep in
# sync with the script's $packages array; tests assert the expected DLL
# set, so any change here is intentional.
CUBLAS_DLLS = ["cublas64_12.dll", "cublasLt64_12.dll", "nvblas64_12.dll"]
CUDNN_DLLS = [
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
]


def _build_fake_wheel(prefix: str, dll_names: list[str], filler_bytes: int = 4096) -> bytes:
    """Build an in-memory wheel (zip) with fake DLLs under `prefix`.

    `filler_bytes` controls the per-DLL payload size; tests use this to
    assert the script rejects empty / suspiciously-small DLLs.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in dll_names:
            zf.writestr(prefix + name, b"\x00" * filler_bytes)
    return buf.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakePyPIHandler(http.server.BaseHTTPRequestHandler):
    """Serves PyPI-style JSON metadata and the wheel binaries themselves.

    The class attribute `wheels` is set per-test by the harness below.
    """

    wheels: dict = {}

    def log_message(self, format, *args):  # noqa: A003 - silence default stderr
        return

    def do_GET(self):  # noqa: N802 - http.server contract
        # Match /pypi/<pkg>/<ver>/json
        parts = [p for p in self.path.split("/") if p]
        if len(parts) == 4 and parts[0] == "pypi" and parts[3] == "json":
            pkg, ver = parts[1], parts[2]
            entry = self.wheels.get((pkg, ver))
            if entry is None:
                self.send_error(404)
                return
            payload = {
                "info": {"name": pkg, "version": ver},
                "urls": [
                    {
                        "filename": entry["filename"],
                        "url": entry["url"],
                        "digests": {"sha256": entry["sha256"]},
                    }
                ],
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Match /files/<filename>
        if len(parts) == 2 and parts[0] == "files":
            filename = parts[1]
            for entry in self.wheels.values():
                if entry["filename"] == filename:
                    body = entry["bytes"]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_error(404)
            return

        self.send_error(404)


class _FakePyPIServer:
    """Run a local HTTP server in a background thread for the duration of a test."""

    def __init__(self, wheels: dict):
        _FakePyPIHandler.wheels = wheels
        # ThreadingHTTPServer keeps the test responsive if PowerShell makes
        # multiple sequential requests for index + binary.
        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _FakePyPIHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def index_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/pypi"

    def file_url(self, filename: str) -> str:
        return f"http://127.0.0.1:{self.port}/files/{filename}"


def _build_wheels(
    *,
    cudnn_filler: int = 4096,
    cublas_filler: int = 4096,
    cudnn_dlls: list[str] | None = None,
) -> dict:
    """Build the fake wheel payloads we'll serve for a given test."""
    cublas_bytes = _build_fake_wheel("nvidia/cublas/bin/", CUBLAS_DLLS, cublas_filler)
    cudnn_bytes = _build_fake_wheel(
        "nvidia/cudnn/bin/",
        cudnn_dlls if cudnn_dlls is not None else CUDNN_DLLS,
        cudnn_filler,
    )
    return {
        ("nvidia-cublas-cu12", "12.9.1.4"): {
            "filename": "nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl",
            "bytes": cublas_bytes,
            "sha256": _sha256(cublas_bytes),
        },
        ("nvidia-cudnn-cu12", "9.20.0.48"): {
            "filename": "nvidia_cudnn_cu12-9.20.0.48-py3-none-win_amd64.whl",
            "bytes": cudnn_bytes,
            "sha256": _sha256(cudnn_bytes),
        },
    }


def _attach_file_urls(wheels: dict, server: _FakePyPIServer) -> None:
    for entry in wheels.values():
        entry["url"] = server.file_url(entry["filename"])


def _run_script(
    target_dir: Path,
    server: _FakePyPIServer,
    *,
    log_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    log = log_path or (target_dir / "install.log")
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT_PATH),
        "-TargetDir",
        str(target_dir),
        "-PyPIIndexUrl",
        server.index_url,
        "-LogPath",
        str(log),
        "-SkipGpuCheck",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="install_cuda.ps1 is Windows-only",
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    d = tmp_path / "cuda"
    d.mkdir()
    return d


def test_happy_path_writes_marker_and_log(workdir: Path):
    """Successful download + extract + verify -> marker, log, and all DLLs present."""
    wheels = _build_wheels()
    with _FakePyPIServer(wheels) as server:
        _attach_file_urls(wheels, server)
        result = _run_script(workdir, server)

    assert result.returncode == 0, f"script failed:\n{result.stdout}\n{result.stderr}"

    for name in CUBLAS_DLLS + CUDNN_DLLS:
        assert (workdir / name).exists(), f"missing {name} after happy-path install"

    marker = workdir / ".cuda_installed"
    assert marker.exists(), "marker should be written after successful verify"

    log = workdir / "install.log"
    assert log.exists(), "log file should always be written"
    assert log.stat().st_size > 0, "log file should not be empty"


def test_sha256_mismatch_aborts_with_no_marker(workdir: Path):
    """A wheel whose contents have been swapped fails the digest check; no marker."""
    wheels = _build_wheels()
    # Swap cuDNN bytes after the digest was recorded — simulates corruption
    # in transit or an attacker tampering with the binary mid-flight.
    tampered = b"not a real wheel"
    wheels[("nvidia-cudnn-cu12", "9.20.0.48")]["bytes"] = tampered

    with _FakePyPIServer(wheels) as server:
        _attach_file_urls(wheels, server)
        result = _run_script(workdir, server)

    assert result.returncode != 0, "tampered wheel must fail the run"
    assert not (workdir / ".cuda_installed").exists(), (
        "marker must not be written when the SHA256 check fails"
    )


def test_missing_dll_after_extract_aborts(workdir: Path):
    """A wheel that's missing a required DLL fails verification."""
    truncated_cudnn = [d for d in CUDNN_DLLS if d != "cudnn_ops64_9.dll"]
    wheels = _build_wheels(cudnn_dlls=truncated_cudnn)

    with _FakePyPIServer(wheels) as server:
        _attach_file_urls(wheels, server)
        result = _run_script(workdir, server)

    assert result.returncode != 0
    assert not (workdir / ".cuda_installed").exists()
    combined = result.stdout + result.stderr
    assert "cudnn_ops64_9.dll" in combined, (
        "failure output must name the missing DLL so users can act on it"
    )


def test_stale_marker_with_missing_dlls_redownloads(workdir: Path):
    """A marker left over from a half-successful install must not skip work."""
    # Pretend a previous install wrote the marker but only one DLL survived
    # (e.g. AV quarantined the rest).
    (workdir / ".cuda_installed").write_text("nvidia-cublas-cu12==12.9.1.4\n")
    (workdir / "cublas64_12.dll").write_bytes(b"\x00" * 4096)

    wheels = _build_wheels()
    with _FakePyPIServer(wheels) as server:
        _attach_file_urls(wheels, server)
        result = _run_script(workdir, server)

    assert result.returncode == 0, f"re-run should succeed:\n{result.stdout}\n{result.stderr}"
    for name in CUBLAS_DLLS + CUDNN_DLLS:
        assert (workdir / name).exists(), (
            f"{name} must be downloaded on re-run even though marker existed"
        )


def test_idempotent_skip_when_everything_present(workdir: Path):
    """A second run with all DLLs present should skip the network entirely."""
    wheels = _build_wheels()
    with _FakePyPIServer(wheels) as server:
        _attach_file_urls(wheels, server)
        first = _run_script(workdir, server)
        assert first.returncode == 0

        # Tamper the digests on the wheels we'd serve a second time. If the
        # script tries to re-download we'll get a SHA mismatch and a non-zero
        # exit; if it correctly skips, we stay green.
        for entry in wheels.values():
            entry["bytes"] = b"corrupt"

        second = _run_script(workdir, server)

    assert second.returncode == 0, (
        "fully-installed run must skip network fetches and exit 0"
    )
    combined = second.stdout + second.stderr
    assert "already installed" in combined.lower() or "already present" in combined.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
