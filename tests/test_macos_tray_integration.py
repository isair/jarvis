"""Native Qt tray callbacks survive non-mouse activation in isolation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(sys.platform != "darwin", reason="Native Cocoa tray test")]


def test_native_tray_non_mouse_and_mouse_activation():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["QT_QPA_PLATFORM"] = "cocoa"
    result = subprocess.run(
        [sys.executable, str(root / "tests" / "fixtures" / "macos_tray_probe.py")],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Tray tracking returned safely" in result.stdout
    assert "Real mouse activation is preserved" in result.stdout


def test_frozen_native_tray_event_guard(tmp_path):
    pytest.importorskip("PyInstaller")
    root = Path(__file__).resolve().parents[1]
    spec = tmp_path / "tray-probe.spec"
    spec.write_text(f'''a=Analysis([{str(root / "tests" / "fixtures" / "macos_tray_probe.py")!r}],pathex=[{str(root / "src")!r}],hiddenimports=[],excludes=['torch','tensorflow','transformers','mlx','matplotlib','IPython','notebook','sklearn'])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='tray-probe',console=True)
collect=COLLECT(exe,a.binaries,a.datas,name='tray-probe')
''')
    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(tmp_path / "cache")
    environment["QT_QPA_PLATFORM"] = "cocoa"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(tmp_path / "dist"),
         "--workpath", str(tmp_path / "build"), str(spec)],
        env=environment, check=True, capture_output=True, text=True, timeout=600,
    )
    result = subprocess.run(
        [str(tmp_path / "dist" / "tray-probe" / "tray-probe")],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Tray tracking returned safely" in result.stdout
    assert "Real mouse activation is preserved" in result.stdout
