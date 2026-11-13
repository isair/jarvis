"""A frozen local Whisper runtime retains its capabilities and speech assets."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def test_frozen_faster_whisper_runtime(tmp_path):
    pytest.importorskip("PyInstaller")
    pytest.importorskip("faster_whisper")
    spec = tmp_path / "probe.spec"
    spec.write_text(f'''a=Analysis(
    [{str(ROOT/'tests'/'fixtures'/'faster_whisper_bundle_probe.py')!r}],
    pathex=[{str(ROOT/'src')!r}],
    hookspath=[{str(ROOT/'installer'/'hooks')!r}],
    excludes=['torch','torchaudio','torchvision','transformers','matplotlib','IPython','notebook','sklearn'])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='whisper-probe',console=True)
collect=COLLECT(exe,a.binaries,a.datas,name='whisper-probe')
''')
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--distpath", str(tmp_path / "dist"), "--workpath", str(tmp_path / "build"),
                    str(spec)], check=True, capture_output=True, text=True, timeout=600)
    executable = "whisper-probe.exe" if sys.platform == "win32" else "whisper-probe"
    result = subprocess.run([str(tmp_path / "dist" / "whisper-probe" / executable)],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FROZEN_WHISPER_RUNTIME_PASSED" in result.stdout
