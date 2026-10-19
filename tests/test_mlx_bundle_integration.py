"""A frozen Apple Silicon speech backend can execute its native runtime paths."""
from pathlib import Path
import os
import platform
import subprocess
import sys
import pytest

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(sys.platform != 'darwin' or platform.machine() != 'arm64',
                                 reason='Apple Silicon packaging integration')]
ROOT = Path(__file__).resolve().parents[1]


def test_frozen_mlx_speech_runtime(tmp_path):
    pytest.importorskip('PyInstaller')
    pytest.importorskip('mlx_whisper')
    spec = tmp_path/'probe.spec'
    spec.write_text(f'''import runpy
helper=runpy.run_path({str(ROOT/'installer'/'mlx_bundle.py')!r})
hidden,data,native=helper['collect_mlx_whisper']()
a=Analysis([{str(ROOT/'tests'/'fixtures'/'mlx_bundle_probe.py')!r}],pathex=[],binaries=native,datas=data,hiddenimports=hidden,excludes=['torch','torchaudio','torchvision','mlx_whisper.torch_whisper','matplotlib','IPython','notebook','sklearn'])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='mlx-probe',console=True)
collect=COLLECT(exe,a.binaries,a.datas,name='mlx-probe')
''')
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm',
                    '--distpath',str(tmp_path/'dist'),'--workpath',str(tmp_path/'build'),
                    str(spec)],check=True,capture_output=True,text=True,timeout=600)
    env={**os.environ,'HF_HUB_OFFLINE':'1'}
    arguments=[str(tmp_path/'dist'/'mlx-probe'/'mlx-probe')]
    model=os.environ.get('JARVIS_MLX_SMOKE_MODEL')
    audio=os.environ.get('JARVIS_MLX_SMOKE_AUDIO')
    if model and audio:
        arguments += [model,audio]
    result=subprocess.run(arguments,
                          capture_output=True,text=True,env=env,timeout=120)
    assert result.returncode==0, result.stdout+result.stderr
    assert 'FROZEN_MLX_METAL_PASSED' in result.stdout
    if model and audio:
        assert 'FROZEN_MLX_TRANSCRIPTION_PASSED' in result.stdout
