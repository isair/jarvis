"""Desktop resource declarations carry a usable alarm into the bundle."""
import ast
from glob import glob
from pathlib import Path
import shutil
import wave

import pytest

pytestmark = pytest.mark.unit


def test_desktop_data_manifest_stages_the_alarm_clip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'jarvis_desktop.spec').read_text(encoding='utf-8'))
    declaration = next(node.value for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'datas'
                               for target in node.targets))
    manifest = eval(compile(ast.Expression(declaration), 'desktop-data', 'eval'),
                    {'str': str, 'src_path': root / 'src', '__builtins__': {}})
    for pattern, destination in manifest:
        for source in glob(pattern):
            staged = tmp_path / destination / Path(source).name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)
    alarm = tmp_path / 'desktop_app' / 'desktop_assets' / 'alarm.wav'
    assert alarm.is_file(), 'The desktop data manifest omits its alarm clip'
    with wave.open(str(alarm), 'rb') as clip:
        assert clip.getnchannels() > 0
        assert clip.getframerate() > 0
        assert clip.getnframes() > 0
        assert clip.readframes(clip.getnframes())
