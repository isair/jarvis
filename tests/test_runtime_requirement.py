"""Unsupported source runtimes stop before application dependencies are imported."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('module', ['jarvis.daemon', 'jarvis.memory.graph', 'desktop_app'])
@pytest.mark.parametrize('version', [(3, 10, 9), (3, 11, 0), (3, 12, 0)])
def test_runtime_floor_precedes_application_initialisation(tmp_path, module, version):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    config = runtime / 'config.json'
    config.write_text('{}')
    code = '''
import importlib.abc
import sys

class ApplicationBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('jarvis.config', 'desktop_app.app'):
            print('application boundary reached', flush=True)
            raise SystemExit(0)
        if fullname.startswith(('sounddevice', 'PyQt6', 'numpy')):
            raise RuntimeError('unexpected native dependency')

sys.meta_path.insert(0, ApplicationBoundary())
sys.version_info = VERSION
__import__(MODULE)
raise RuntimeError('application boundary was not reached')
'''.replace('VERSION', repr(version)).replace('MODULE', repr(module))
    env = {
        'PATH': os.environ.get('PATH', ''),
        'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONIOENCODING': 'utf-8',
        'JARVIS_CONFIG_PATH': str(config), 'XDG_CONFIG_HOME': str(runtime / 'xdg'),
        'TMPDIR': str(runtime), 'TMP': str(runtime), 'TEMP': str(runtime),
    }
    result = subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path,
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    if version < (3, 11):
        assert result.returncode != 0
        assert 'Python 3.11' in result.stderr
        assert 'application boundary reached' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'application boundary reached'
    assert config.read_text() == '{}'
