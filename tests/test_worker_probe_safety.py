"""Worker subprocess probes reject effects outside their owned sandbox."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('effect,timing', [
    ('none', 'immediate'), ('write', 'immediate'), ('write', 'shutdown'),
    ('audio', 'immediate'), ('audio', 'shutdown'),
])
def test_worker_probe_enforces_owned_write_and_audio_boundaries(tmp_path, effect, timing):
    owned = tmp_path / 'owned'
    owned.mkdir()
    runtime = owned / 'runtime'
    runtime.mkdir()
    outside = tmp_path / 'outside'
    native_marker = owned / 'native-audio'
    scenario = owned / 'scenario.py'
    code = '''from pathlib import Path
import os, sounddevice
owned = Path(os.environ['WORKER_PROBE_ROOT'])
source = owned/'source'
source.write_text('owned')
os.replace(source, owned/'destination')
descriptor = os.open(os.devnull, os.O_WRONLY)
os.close(descriptor)
'''
    attempts = '''
for operation in [lambda: open(os.environ['OUTSIDE_MARKER'], 'w').close(),
                  lambda: os.close(os.open(os.environ['OUTSIDE_MARKER'], os.O_WRONLY|os.O_CREAT)),
                  lambda: Path(os.environ['OUTSIDE_MARKER']).write_text('escaped'),
                  lambda: os.rename(owned/'destination',os.environ['OUTSIDE_MARKER']),
                  lambda: os.replace(owned/'destination',os.environ['OUTSIDE_MARKER'])]:
    try: operation()
    except AssertionError: pass
'''
    if effect == 'audio':
        attempts = '''
for operation in [lambda: sounddevice.play(), lambda: sounddevice.InputStream()]:
    try: operation()
    except AssertionError: pass
'''
    if effect != 'none' and timing == 'shutdown':
        code += '\nimport atexit\n@atexit.register\ndef caught_late_attempts():\n'+'\n'.join('    '+line for line in attempts.splitlines())+'\n'
    elif effect != 'none':
        code += attempts
    scenario.write_text(code)
    # Native entries are marker stubs before the policy loads, even if it breaks.
    driver = '''import os, runpy, sys, types
from pathlib import Path
def native(*args,**kwargs): Path(os.environ['NATIVE_MARKER']).write_text('reached')
for name,operations in {'sounddevice':('play','rec','playrec','InputStream','OutputStream','RawInputStream','RawOutputStream','Stream','RawStream'),'winsound':('PlaySound','Beep','MessageBeep')}.items():
    module=types.ModuleType(name)
    for operation in operations:setattr(module,operation,native)
    sys.modules[name]=module
pygame=types.ModuleType('pygame');pygame.init=native
pygame.mixer=types.SimpleNamespace(init=native,music=types.SimpleNamespace(play=native))
sys.modules['pygame']=pygame
runpy.run_path(sys.argv.pop(1),run_name='__main__')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(runtime), TEMP=str(runtime), TMP=str(runtime),
               OUTSIDE_MARKER=str(outside), NATIVE_MARKER=str(native_marker))
    result = subprocess.run(
        [sys.executable, '-c', driver, str(ROOT/'tests/fixtures/worker_probe.py'),
         str(owned), str(scenario)], env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == (1 if effect != 'none' else 0), result.stdout+result.stderr
    assert not outside.exists()
    assert not native_marker.exists()
    if effect != 'none':
        assert 'Worker probe attempted forbidden effects' in result.stderr
    else:
        assert (owned/'data/destination').read_text() == 'owned'
