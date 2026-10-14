"""Run a desktop-worker scenario with owned storage and physical audio blocked."""
import atexit
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

owned = Path(sys.argv[1]).resolve(strict=True)
scenario = Path(sys.argv[2]).resolve(strict=True)
scenario.relative_to(owned)
data = owned/'data'
data.mkdir(exist_ok=True)
runtime = owned/'runtime'
runtime.mkdir(exist_ok=True)
for name in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ[name] = str(runtime)
tempfile.tempdir = str(runtime)
os.environ.update(WORKER_PROBE_ROOT=str(data), WORKER_PRIVATE_ROOT=str(data),
                  XDG_CONFIG_HOME=str(data/'xdg'), XDG_CACHE_HOME=str(runtime/'cache'),
                  JARVIS_CONFIG_PATH=str(data/'config.json'), QT_QPA_PLATFORM='offscreen')
(data/'config.json').write_text(json.dumps({'_config_version': 3, 'voice_debug': False,
                                           'db_path': str(data/'synthetic.sqlite')}))
Path.home = classmethod(lambda cls: data)
attempts = []


def check_write(path):
    if isinstance(path, int) or os.fspath(path) == os.devnull:
        return
    if not Path(path).resolve().is_relative_to(owned):
        attempts.append('outside write or replacement')
        raise AssertionError('Worker probe write is outside owned storage')


def guarded_open(original):
    def open_file(path, mode='r', *args, **kwargs):
        if any(flag in str(mode) for flag in ('w', 'a', 'x', '+')):
            check_write(path)
        return original(path, mode, *args, **kwargs)
    return open_file


builtins.open = guarded_open(builtins.open)
io.open = guarded_open(io.open)
original_os_open = os.open


def open_descriptor(path, flags, *args, **kwargs):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        check_write(path)
    return original_os_open(path, flags, *args, **kwargs)


def guarded_move(original):
    def move(source, destination, *args, **kwargs):
        check_write(source)
        check_write(destination)
        return original(source, destination, *args, **kwargs)
    return move


os.open = open_descriptor
os.replace = guarded_move(os.replace)
os.rename = guarded_move(os.rename)


def forbidden_effects():
    return attempts + sorted(getattr(globals().get('policy'), '_audio_attempts', ()))


@atexit.register
def audit_shutdown():
    effects = forbidden_effects()
    if effects:
        print('🔇 Worker probe attempted forbidden effects: '+str(effects), file=sys.stderr, flush=True)
        os._exit(1)


policy_path = Path(__file__).resolve().parents[2]/'conftest.py'
spec = importlib.util.spec_from_file_location('worker_audio_policy', policy_path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
policy.pytest_sessionstart(None)
try:
    runpy.run_path(str(scenario), run_name='__main__')
finally:
    if forbidden_effects():
        raise AssertionError('Worker probe attempted forbidden effects: '+str(forbidden_effects()))
