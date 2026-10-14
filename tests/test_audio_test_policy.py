"""Ordinary pytest runs reject native audio while permitting owned fakes."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


_CASES = {
    'direct': '''
        @pytest.mark.parametrize('backend,operation', [
            ('sounddevice', 'InputStream'), ('sounddevice', 'OutputStream'),
            ('sounddevice', 'play'), ('winsound', 'PlaySound'),
            ('pygame', 'init'), ('pygame.mixer', 'init'), ('pygame.mixer.music', 'play'),
        ])
        def test_audio(backend, operation):
            getattr(importlib.import_module(backend), operation)()
    ''',
    'caught': '''
        def test_caught_audio_error():
            try:
                sounddevice.play()
            except AssertionError:
                pass
    ''',
    'teardown': '''
        @pytest.fixture
        def delayed_audio():
            resume = threading.Event()
            def play():
                assert resume.wait(5)
                try:
                    sounddevice.play()
                except AssertionError:
                    pass
            worker = threading.Thread(target=play)
            worker.start()
            try:
                yield
            finally:
                resume.set()
                worker.join(5)
                assert not worker.is_alive()
        def test_delayed_audio(delayed_audio):
            pass
    ''',
    'late_hook': '''
        def test_finishes_before_shutdown_hook():
            pass
    ''',
    'eval_only': '''
        def test_eval_audio():
            sounddevice.play()
    ''',
    'fake': '''
        def test_owned_feedback(monkeypatch):
            monkeypatch.setattr(sounddevice, 'play', lambda: Path(os.environ['FAKE_MARKER']).write_text('owned feedback'))
            sounddevice.play()
    ''',
}


@pytest.mark.parametrize('mode', list(_CASES))
def test_audio_policy_at_native_boundary(tmp_path, mode):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    config = tmp_path / 'config.json'
    config.write_text('{}')
    ini = tmp_path / 'pytest.ini'
    ini.write_text('[pytest]\nmarkers =\n    unit: owned synthetic unit case\n')
    case_dir = tmp_path / 'evals' if mode == 'eval_only' else tmp_path
    case_dir.mkdir(exist_ok=True)
    case = case_dir / 'test_probe.py'
    case.write_text(
        'import importlib, os, threading\nfrom pathlib import Path\nimport pytest, sounddevice, winsound, pygame\npytestmark = pytest.mark.unit\n'
        + textwrap.dedent(_CASES[mode])
    )
    native_marker = tmp_path / 'native-call'
    fake_marker = tmp_path / 'fake-call'
    conftest = Path(__file__).resolve().parents[1] / 'conftest.py'
    repo = Path(__file__).resolve().parents[1]
    if mode == 'eval_only':
        for source in (repo / 'conftest.py', repo / 'tests' / 'conftest.py'):
            if source.exists():
                destination = tmp_path / source.relative_to(repo)
                destination.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(source, destination)
    if mode == 'late_hook':
        (tmp_path / 'conftest.py').write_text('import pytest, sounddevice\n@pytest.hookimpl(trylast=True)\ndef pytest_sessionfinish(session, exitstatus):\n    try: sounddevice.play()\n    except AssertionError: pass\n')
    # Every native entry is a harmless marker stub before pytest loads policy.
    driver = '''
import importlib.util, os, sys, types
from pathlib import Path
import pytest, sounddevice
marker = Path(os.environ['NATIVE_MARKER'])
def native(*args, **kwargs):
    marker.write_text('synthetic native boundary reached')
for operation in ('play', 'rec', 'playrec', 'InputStream', 'OutputStream', 'RawInputStream', 'RawOutputStream', 'Stream', 'RawStream'):
    setattr(sounddevice, operation, native)
winsound = types.ModuleType('winsound')
for operation in ('PlaySound', 'Beep', 'MessageBeep'):
    setattr(winsound, operation, native)
sys.modules['winsound'] = winsound
pygame = types.ModuleType('pygame')
pygame.__path__ = []
pygame.init = native
mixer = types.ModuleType('pygame.mixer')
mixer.init = native
music = types.ModuleType('pygame.mixer.music')
music.play = native
mixer.music = music
pygame.mixer = mixer
for name, module in (('pygame', pygame), ('pygame.mixer', mixer), ('pygame.mixer.music', music)):
    sys.modules[name] = module
name = 'owned_repo_policy'
arguments = ['-q', '-c', sys.argv[2], sys.argv[3]]
if os.environ['PROBE_MODE'] != 'eval_only':
    spec = importlib.util.spec_from_file_location(name, sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    arguments.extend(['-p', name])
raise SystemExit(pytest.main(arguments))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', QT_QPA_PLATFORM='offscreen',
               TMPDIR=str(runtime), TEMP=str(runtime), TMP=str(runtime),
               JARVIS_CONFIG_PATH=str(config), XDG_CONFIG_HOME=str(tmp_path / 'xdg'),
               NATIVE_MARKER=str(native_marker), FAKE_MARKER=str(fake_marker),
               PROBE_MODE=mode, PYTHONPATH=str(repo / 'src'))
    result = subprocess.run(
        [sys.executable, '-c', driver, str(conftest), str(ini), str(case)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == (0 if mode == 'fake' else 1), output
    assert not native_marker.exists(), output
    if mode == 'fake':
        assert fake_marker.read_text() == 'owned feedback'
    else:
        assert 'Physical audio access' in output
