"""Collection is offline; selecting a benchmark owns its availability check."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('mode', ['import', 'collect', 'selected', 'available'])
def test_benchmark_network_is_deferred_until_selected(tmp_path, mode):
    repo = Path(__file__).resolve().parents[1]
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    config = tmp_path / 'config.json'
    config.write_text('{}')
    marker = tmp_path / 'availability'
    native = tmp_path / 'unowned-network'
    control = tmp_path / 'test_control.py'
    control.write_text('import pytest\n@pytest.mark.unit\ndef test_owned_control(): pass\n')
    script = '''
import importlib, os
from pathlib import Path
import requests
marker = Path(os.environ['AVAILABILITY_MARKER'])
def check(*args, **kwargs):
    marker.write_text('availability checked')
    if os.environ['PROBE_MODE'] == 'available':
        class Response:
            status_code = 200
            def json(self):
                return {'models': [{'name': os.environ['JARVIS_PERF_MODEL']}]}
        return Response()
    return None
requests.get = check
def blocked(*args, **kwargs):
    Path(os.environ['NETWORK_MARKER']).write_text('unowned request')
    raise AssertionError('Unowned transport forbidden')
requests.Session.request = blocked
if os.environ['PROBE_MODE'] == 'available':
    from jarvis.llm.ollama import OllamaBackend
    from jarvis.reply import engine
    OllamaBackend.direct = lambda *args, **kwargs: 'OK'
    def run_reply_engine(db, cfg, location, query, dialogue):
        return OllamaBackend('http://owned.invalid').direct(cfg.llm_chat_model, 'system', query)
    engine.run_reply_engine = run_reply_engine
if os.environ['PROBE_MODE'] == 'import':
    importlib.import_module('tests.performance.test_pipeline_timings')
else:
    import pytest
    args = ['-q', '-c', os.environ['INI_PATH'], '-m', 'performance' if os.environ['PROBE_MODE'] in ('selected', 'available') else 'unit', os.environ['PERF_PATH']]
    if os.environ['PROBE_MODE'] == 'collect':
        args.extend(['--collect-only', os.environ['CONTROL_PATH']])
    raise SystemExit(pytest.main(args))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', QT_QPA_PLATFORM='offscreen',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PROBE_MODE=mode,
               PYTHONPATH=str(repo / 'src') + os.pathsep + str(repo),
               TMPDIR=str(runtime), TEMP=str(runtime), TMP=str(runtime),
               JARVIS_CONFIG_PATH=str(config), XDG_CONFIG_HOME=str(tmp_path/'xdg'),
               JARVIS_PERF_OLLAMA_URL='http://owned.invalid', JARVIS_PERF_MODEL='owned-benchmark-model',
               JARVIS_PERF_RUNS='1', JARVIS_PERF_REPORT_DIR=str(tmp_path/'reports'),
               AVAILABILITY_MARKER=str(marker), NETWORK_MARKER=str(native),
               INI_PATH=str(repo/'pytest.ini'), CONTROL_PATH=str(control),
               PERF_PATH=str(repo/'tests/performance/test_pipeline_timings.py'))
    result = subprocess.run([sys.executable, '-c', script], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert marker.exists() == (mode in ('selected', 'available')), output
    assert not native.exists(), output
    if mode == 'selected':
        assert 'skipped' in output
        assert ' passed' not in output
    elif mode == 'available':
        assert 'passed' in output
        reports = list((tmp_path/'reports').glob('*.json'))
        assert reports
        assert all(json.loads(report.read_text())['status'] == 'completed' for report in reports)
