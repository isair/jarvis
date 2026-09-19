"""Qt keeps one application alive while test-owned widgets are disposed safely."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_qt_fixture_survives_test_boundaries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scenario = tmp_path / 'test_lifecycle.py'
    scenario.write_text(textwrap.dedent('''
        import weakref
        import pytest
        from PyQt6 import sip
        from PyQt6.QtWidgets import QWidget

        state = {}

        def test_first(qapp):
            state['app'] = weakref.ref(qapp)
            state['widget'] = QWidget()
            state['widget'].show()

        def test_second(qapp):
            assert state['app']() is qapp, 'QApplication was destroyed between tests'
            assert sip.isdeleted(state['widget']), 'Test-owned widget leaked'

        @pytest.mark.parametrize('iteration', range(20))
        def test_chat_churn(qapp, monkeypatch, iteration):
            from desktop_app.chat_window import ChatWindow
            monkeypatch.setattr('desktop_app.chat_window.get_hot_window_messages', lambda: [])
            window = ChatWindow(submit_fn=lambda text: None)
            window._append_user('test message')
            window.show()
            qapp.processEvents()
            window.close()
    '''), encoding='utf-8')
    runner = '''
import importlib.util, pytest, sys
spec = importlib.util.spec_from_file_location('qt_fixtures', sys.argv[1])
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)
raise SystemExit(pytest.main(['-q', sys.argv[2]], plugins=[fixtures]))
'''
    env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
    result = subprocess.run(
        [sys.executable, '-c', runner, str(root / 'tests/conftest.py'), str(scenario)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
