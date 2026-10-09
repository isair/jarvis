"""Default configuration advertises active controls and accepts saved older files."""
import json

import pytest

from jarvis.config import get_default_config, load_settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('key', ['evaluator_enabled', 'evaluator_nudge_max'])
def test_default_configuration_excludes_inactive_turn_controls(key):
    assert key not in get_default_config()


def test_saved_turn_controls_do_not_disrupt_active_settings(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    values = {
        'evaluator_enabled': True,
        'evaluator_nudge_max': 99,
        'planner_enabled': False,
        'agentic_max_turns': get_default_config()['agentic_max_turns'] + 1,
        'fast_model': 'local-test-fast',
    }
    path.write_text(json.dumps(values))
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))

    settings = load_settings()

    assert settings.planner_enabled == values['planner_enabled']
    assert settings.agentic_max_turns == values['agentic_max_turns']
    assert settings.fast_model == values['fast_model']
    saved = json.loads(path.read_text())
    assert all(saved[key] == value for key, value in values.items())
