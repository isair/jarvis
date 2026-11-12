"""The published configuration example follows the canonical portable defaults."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jarvis.config import export_example_config

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def test_published_example_matches_portable_configuration_defaults():
    path = Path(__file__).resolve().parents[1] / "examples" / "config.json"
    assert json.loads(path.read_text(encoding="utf-8")) == export_example_config()


def test_generator_writes_repeatable_portable_defaults(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_config_examples.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("config_example_generator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "project_root", tmp_path)
    examples = tmp_path / "examples"
    examples.mkdir()
    path = examples / "config.json"
    path.write_text('{"unmanaged": "Stale example"}', encoding="utf-8")
    module.generate_config_example()
    assert json.loads(path.read_text(encoding="utf-8")) == export_example_config()
    first = path.read_bytes()
    module.generate_config_example()
    assert path.read_bytes() == first
