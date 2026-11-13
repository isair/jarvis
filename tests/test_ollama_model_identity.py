"""Model availability and advice preserve complete provider identities."""

import json

import pytest
import requests

from tests.test_performance_sample_validity import benchmark

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('provider,selected,recommended,unlisted', [
    ('ollama', 'registry.invalid:5000/team/assistant', 'registry.invalid:5000/team/assistant:latest', False),
    ('ollama', 'registry.invalid:5000/team/assistant:latest', 'registry.invalid:5000/team/assistant', False),
    ('ollama', 'registry.invalid:5000/team/assistant:small', 'registry.invalid:5000/team/assistant:wide', True),
    ('openai_compatible', 'registry.invalid:5000/team/assistant', 'registry.invalid:5000/team/assistant:latest', True),
])
def test_recommendation_catalogue_respects_provider_identity(
        monkeypatch, mock_config, provider, selected, recommended, unlisted):
    from desktop_app import app
    mock_config.llm_provider = provider
    mock_config.llm_chat_model = selected
    mock_config.fast_model = 'owned-fast:small'
    mock_config.embedding_model = 'owned-embedding'
    catalogue = {
        'Chat': {recommended}, 'Fast': {mock_config.fast_model},
        'Embedding': {mock_config.embedding_model},
    }
    monkeypatch.setattr(app, 'get_model_recommendations', lambda provider: catalogue)
    assert app.check_model_support(mock_config) == ({'Chat': selected} if unlisted else {})


@pytest.mark.parametrize('required,installed,available', [
    ('owned-chat:wide', 'owned-chat:small', False),
    ('registry.invalid:5000/team/assistant:wide',
     'registry.invalid:5000/team/assistant:small', False),
    ('owned-chat:latest', 'not-owned-chat:latest', False),
    ('registry.invalid:5000/team/assistant',
     'registry.invalid:5000/team/assistant:latest', True),
    ('registry.invalid:5000/team/assistant:latest',
     'registry.invalid:5000/team/assistant', True),
    ('owned-chat:wide', 'owned-chat:wide', True),
])
def test_benchmark_requires_the_selected_model_variant(
        benchmark, monkeypatch, required, installed, available):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({'models': [{'name': installed}]}).encode()
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: response)
    monkeypatch.setattr(benchmark, 'PERF_MODEL', required)
    assert benchmark._ollama_reachable() is available


def test_wizard_keeps_registry_ports_and_required_model_names(monkeypatch):
    from desktop_app import setup_wizard as wizard
    required = ['registry.invalid:5000/team/needed:latest', 'owned-embedding']
    installed = ['registry.invalid:5000/team/other:latest', 'owned-embedding:latest']
    monkeypatch.setattr(wizard, 'check_ollama_cli', lambda: (True, '/owned/ollama'))
    monkeypatch.setattr(wizard, 'check_ollama_server', lambda: (True, 'owned-version'))
    monkeypatch.setattr(wizard, 'get_required_models', lambda: required)
    monkeypatch.setattr(wizard, 'check_installed_models', lambda path: installed)
    status = wizard.check_ollama_status()
    assert status.missing_models == required[:1]
    assert status.installed_models == installed


@pytest.mark.parametrize('required,available,missing', [
    (['registry.invalid:5000/team/needed:latest', 'owned-embedding'],
     ['registry.invalid:5000/team/other:latest', 'owned-embedding:latest'],
     ['registry.invalid:5000/team/needed:latest']),
    (['registry.invalid:5000/team/assistant'],
     ['registry.invalid:5000/team/assistant:latest'], []),
    (['registry.invalid:5000/team/assistant:latest'],
     ['registry.invalid:5000/team/assistant'], []),
    (['owned-chat:small', 'owned-chat:wide'], ['owned-chat:small'],
     ['owned-chat:wide']),
    (['team/assistant', 'other/assistant'], ['team/assistant:latest'],
     ['other/assistant']),
    (['owned-chat:latest', 'owned-embedding'], [],
     ['owned-chat:latest', 'owned-embedding']),
    ([], ['owned-chat:latest'], []),
])
def test_missing_models_preserve_complete_identity_and_required_names(
        required, available, missing):
    from jarvis.llm.model_names import missing_ollama_models

    original_required, original_available = required[:], available[:]
    assert missing_ollama_models(iter(required), iter(available)) == missing
    assert required == original_required
    assert available == original_available


@pytest.mark.parametrize('installed_chat,complete', [
    ('registry.invalid:5000/team/assistant:latest', True),
    ('registry.invalid:5000/team/other:latest', False),
])
def test_model_page_reports_actual_missing_model(
        monkeypatch, mock_config, qapp, installed_chat, complete):
    from desktop_app import setup_wizard as wizard

    mock_config.ollama_embed_model = 'owned-embedding'
    monkeypatch.setattr(wizard, 'load_settings', lambda: mock_config)
    monkeypatch.setattr(wizard, 'detect_total_vram_mb', lambda: None)
    window = wizard.SetupWizard()
    try:
        window.ollama_status = wizard.OllamaStatus(installed_models=[
            installed_chat, 'owned-fast:small', 'owned-embedding:latest',
        ])
        page = window.models_page
        page._chat_model = 'registry.invalid:5000/team/assistant'
        page._fast_model = 'owned-fast:small'
        page._update_models_display()
        assert page.isComplete() is complete
        assert page.install_btn.isHidden() is complete
        if complete:
            assert 'All required models are installed' in page.models_label.text()
        else:
            assert page.models_label.text() == f'Missing: X {page._chat_model}'
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
