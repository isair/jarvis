"""Recommendation advice follows each model's resolved role and provider."""
import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def settings_factory(tmp_path, monkeypatch):
    from jarvis.config import load_settings
    path = tmp_path / 'configuration.json'
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))
    def load(**values):
        path.write_text(json.dumps({'_config_version': 3, **values}))
        return load_settings()
    return load


@pytest.mark.parametrize('key,role', [
    ('ollama_chat_model', 'Chat'), ('fast_model', 'Fast'),
    ('ollama_embed_model', 'Embedding'),
])
def test_unlisted_selection_is_reported_for_its_role(settings_factory, key, role):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(**{key: 'synthetic-unlisted:1'})) == {role: 'synthetic-unlisted:1'}


def test_all_unlisted_roles_are_reported_together(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        ollama_chat_model='synthetic-chat', fast_model='synthetic-fast',
        ollama_embed_model='synthetic-embedding',
    )) == {'Chat': 'synthetic-chat', 'Fast': 'synthetic-fast', 'Embedding': 'synthetic-embedding'}


def test_default_selections_are_recommended(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory()) == {}


def test_chat_recommendation_does_not_qualify_a_fast_model(settings_factory):
    from jarvis.config import SUPPORTED_CHAT_MODELS, SUPPORTED_FAST_MODEL_IDS
    from desktop_app.app import check_model_support
    model = next(model for model in SUPPORTED_CHAT_MODELS if model not in SUPPORTED_FAST_MODEL_IDS)
    assert check_model_support(settings_factory(fast_model=model)) == {'Fast': model}


def test_local_embeddings_are_checked_with_external_chat(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model='external-chat',
        embedding_provider='ollama', ollama_embed_model='synthetic-embedding',
    )) == {'Embedding': 'synthetic-embedding'}


def test_external_provider_names_are_not_checked_against_ollama_ids(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model='external-chat',
        fast_model='external-fast', embedding_model='external-embedding',
    )) == {}


def test_an_unlisted_family_variant_is_not_qualified(settings_factory):
    from jarvis.config import DEFAULT_CHAT_MODEL
    from desktop_app.app import check_model_support
    model = DEFAULT_CHAT_MODEL.split(':')[0] + ':synthetic-unlisted-variant'
    assert check_model_support(settings_factory(ollama_chat_model=model)) == {'Chat': model}


def test_embedding_latest_alias_preserves_default_identity(settings_factory):
    from jarvis.config import get_default_config
    from desktop_app.app import check_model_support
    model = get_default_config()['ollama_embed_model'] + ':latest'
    assert check_model_support(settings_factory(ollama_embed_model=model)) == {}


@pytest.mark.parametrize('role,catalogue,key', [
    ('Chat', 'SUPPORTED_CHAT_MODELS', 'ollama_chat_model'),
    ('Fast', 'SUPPORTED_FAST_MODEL_IDS', 'fast_model'),
    ('Embedding', 'SUPPORTED_EMBEDDING_MODEL_IDS', 'ollama_embed_model'),
])
def test_every_role_recommendation_is_accepted(settings_factory, role, catalogue, key):
    from jarvis import config
    from desktop_app.app import check_model_support
    for model in getattr(config, catalogue):
        assert role not in check_model_support(settings_factory(**{key: model}))


@pytest.mark.parametrize('button,opens_setup', [
    ('Continue Anyway', False), ('🔧 Open Setup Wizard', True),
])
def test_advice_lists_roles_and_preserves_settings(qapp, settings_factory, button, opens_setup):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QPushButton
    from desktop_app.app import UnsupportedModelDialog, check_model_support
    from jarvis.config import get_config_path
    settings = settings_factory(ollama_chat_model='<b>custom-chat</b>',
                                fast_model='custom-fast', ollama_embed_model='custom-embedding')
    path = get_config_path()
    before = path.read_bytes()
    dialog = UnsupportedModelDialog(check_model_support(settings))
    dialog.show()
    qapp.processEvents()
    descriptions = dialog.findChildren(QLabel)
    for role, model in check_model_support(settings).items():
        label = next(label for label in descriptions if f'{role}: {model}' in label.text())
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert label.isVisible()
    action = next(item for item in dialog.findChildren(QPushButton) if item.text() == button)
    action.click()
    assert not dialog.isVisible()
    assert dialog.open_wizard is opens_setup
    assert path.read_bytes() == before
    dialog.close()
