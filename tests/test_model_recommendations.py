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
    from jarvis.config import DEFAULT_CHAT_MODEL, SUPPORTED_CHAT_MODELS
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model=SUPPORTED_CHAT_MODELS[DEFAULT_CHAT_MODEL]['openai_id'],
        embedding_provider='ollama', ollama_embed_model='synthetic-embedding',
    )) == {'Embedding': 'synthetic-embedding'}


def test_external_unlisted_roles_receive_advice(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model='external-chat',
        fast_model='external-fast', embedding_model='external-embedding',
    )) == {'Chat': 'external-chat', 'Fast': 'external-fast', 'Embedding': 'external-embedding'}


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


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('role,key', [('Chat', 'llm_chat_model'), ('Fast', 'fast_model'), ('Embedding', 'embedding_model')])
def test_every_role_recommendation_is_accepted(settings_factory, provider, role, key):
    from jarvis.config import get_model_recommendations
    from desktop_app.app import check_model_support
    config_key = {'Chat': 'ollama_chat_model', 'Embedding': 'ollama_embed_model'}.get(role, key) if provider == 'ollama' else key
    for model in get_model_recommendations(provider)[role]:
        settings = settings_factory(llm_provider=provider, **{config_key: model})
        assert getattr(settings, key) == model
        assert role not in check_model_support(settings)


@pytest.mark.parametrize('button,opens_setup', [
    ('Continue Anyway', False), ('🔧 Open Setup Wizard', True),
])
@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_advice_lists_roles_and_preserves_settings(qapp, settings_factory, button, opens_setup, provider):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QPushButton
    from desktop_app.app import UnsupportedModelDialog, check_model_support
    from jarvis.config import get_config_path
    settings = settings_factory(llm_provider=provider, llm_chat_model='<b>custom-chat</b>',
                                ollama_chat_model='<b>custom-chat</b>', fast_model='custom-fast',
                                embedding_model='custom-embedding', ollama_embed_model='custom-embedding')
    assert check_model_support(settings) == {'Chat': '<b>custom-chat</b>', 'Fast': 'custom-fast', 'Embedding': 'custom-embedding'}
    path = get_config_path()
    before = path.read_bytes()
    dialog = UnsupportedModelDialog(check_model_support(settings), settings)
    dialog.show()
    qapp.processEvents()
    descriptions = dialog.findChildren(QLabel)
    for role, model in check_model_support(settings).items():
        label = next(label for label in descriptions if f'{role}: {model}' in label.text())
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert label.isVisible()
        assert label.height() >= label.heightForWidth(label.width())
    if provider == 'openai_compatible':
        assert any('Connect & load models' in label.text() and 'custom model names' in label.text() for label in descriptions)
    action = next(item for item in dialog.findChildren(QPushButton) if item.text() == button)
    action.click()
    assert not dialog.isVisible()
    assert dialog.open_wizard is opens_setup
    assert path.read_bytes() == before
    dialog.close()


def test_external_default_chat_family_is_also_fast_recommended(settings_factory):
    from jarvis.config import DEFAULT_CHAT_MODEL, SUPPORTED_CHAT_MODELS, SUPPORTED_OPENAI_EMBEDDING_MODEL_IDS
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model=SUPPORTED_CHAT_MODELS[DEFAULT_CHAT_MODEL]['openai_id'],
        embedding_model=SUPPORTED_OPENAI_EMBEDDING_MODEL_IDS[0],
    )) == {}


def test_external_chat_recommendation_is_not_a_fast_recommendation(settings_factory):
    from jarvis.config import SUPPORTED_CHAT_MODELS, SUPPORTED_FAST_MODEL_IDS
    from desktop_app.app import check_model_support
    model = next(info['openai_id'] for key, info in SUPPORTED_CHAT_MODELS.items() if key not in SUPPORTED_FAST_MODEL_IDS)
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model=model,
        fast_model=model, embedding_provider='ollama',
    )) == {'Fast': model}


def test_external_embeddings_are_checked_with_ollama_chat(settings_factory):
    from desktop_app.app import check_model_support
    assert check_model_support(settings_factory(
        embedding_provider='openai_compatible', embedding_model='unknown-embedder',
    )) == {'Embedding': 'unknown-embedder'}


def test_an_external_server_alias_is_not_model_provenance(settings_factory):
    from desktop_app.app import check_model_support
    from jarvis.config import get_model_recommendations
    assert check_model_support(settings_factory(
        llm_provider='openai_compatible', llm_chat_model='my-gemma-server-alias',
        fast_model=get_model_recommendations('openai_compatible')['Fast'][0], embedding_provider='ollama',
    )) == {'Chat': 'my-gemma-server-alias'}


def test_external_advice_scrolls_without_hiding_actions(qapp, settings_factory):
    from PyQt6.QtWidgets import QScrollArea, QPushButton
    from desktop_app.app import UnsupportedModelDialog, check_model_support
    settings = settings_factory(llm_provider='openai_compatible', llm_chat_model='unknown-chat',
                                fast_model='unknown-fast', embedding_model='unknown-embedder')
    dialog = UnsupportedModelDialog(check_model_support(settings), settings)
    dialog.resize(dialog.minimumWidth(), 320)
    dialog.show()
    qapp.processEvents()
    scroll = dialog.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() > 0
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qapp.processEvents()
    assert scroll.verticalScrollBar().value() == scroll.verticalScrollBar().maximum()
    for button in dialog.findChildren(QPushButton):
        assert button.isVisible()
        assert dialog.rect().contains(button.geometry())
    dialog.close()
