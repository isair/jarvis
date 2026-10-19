"""Delayed completion restores activity only while its publication is owned."""
import pytest
from jarvis import activity

pytestmark = pytest.mark.unit


@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / 'owned-state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    return activity.JarvisStateManager(), path


def test_live_activity_owner_restores_its_prior_state_once(state):
    manager, path = state
    prior = activity.JarvisState.THINKING
    manager.set_state(prior)
    owner = manager.acquire_state(activity.JarvisState.SPEAKING)
    assert owner is not None
    assert manager.state is activity.JarvisState.SPEAKING
    assert manager.release_state(owner)
    assert manager.state is prior
    assert not manager.release_state(owner)
    assert manager.state is prior


@pytest.mark.parametrize('newer', [activity.JarvisState.LISTENING, activity.JarvisState.THINKING,
                                    activity.JarvisState.DICTATING, activity.JarvisState.DICTATION_PROCESSING,
                                    activity.JarvisState.SPEAKING])
def test_delayed_speech_completion_cannot_replace_a_newer_activity(state, newer):
    manager, path = state
    manager.set_state(activity.JarvisState.IDLE)
    owner = manager.acquire_state(activity.JarvisState.SPEAKING)
    manager.set_state(newer)
    assert not manager.release_state(owner)
    assert manager.state is newer
    assert path.read_text() == newer.value


@pytest.mark.parametrize('newer', [activity.JarvisState.LISTENING, activity.JarvisState.SPEAKING])
def test_another_publishers_activity_invalidates_delayed_release(state, newer):
    manager, path = state
    other = activity.JarvisStateManager()
    manager.set_state(activity.JarvisState.IDLE)
    owner = manager.acquire_state(activity.JarvisState.SPEAKING)
    other.set_state(newer)
    assert not manager.release_state(owner)
    assert manager.state is newer
    assert path.read_text() == newer.value


@pytest.mark.parametrize('protected', [activity.JarvisState.DICTATING, activity.JarvisState.DICTATION_PROCESSING])
def test_activity_acquisition_respects_protected_work(state, protected):
    manager, path = state
    manager.set_state(protected)
    owner = manager.acquire_state(activity.JarvisState.SPEAKING, protected_states={protected})
    assert owner is None
    assert manager.state is protected
    assert path.read_text() == protected.value


def test_failed_activity_publication_does_not_acquire_or_change_state(state, monkeypatch):
    manager, path = state
    prior = activity.JarvisState.IDLE
    manager.set_state(prior)

    def failed(*args, **kwargs):
        raise PermissionError('Owned publication failure')

    monkeypatch.setattr(activity.os, 'replace', failed)
    assert manager.acquire_state(activity.JarvisState.SPEAKING) is None
    assert manager.state is prior
    assert path.read_text() == prior.value


def test_owner_can_restore_an_explicit_completion_state(state):
    manager, path = state
    manager.set_state(activity.JarvisState.THINKING)
    owner = manager.acquire_state(activity.JarvisState.SPEAKING, restore_state=activity.JarvisState.IDLE)
    assert owner is not None
    assert manager.release_state(owner)
    assert manager.state is activity.JarvisState.IDLE


def test_replacement_owner_does_not_restore_an_abandoned_temporary_state(state):
    manager, path = state
    prior = activity.JarvisState.IDLE
    manager.set_state(prior)
    first = manager.acquire_state(activity.JarvisState.SPEAKING)
    replacement = manager.acquire_state(activity.JarvisState.SPEAKING)
    assert not manager.release_state(first)
    assert manager.state is activity.JarvisState.SPEAKING
    assert manager.release_state(replacement)
    assert manager.state is prior
