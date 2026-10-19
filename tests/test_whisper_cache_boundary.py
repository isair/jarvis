"""Automatic speech recovery deletes only the configured Hub model cache."""
import pytest

from jarvis.listening.listener import _clear_corrupted_whisper_cache

pytestmark = pytest.mark.unit


def model_files(root, *, snapshot=True):
    model = root / 'models--fixture--speech'
    path = model / 'snapshots' / 'fixture' if snapshot else model / 'user-files'
    path.mkdir(parents=True)
    marker = model / 'model.bin'
    marker.write_bytes(b'synthetic local speech weights')
    return path, marker


def recovery_error(path):
    return f"Unable to open file 'model.bin' in model '{path}'"


@pytest.mark.parametrize('location', ['outside', 'nested', 'not_snapshot'])
def test_recovery_preserves_non_hub_model_files(monkeypatch, tmp_path, location):
    hub = tmp_path / 'hub'
    monkeypatch.setattr('huggingface_hub.constants.HF_HUB_CACHE', str(hub))
    root = {'outside': tmp_path / 'local-models', 'nested': hub / 'local-models',
            'not_snapshot': hub}[location]
    path, marker = model_files(root, snapshot=location != 'not_snapshot')
    assert not _clear_corrupted_whisper_cache(recovery_error(path))
    assert marker.read_bytes() == b'synthetic local speech weights'


def test_recovery_clears_only_affected_configured_hub_model(monkeypatch, tmp_path):
    hub = tmp_path / 'custom-hub'
    monkeypatch.setattr('huggingface_hub.constants.HF_HUB_CACHE', str(hub))
    path, marker = model_files(hub)
    other = hub / 'models--other--speech' / 'model.bin'
    other.parent.mkdir(); other.write_bytes(b'other cached model')
    assert _clear_corrupted_whisper_cache(recovery_error(path))
    assert not marker.exists()
    assert other.read_bytes() == b'other cached model'


def test_recovery_preserves_symlinked_local_model(monkeypatch, tmp_path):
    hub = tmp_path / 'hub'
    hub.mkdir()
    monkeypatch.setattr('huggingface_hub.constants.HF_HUB_CACHE', str(hub))
    local_path, marker = model_files(tmp_path / 'local-models')
    link = hub / local_path.parent.parent.name
    try:
        link.symlink_to(local_path.parent.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('Directory symlinks unavailable')
    snapshot = link / 'snapshots' / local_path.name
    assert not _clear_corrupted_whisper_cache(recovery_error(snapshot))
    assert marker.read_bytes() == b'synthetic local speech weights'
    assert link.is_symlink()
