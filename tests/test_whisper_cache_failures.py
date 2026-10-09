"""Download failures expose actionable resource problems without private paths."""
import errno

import pytest

from jarvis.listening import model_download

pytestmark = pytest.mark.unit


def resource_error(code=None, winerror=None):
    error = OSError(code, 'Δεν είναι δυνατή η πρόσβαση', '/private/fixture-cache')
    if winerror is not None:
        error.winerror = winerror
    return error


@pytest.mark.parametrize('error,category', [
    (resource_error(errno.ENOSPC), 'disk_space'),
    pytest.param(resource_error(getattr(errno, 'EDQUOT', 0)), 'disk_space',
                 marks=pytest.mark.skipif(not hasattr(errno, 'EDQUOT'), reason='No POSIX disk quota errno')),
    (resource_error(winerror=112), 'disk_space'),
    (resource_error(errno.EACCES), 'cache_access'),
    (resource_error(errno.EPERM), 'cache_access'),
    (resource_error(errno.EROFS), 'cache_access'),
    (resource_error(winerror=5), 'cache_access'),
    (resource_error(winerror=1314), 'cache_access'),
    (resource_error(errno.EIO), 'download'),
])
def test_worker_preserves_resource_failure_category_without_private_paths(monkeypatch, error, category):
    wrapped = RuntimeError('snapshot unavailable')
    wrapped.__cause__ = error
    def fail_download(*args, **kwargs):
        raise wrapped
    monkeypatch.setattr('faster_whisper.utils.download_model', fail_download)
    class Connection:
        result = None
        def send(self, result):
            self.result = result
        def close(self):
            pass
    connection = Connection()
    model_download._download_worker(connection, 'small')
    assert connection.result[0:2] == ('error', category)
    assert '/private/' not in str(connection.result)


@pytest.mark.parametrize('category,actions', [
    ('disk_space', ('free', 'space', 'smaller', 'settings', 'restart jarvis')),
    ('cache_access', ('cache', 'permissions', 'security software', 'restart jarvis')),
])
def test_resource_failure_explains_recovery_and_keeps_existing_cache(monkeypatch, tmp_path, capsys, category, actions):
    cached = tmp_path / 'model-cache' / 'model.bin'
    cached.parent.mkdir()
    cached.write_bytes(b'partial model weights')
    def no_complete_cache(*args, **kwargs):
        return str(cached.parent)
    monkeypatch.setattr('faster_whisper.utils.download_model', no_complete_cache)
    def fail_worker(*args):
        raise model_download.ModelDownloadError(category, 'OSError')
    monkeypatch.setattr(model_download, '_run_download_worker', fail_worker)
    with pytest.raises(model_download.ModelDownloadError) as failure:
        model_download.prepare_faster_whisper_model('small')
    assert failure.value.category == category
    assert cached.read_bytes() == b'partial model weights'
    output = capsys.readouterr().out.lower()
    assert all(action in output for action in actions)
    assert 'cached files' in output and 'kept' in output
