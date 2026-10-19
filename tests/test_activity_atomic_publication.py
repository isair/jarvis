"""An activity observer sees a complete state while another publisher writes."""
import builtins
import io
from pathlib import Path
import threading

import pytest
from jarvis import activity

pytestmark = pytest.mark.unit


def test_observer_keeps_the_published_state_during_another_write(monkeypatch, tmp_path):
    path = tmp_path / 'state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    writer = activity.JarvisStateManager()
    observer = activity.JarvisStateManager()
    writer.set_state(activity.JarvisState.LISTENING)
    assert observer.state is activity.JarvisState.LISTENING
    started, release = threading.Event(), threading.Event()

    class PausedWrite:
        def __init__(self, stream): self.stream = stream
        def __getattr__(self, name): return getattr(self.stream, name)
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def write(self, text):
            if text == activity.JarvisState.THINKING.value:
                started.set()
                assert release.wait(3), 'Owned publication probe was not released'
            return self.stream.write(text)

    def wrap(original):
        def open_owned(file, mode='r', *args, **kwargs):
            stream = original(file, mode, *args, **kwargs)
            if 'w' in mode and isinstance(file, (str, Path)) and (Path(file).parent == tmp_path or Path(file) == tmp_path):
                return PausedWrite(stream)
            return stream
        return open_owned

    monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    monkeypatch.setattr(io, 'open', wrap(io.open))
    worker = threading.Thread(target=writer.set_state, args=(activity.JarvisState.THINKING,))
    worker.start()
    try:
        assert started.wait(2), 'Owned write never reached the test boundary'
        observed = observer.state
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert observed is activity.JarvisState.LISTENING
    assert observer.state is activity.JarvisState.THINKING


def test_publishing_state_does_not_follow_a_shared_path_symlink(monkeypatch, tmp_path):
    victim = tmp_path / 'unrelated'
    victim.write_text('owned unrelated content')
    path = tmp_path / 'state'
    try:
        path.symlink_to(victim)
    except OSError:
        pytest.skip('Symlink creation is unavailable on this platform')
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    manager = activity.JarvisStateManager()
    manager.set_state(activity.JarvisState.LISTENING)
    assert victim.read_text() == 'owned unrelated content'
    assert path.read_text() == activity.JarvisState.LISTENING.value


def test_failed_publication_preserves_the_previous_file_and_cleans_owned_partials(monkeypatch, tmp_path):
    path = tmp_path / 'state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    manager = activity.JarvisStateManager()
    manager.set_state(activity.JarvisState.LISTENING)

    def denied(*args, **kwargs):
        raise PermissionError('Owned synthetic publication rejection')

    monkeypatch.setattr(activity.os, 'replace', denied)
    manager.set_state(activity.JarvisState.THINKING)
    assert path.read_text() == activity.JarvisState.LISTENING.value
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state']


def test_failed_temporary_write_preserves_the_published_state(monkeypatch, tmp_path):
    path = tmp_path / 'state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    manager = activity.JarvisStateManager()
    manager.set_state(activity.JarvisState.LISTENING)
    create = activity.tempfile.NamedTemporaryFile

    class InterruptedWrite:
        def __init__(self, *args, **kwargs): self.pending = create(*args, **kwargs)
        @property
        def name(self): return self.pending.name
        def __enter__(self): return self
        def __exit__(self, *args): self.pending.close()
        def write(self, text):
            self.pending.write(text[:2])
            raise OSError('Owned interrupted write')

    monkeypatch.setattr(activity.tempfile, 'NamedTemporaryFile', InterruptedWrite)
    manager.set_state(activity.JarvisState.THINKING)
    assert path.read_text() == activity.JarvisState.LISTENING.value
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state']


@pytest.mark.parametrize('reject_second', [False, True])
def test_concurrent_publishers_do_not_replace_or_clean_each_others_pending_state(monkeypatch, tmp_path, reject_second):
    path = tmp_path / 'state'
    monkeypatch.setattr(activity, '_get_jarvis_state_file', lambda: str(path))
    first, second = activity.JarvisStateManager(), activity.JarvisStateManager()
    first.set_state(activity.JarvisState.LISTENING)
    replace = activity.os.replace
    ready, release = threading.Event(), threading.Event()

    def publish(source, destination):
        value = Path(source).read_text()
        if value == activity.JarvisState.THINKING.value:
            ready.set()
            assert release.wait(10), 'Owned publication probe was not released'
        elif reject_second and value == activity.JarvisState.SPEAKING.value:
            raise PermissionError('Owned second publication rejection')
        replace(source, destination)

    monkeypatch.setattr(activity.os, 'replace', publish)
    worker = threading.Thread(target=first.set_state, args=(activity.JarvisState.THINKING,))
    worker.start()
    try:
        assert ready.wait(5)
        assert path.read_text() == activity.JarvisState.LISTENING.value
        second.set_state(activity.JarvisState.SPEAKING)
        expected = activity.JarvisState.LISTENING if reject_second else activity.JarvisState.SPEAKING
        assert path.read_text() == expected.value
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert path.read_text() == activity.JarvisState.THINKING.value
    assert sorted(p.name for p in tmp_path.iterdir()) == ['state']
