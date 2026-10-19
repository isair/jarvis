"""Exercise deferred close-time metadata with an owned activity snapshot."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def verify_publication(data):
    source = Path(__file__).resolve().parents[2] / 'src' / 'jarvis' / 'activity.py'
    spec = importlib.util.spec_from_file_location('_owned_activity_publication', source)
    activity = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = activity
    spec.loader.exec_module(activity)
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    state_file = data / 'state'
    original = activity.tempfile.NamedTemporaryFile

    class FinaliseOnClose:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self.stream.__enter__()
        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            path = Path(self.stream.name)
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
            return result

    with patch.object(activity, '_get_jarvis_state_file', lambda: str(state_file)), \
         patch.object(activity.tempfile, 'NamedTemporaryFile',
                      lambda *args, **kwargs: FinaliseOnClose(original(*args, **kwargs))):
        manager = activity.JarvisStateManager()
        manager.set_state(activity.JarvisState.IDLE)
        owner = manager.acquire_state(activity.JarvisState.SPEAKING)
        assert manager.state is activity.JarvisState.SPEAKING
        assert manager.release_state(owner), 'Unchanged publication lost ownership when the writer closed'
        assert manager.state is activity.JarvisState.IDLE
        assert state_file.read_text() == activity.JarvisState.IDLE.value
        assert sorted(path.name for path in data.iterdir()) == ['state']


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='jarvis-activity-publication-') as owned:
        verify_publication(Path(owned) / 'data')
    print('✅ Activity ownership survives metadata finalised when its writer closes')
