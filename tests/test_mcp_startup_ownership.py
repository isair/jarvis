"""Pending MCP startup remains owned without blocking unrelated servers."""
import asyncio
import concurrent.futures
import itertools
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from jarvis.tools.external import mcp_client, mcp_runtime

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def startup_runtime(monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()
    opened = []
    connection_closed = defaultdict(threading.Event)
    session_closed = defaultdict(threading.Event)
    identities = itertools.count()
    threads = []

    class Connection:
        def __init__(self, name):
            self.name = name
            self.identity = next(identities)

        async def __aenter__(self):
            opened.append(self.name)
            return SimpleNamespace(name=self.name, identity=self.identity), object()

        async def __aexit__(self, *args):
            await asyncio.sleep(0.01)
            connection_closed[self.name].set()

    class Session:
        def __init__(self, read, write):
            self.stream = read

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            if self.stream.name == 'retiring':
                cleanup_entered.set()
                while not cleanup_release.is_set():
                    try:
                        await asyncio.sleep(0.005)
                    except asyncio.CancelledError:
                        continue
            await asyncio.sleep(0.01)
            session_closed[self.stream.name].set()

        async def initialize(self):
            if self.stream.name == 'slow':
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.005)

        async def list_tools(self):
            return self.stream.name, self.stream.identity

    monkeypatch.setattr(mcp_client.MCPClient, '_connect_stdio', lambda self, cfg: Connection(cfg['command']))
    monkeypatch.setattr(mcp_client, 'ClientSession', Session)
    monkeypatch.setattr(mcp_runtime, '_SHUTDOWN_GRACE_SEC', 0.01)
    monkeypatch.setattr(mcp_runtime, '_SHUTDOWN_CLEANUP_SEC', 0.5)
    monkeypatch.setattr(mcp_runtime, '_SHUTDOWN_THREAD_JOIN_SEC', 1)
    runtime = mcp_runtime._PersistentMCPRuntime()

    def background(fn):
        result = {}
        done = threading.Event()
        def run():
            try:
                result['value'] = fn()
            except BaseException as error:
                result['error'] = error
            finally:
                done.set()
        thread = threading.Thread(target=run, daemon=True)
        threads.append(thread)
        thread.start()
        return done, result

    state = SimpleNamespace(
        release=release, entered=entered, opened=opened,
        cleanup_entered=cleanup_entered, cleanup_release=cleanup_release,
        connection_closed=connection_closed, session_closed=session_closed,
        background=background,
    )
    try:
        yield runtime, state
    finally:
        release.set()
        cleanup_release.set()
        runtime.shutdown()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive(), 'An owned MCP caller failed to drain'
        runtime._thread.join(timeout=2)
        assert not runtime._thread.is_alive(), 'The owned MCP loop failed to drain'


def _cfg(name):
    return {'transport': 'stdio', 'command': name}


def test_slow_startup_does_not_block_a_ready_server(startup_runtime):
    runtime, state = startup_runtime
    ready = runtime.list_tools('fast', _cfg('fast'))
    slow_done, _ = state.background(lambda: runtime.list_tools('slow', _cfg('slow')))
    assert state.entered.wait(1)
    fast_done, result = state.background(lambda: runtime.list_tools('fast', _cfg('fast')))
    assert fast_done.wait(1), 'A ready server waited for another server to initialise'
    assert result.get('value') == ready
    assert not slow_done.is_set()


def test_same_server_startup_keeps_one_persistent_connection(startup_runtime):
    runtime, state = startup_runtime
    first_done, first = state.background(lambda: runtime.list_tools('slow', _cfg('slow')))
    assert state.entered.wait(1)
    second_done, second = state.background(lambda: runtime.list_tools('slow', _cfg('slow')))
    state.release.set()
    assert first_done.wait(1) and second_done.wait(1)
    assert first.get('value') == second.get('value')
    assert 'error' not in first and 'error' not in second
    assert state.opened == ['slow']


def test_shutdown_cancels_a_pending_handshake_and_finishes_cleanup(startup_runtime):
    runtime, state = startup_runtime
    caller_done, _ = state.background(lambda: runtime.list_tools('slow', _cfg('slow')))
    assert state.entered.wait(1)
    closed, result = state.background(runtime.shutdown)
    assert closed.wait(1), 'Shutdown waited for the server handshake instead of cancelling it'
    assert 'error' not in result
    assert state.connection_closed['slow'].is_set()
    assert state.session_closed['slow'].is_set()
    assert caller_done.wait(1), 'The interrupted startup caller remained blocked'


def test_shutdown_before_start_prevents_a_late_connection(startup_runtime, monkeypatch):
    runtime, state = startup_runtime
    admitted = threading.Event()
    start = mcp_runtime._ServerWorker.start
    def paused_start(worker):
        admitted.set()
        assert state.release.wait(3)
        return start(worker)
    monkeypatch.setattr(mcp_runtime._ServerWorker, 'start', paused_start)
    caller_done, result = state.background(lambda: runtime.list_tools('slow', _cfg('slow')))
    assert admitted.wait(1)
    closed, close_result = state.background(runtime.shutdown)
    assert closed.wait(1), 'An admitted startup was hidden from shutdown'
    assert 'error' not in close_result
    state.release.set()
    assert caller_done.wait(1)
    assert 'error' in result
    assert not state.opened, 'A server connection was created after shutdown'


def test_failed_startup_releases_its_connection_before_returning(startup_runtime, monkeypatch):
    runtime, state = startup_runtime
    monkeypatch.setattr(mcp_runtime, '_SETUP_TIMEOUT_SEC', 0.05)
    with pytest.raises(concurrent.futures.TimeoutError):
        runtime.list_tools('slow', _cfg('slow'))
    assert state.entered.is_set()
    assert state.session_closed['slow'].is_set(), 'Failed startup left an unowned session'
    assert state.connection_closed['slow'].is_set(), 'Failed startup left an unowned connection'


def test_late_failure_does_not_discard_a_replacement_sessions_state(startup_runtime, monkeypatch):
    runtime, state = startup_runtime
    failed = threading.Event()
    identities = itertools.count()

    class Worker:
        def __init__(self, loop, name, cfg):
            self._server_name = name
            self.config = dict(cfg)
            self.alive = True
            self.identity = next(identities)
            self.value = None

        @property
        def stopped(self):
            return not self.alive

        def start(self):
            pass

        def shutdown(self):
            self.alive = False

        def invoke(self, tool, arguments, timeout):
            if self.identity == 0:
                self.alive = False
                failed.set()
                assert state.release.wait(3)
                raise mcp_runtime._WorkerDeadError('Owned simulated session loss')
            if tool == 'remember':
                self.value = arguments['value']
            return self.value

    monkeypatch.setattr(mcp_runtime, '_ServerWorker', Worker)
    done, first = state.background(lambda: runtime.invoke('same', _cfg('same'), 'read', {}))
    assert failed.wait(1)
    assert runtime.invoke('same', _cfg('same'), 'remember', {'value': 'owned session state'}) == 'owned session state'
    state.release.set()
    assert done.wait(1)
    assert first.get('value') == 'owned session state', 'A late failure discarded the replacement session'


def test_replacement_waits_for_the_previous_sessions_resources(startup_runtime, monkeypatch):
    runtime, state = startup_runtime
    monkeypatch.setattr(mcp_runtime, '_SHUTDOWN_CLEANUP_SEC', 0.05)
    assert runtime.list_tools('target', _cfg('retiring'))[0] == 'retiring'
    monkeypatch.setattr(mcp_runtime, 'get_runtime', lambda: runtime)
    client = mcp_client.MCPClient({'target': _cfg('replacement')})
    with pytest.raises(mcp_client.MCPServerSessionError):
        client.list_tools('target')
    assert state.cleanup_entered.is_set()
    assert state.opened == ['retiring'], 'Replacement overlapped the previous session cleanup'
    state.cleanup_release.set()
    assert state.connection_closed['retiring'].wait(1)
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), runtime._loop).result(timeout=1)
    assert runtime.list_tools('target', _cfg('replacement'))[0] == 'replacement'


@pytest.mark.parametrize('retirement', ['idle', 'shutdown'])
def test_retired_session_rejects_late_request_without_waiting_for_timeout(startup_runtime, monkeypatch, retirement):
    runtime, state = startup_runtime
    cfg = dict(_cfg('fast'), idle_timeout_sec=0.1)
    initial = runtime.list_tools('same', cfg)
    admitted = threading.Event()
    schedule = runtime._loop.call_soon_threadsafe
    caller = {}
    monkeypatch.setattr(mcp_runtime, 'get_runtime', lambda: runtime)
    client = mcp_client.MCPClient({'same': cfg})

    def pause_queue_admission(callback, *args, **kwargs):
        if threading.get_ident() == caller.get('identity') and not admitted.is_set():
            admitted.set()
            assert state.release.wait(3)
        return schedule(callback, *args, **kwargs)

    monkeypatch.setattr(runtime._loop, 'call_soon_threadsafe', pause_queue_admission)
    def request():
        caller['identity'] = threading.get_ident()
        if retirement == 'shutdown':
            return client.list_tools('same')
        return runtime.list_tools('same', cfg, timeout=2)

    done, result = state.background(request)
    assert admitted.wait(1)
    if retirement == 'shutdown':
        runtime.shutdown()
    else:
        assert state.connection_closed['fast'].wait(1)
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), runtime._loop).result(timeout=1)
    state.release.set()
    assert done.wait(0.5), 'A retired session delayed retry until the request timeout'
    if retirement == 'shutdown':
        assert isinstance(result.get('error'), mcp_client.MCPServerSessionError)
    else:
        assert 'error' not in result
        assert result['value'][0] == initial[0]
        assert result['value'][1] != initial[1], 'The request did not reach a fresh session'
