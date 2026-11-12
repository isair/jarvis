"""The local memory browser rejects foreign web origins and rebinding hosts."""

import sqlite3

import pytest

from desktop_app import memory_viewer


pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    connection.executescript('''
        CREATE TABLE conversation_summaries (
            id INTEGER PRIMARY KEY, date_utc TEXT, ts_utc TEXT, summary TEXT,
            topics TEXT, source_app TEXT
        );
        INSERT INTO conversation_summaries VALUES
            (1, '2026-01-01', '2026-01-01T12:00:00Z', 'Private synthetic memory', '', 'test');
    ''')
    monkeypatch.setattr(memory_viewer, '_db_conn', connection)
    with memory_viewer.app.test_client() as client:
        yield client
    connection.close()


@pytest.mark.parametrize('host', ['attacker.test:5050', '127.0.0.1.attacker.test:5050', 'localhost.attacker.test:5050', '192.168.1.10:5050'])
def test_foreign_host_cannot_read_memories(client, host):
    response = client.get('/api/memories', headers={'Host': host})
    assert response.status_code == 403
    assert b'Private synthetic memory' not in response.data


@pytest.mark.parametrize('headers', [
    {'Origin': 'https://attacker.test'},
    {'Origin': 'null'},
    {'Origin': 'http://localhost:6000'},
    {'Origin': 'http://127.0.0.1:5050'},
    {'Origin': 'http://localhost:0'},
    {'Origin': 'http://localhost:invalid'},
    {'Origin': 'http://['},
    {'Origin': 'https://localhost:5050'},
    {'Origin': 'http://attacker@localhost:5050'},
    {'Referer': 'https://attacker.test/page'},
    {'Sec-Fetch-Site': 'cross-site'},
])
def test_foreign_origin_cannot_read_memories(client, headers):
    response = client.get('/api/memories', base_url='http://localhost:5050', headers=headers)
    assert response.status_code == 403
    assert b'Private synthetic memory' not in response.data


@pytest.mark.parametrize('host', ['localhost:5050', '127.0.0.1:5050', '[::1]:5050'])
@pytest.mark.parametrize('metadata', ['direct', 'origin', 'referer'])
def test_local_browser_can_read_its_memories(client, host, metadata):
    base_url = f'http://{host}'
    headers = {} if metadata == 'direct' else {
        'Origin' if metadata == 'origin' else 'Referer': base_url if metadata == 'origin' else base_url + '/'
    }
    response = client.get('/api/memories', base_url=base_url, headers=headers)
    assert response.status_code == 200
    assert response.get_json()['memories'][0]['summary'] == 'Private synthetic memory'


@pytest.mark.parametrize('scheme,port', [('http', 80), ('https', 443)])
def test_explicit_default_port_matches_local_origin(client, scheme, port):
    response = client.get('/api/memories', base_url=f'{scheme}://localhost',
                          headers={'Origin': f'{scheme}://localhost:{port}'})
    assert response.status_code == 200


@pytest.mark.parametrize('method,path', [('post', '/api/graph/node'), ('put', '/api/graph/node/user'), ('delete', '/api/graph/node/user')])
def test_foreign_browser_cannot_reach_mutating_routes(client, monkeypatch, method, path):
    def forbidden_store():
        pytest.fail('Foreign request reached the memory store')
    monkeypatch.setattr(memory_viewer, 'get_graph_store', forbidden_store)
    response = getattr(client, method)(path, json={'data': 'changed'}, headers={'Origin': 'https://attacker.test'})
    assert response.status_code == 403
