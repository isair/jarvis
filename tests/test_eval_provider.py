"""Provider probes must distinguish reachable models from unavailable ones."""

from unittest.mock import Mock

from scripts.eval_provider import probe_provider


def _response(code, body):
    response = Mock(status_code=code)
    response.json.return_value = body
    return response


def test_ollama_probe_requires_the_requested_model():
    get = Mock(side_effect=[
        _response(200, {"models": [{"name": "small:latest"}]}),
    ])

    result = probe_provider("http://127.0.0.1:11434", "small:latest", get=get)

    assert result.provider == "ollama"
    assert result.available is True
    assert get.call_args.args[0].endswith("/api/tags")


def test_openai_base_with_v1_is_not_double_versioned():
    get = Mock(side_effect=[
        _response(404, {}),
        _response(200, {"data": [{"id": "local-model"}]}),
    ])

    result = probe_provider("http://127.0.0.1:8000/v1", "local-model", get=get)

    assert result.provider == "openai_compatible"
    assert result.available is True
    assert get.call_args.args[0] == "http://127.0.0.1:8000/v1/models"


def test_reachable_provider_with_missing_model_is_unavailable():
    get = Mock(side_effect=[
        _response(200, {"models": [{"name": "other"}]}),
        _response(404, {}),
    ])

    result = probe_provider("http://127.0.0.1:11434", "wanted", get=get)

    assert result.available is False
    assert result.reason == "model_not_found"


def test_connection_failure_is_unavailable():
    get = Mock(side_effect=ConnectionError("refused"))

    result = probe_provider("http://127.0.0.1:8000", "wanted", get=get)

    assert result.available is False
    assert result.reason == "endpoint_unreachable"
