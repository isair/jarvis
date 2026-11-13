"""Turbo support boundaries shared by setup and the listener."""

import sys
from types import SimpleNamespace

import pytest

from jarvis.listening import whisper_capabilities

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("suffix,supported", [(".dev0", False), ("", True), (".post1", True)])
def test_turbo_support_follows_minimum_engine_version(monkeypatch, suffix, supported):
    version = whisper_capabilities.TURBO_MIN_FASTER_WHISPER_VERSION + suffix
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(__version__=version))
    assert whisper_capabilities.is_faster_whisper_turbo_supported() is supported


@pytest.mark.parametrize("engine", [None, SimpleNamespace(), SimpleNamespace(__version__="invalid")])
def test_unusable_engine_or_version_disables_turbo(monkeypatch, engine):
    monkeypatch.setitem(sys.modules, "faster_whisper", engine)
    assert whisper_capabilities.is_faster_whisper_turbo_supported() is False
