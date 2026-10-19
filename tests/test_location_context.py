"""Location context honours user controls and the ordered IP fallback chain."""

import pytest

from jarvis.reply import engine
from jarvis.utils import location

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def test_disabled_location_retains_time_without_looking_up_location(mock_config, monkeypatch):
    mock_config.location_enabled = False
    mock_config.location_ip_address = "93.184.216.34"

    def forbidden_lookup(**kwargs):
        raise AssertionError("Disabled location attempted a lookup")

    monkeypatch.setattr(engine, "get_location_context_with_timezone", forbidden_lookup)
    monkeypatch.setattr(engine, "format_time_context", lambda zone: "System time" if zone is None else "Unexpected timezone")
    assert engine._live_time_location_string(mock_config) == "Current local time: System time. Saved event dates must match the requested calendar date. Location: Disabled"


@pytest.mark.parametrize("timezone", [None, "Europe/London"])
def test_enabled_location_uses_configured_lookup_and_returned_timezone(mock_config, monkeypatch, timezone):
    mock_config.location_enabled = True
    mock_config.location_ip_address = "93.184.216.34"
    mock_config.location_auto_detect = False
    mock_config.location_cgnat_resolve_public_ip = False
    mock_config.location_cache_minutes = 17
    expected = dict(config_ip=mock_config.location_ip_address,
                    auto_detect=mock_config.location_auto_detect,
                    resolve_cgnat_public_ip=mock_config.location_cgnat_resolve_public_ip,
                    location_cache_minutes=mock_config.location_cache_minutes)

    def configured_lookup(**kwargs):
        assert kwargs == expected
        return "Location: Synthetic city", timezone

    monkeypatch.setattr(engine, "get_location_context_with_timezone", configured_lookup)
    monkeypatch.setattr(engine, "format_time_context", lambda zone: "Selected time" if zone == timezone else "Unexpected timezone")
    assert engine._live_time_location_string(mock_config) == "Current local time: Selected time. Saved event dates must match the requested calendar date. Location: Synthetic city"


def test_location_lookup_failure_omits_context_without_interrupting_reply(mock_config, monkeypatch):
    mock_config.location_enabled = True

    def unavailable_lookup(**kwargs):
        raise OSError("Synthetic unavailable lookup")

    monkeypatch.setattr(engine, "get_location_context_with_timezone", unavailable_lookup)
    assert engine._live_time_location_string(mock_config) == ""


@pytest.mark.parametrize("upnp,socket,dns,expected", [
    ("93.184.216.34", None, None, "93.184.216.34"),
    (None, "93.184.216.35", None, "93.184.216.35"),
    (None, None, "93.184.216.36", "93.184.216.36"),
    (None, None, None, None),
    (None, None, "192.168.1.1", None),
])
def test_ip_detection_prefers_local_methods_and_rejects_private_dns_results(monkeypatch, upnp, socket, dns, expected):
    def socket_lookup():
        assert upnp is None, "UPnP success must bypass socket detection"
        return socket

    def dns_lookup():
        assert upnp is None and socket is None, "Local success must bypass external DNS"
        return dns

    monkeypatch.setattr(location, "_get_external_ip_via_upnp", lambda: upnp)
    monkeypatch.setattr(location, "_get_external_ip_via_socket", socket_lookup)
    monkeypatch.setattr(location, "_resolve_public_ip_via_opendns", dns_lookup)
    assert location._get_external_ip_automatically() == expected
