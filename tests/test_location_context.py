from unittest.mock import patch
from jarvis.reply.engine import run_reply_engine
from jarvis.utils.location import (
    _get_external_ip_automatically,
)


def test_get_location_context_disabled_flag(mock_config, db, dialogue_memory):
    mock_config.location_enabled = False
    mock_config.db_path = db.db_path
    with patch("jarvis.reply.engine.select_tools", return_value=[]), \
         patch("jarvis.reply.engine.plan_query", return_value=["Reply to the user."]), \
         patch("jarvis.reply.engine.get_location_context_with_timezone") as location_lookup, \
         patch("jarvis.reply.engine.chat_with_messages", return_value={"message": {"content": "Hello."}}) as chat:
        reply = run_reply_engine(db, mock_config, None, "test message", dialogue_memory)

    assert reply == "Hello."
    location_lookup.assert_not_called()
    assert "Location: Disabled" in chat.call_args.kwargs["messages"][0]["content"]


def test_auto_detect_falls_back_to_opendns_when_upnp_and_socket_fail():
    """OpenDNS DNS query is the final fallback in auto-detection (step 3)."""
    with patch("jarvis.utils.location._get_external_ip_via_upnp", return_value=None), \
         patch("jarvis.utils.location._get_external_ip_via_socket", return_value=None), \
         patch("jarvis.utils.location._resolve_public_ip_via_opendns", return_value="93.184.216.34") as mock_dns:
        result = _get_external_ip_automatically()
        mock_dns.assert_called_once()
        assert result == "93.184.216.34"


def test_auto_detect_skips_opendns_when_upnp_succeeds():
    """OpenDNS is not called when UPnP already returned a public IP."""
    with patch("jarvis.utils.location._get_external_ip_via_upnp", return_value="203.0.113.1"), \
         patch("jarvis.utils.location._resolve_public_ip_via_opendns") as mock_dns:
        result = _get_external_ip_automatically()
        mock_dns.assert_not_called()
        assert result == "203.0.113.1"


def test_auto_detect_skips_opendns_when_socket_succeeds():
    """OpenDNS is not called when socket heuristic already returned a public IP."""
    with patch("jarvis.utils.location._get_external_ip_via_upnp", return_value=None), \
         patch("jarvis.utils.location._get_external_ip_via_socket", return_value="198.51.100.5"), \
         patch("jarvis.utils.location._resolve_public_ip_via_opendns") as mock_dns:
        result = _get_external_ip_automatically()
        mock_dns.assert_not_called()
        assert result == "198.51.100.5"


def test_auto_detect_returns_none_when_all_methods_fail():
    """Returns None when UPnP, socket, and OpenDNS all fail."""
    with patch("jarvis.utils.location._get_external_ip_via_upnp", return_value=None), \
         patch("jarvis.utils.location._get_external_ip_via_socket", return_value=None), \
         patch("jarvis.utils.location._resolve_public_ip_via_opendns", return_value=None):
        result = _get_external_ip_automatically()
        assert result is None


def test_auto_detect_rejects_private_ip_from_opendns():
    """Private IPs from OpenDNS are rejected (not returned as valid)."""
    with patch("jarvis.utils.location._get_external_ip_via_upnp", return_value=None), \
         patch("jarvis.utils.location._get_external_ip_via_socket", return_value=None), \
         patch("jarvis.utils.location._resolve_public_ip_via_opendns", return_value="192.168.1.1"):
        result = _get_external_ip_automatically()
        assert result is None
