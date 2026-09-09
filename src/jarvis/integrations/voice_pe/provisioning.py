"""Onboarding paths for the Voice PE integration.

Two flows, both without a Home Assistant server and without touching the
firmware:
- Improv BLE for Wi-Fi on a fresh retail unit (optional ``bleak`` layer);
- runtime Noise PSK provisioning inside the device provisioning window.

Only the well-known all-zero PSK is used as the handshake key during the
provisioning window, then a fresh random 32-byte key is installed through
``noise_encryption_set_key``. Pairing is idempotent: an already-provisioned
device keeps its key and the second pass re-reads the stored value.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import VoicePEConfig

try:  # pragma: no cover - trivial import shim
    from aioesphomeapi import ZERO_NOISE_PSK
except ImportError:  # pragma: no cover
    ZERO_NOISE_PSK = "0" * 64  # type: ignore[assignment]

#: Improv Serial v1.1 over the BLE UART service.
IMPROV_SERVICE_UUID = "0000ff00-0000-0000-0000-000000000000"
IMPROV_CHARACTERISTIC_UUID = "0000ff01-0000-0000-0000-000000000000"
IMPROV_COMMAND_UUID = "0000ff02-0000-0000-0000-000000000000"

#: Improv RPC commands.
CMD_GET_CURRENT_STATE = 1
CMD_STRING = 2
CMD_WIFI_SSID = 3
CMD_WIFI_CREDENTIAL = 4

#: Improv RPC response types.
RESP_CURRENT_STATE = 1
RESP_ERROR = 2
RESP_STRING = 3

STATE_AWAITING_ENABLE = "awaiting_enable"
STATE_NETWORK_CONNECTING = "network_connecting"
STATE_HAVE_IP = "have_ip"
STATE_NO_NETWORK = "no_network"

#: Provisioning + NIFD window timeouts (seconds).
PROVISION_TIMEOUT_S = 15.0
NIFD_TIMEOUT_S = 12.0


def _parse_improv_packet(raw: bytes) -> tuple[int, bytes]:
    """Split one Improv packet into ``(type, payload)``.

    Layout: ``[total length][rpc version][type][payload...]``. The whole
    packet never exceeds 64 bytes on ESPHome devices.
    """
    data = bytes(raw or b"")
    if len(data) < 3:
        return 0, b""
    # Byte 0 is the length of the rest of the packet.
    payload_start = 3
    return data[2], data[payload_start:]


def improv_state_name(value: int) -> str:
    """Human-readable Improv state for a numeric current-state value."""
    return {
        1: STATE_NO_NETWORK,
        2: STATE_AWAITING_ENABLE,
        3: STATE_NETWORK_CONNECTING,
        4: STATE_HAVE_IP,
    }.get(int(value), "unknown")


async def improv_provision(
    ssids_and_credentials: list[tuple[str, str]],
    *,
    address: Optional[str] = None,
) -> dict:
    """Provision Wi-Fi over Improv BLE. Returns a small status dict.

    ``bleak`` is the only extra dependency and stays behind this optional
    import; without it the caller shows the manual onboarding hint.
    """
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as err:
        return {"ok": False, "error": f"bleak not installed: {err}"}

    target = address
    if target is None:
        try:
            for device in await BleakScanner.discover(timeout=NIFD_TIMEOUT_S):
                name = (device.name or "").lower()
                if name.startswith("home-assistant-voice") or "home-assistant-voice" in name:
                    target = device.address
                    break
        except Exception as err:
            return {"ok": False, "error": f"scan failed: {err}"}
    if target is None:
        return {"ok": False, "error": "no Improv device found"}

    state = {"ok": False, "mac_suffix": target[-6:], "state": "unknown"}
    try:
        async with BleakClient(target, timeout=NIFD_TIMEOUT_S) as client:
            services = await client.get_services()
            uuids = {
                svc.uuid.lower(): svc for svc in services if svc.uuid.lower() != "1800"
            }
            chars = {
                char.uuid.lower(): char
                for svc in uuids.values()
                for char in svc.characteristics
            }
            read_uuid = IMPROV_CHARACTERISTIC_UUID if IMPROV_CHARACTERISTIC_UUID in chars else None
            write_uuid = IMPROV_COMMAND_UUID if IMPROV_COMMAND_UUID in chars else None
            if read_uuid is None or write_uuid is None:
                return {"ok": False, "error": "improv characteristics missing"}

            async def _command(payload: bytes) -> None:
                packet = bytes([len(payload) + 1, 1]) + payload
                await client.write_gatt_char(write_uuid, packet, response=True)

            async def _read() -> tuple[int, bytes]:
                raw = await client.read_gatt_char(read_uuid)
                return _parse_improv_packet(bytes(raw))

            await _command(bytes([CMD_GET_CURRENT_STATE]))
            rtype, rpayload = await _read()
            if rtype == RESP_ERROR:
                return {"ok": False, "error": f"improv error {rpayload[:1].hex()}"}
            if rtype == RESP_CURRENT_STATE and rpayload:
                state["state"] = improv_state_name(rpayload[0])

            for ssid, credential in ssids_and_credentials:
                if ssid:
                    await _command(bytes([CMD_WIFI_SSID]) + ssid.encode("utf-8"))
                if credential:
                    await _command(bytes([CMD_WIFI_CREDENTIAL]) + credential.encode("utf-8"))
                await _command(bytes([CMD_GET_CURRENT_STATE]))
                rtype, rpayload = await _read()
                if rtype == RESP_CURRENT_STATE and rpayload:
                    state["state"] = improv_state_name(rpayload[0])
                    if state["state"] == STATE_HAVE_IP:
                        state["ok"] = True
                        return state

            # NIFD: wait for the device to report an IP inside its window.
            for _ in range(int(PROVISION_TIMEOUT_S)):
                await _command(bytes([CMD_GET_CURRENT_STATE]))
                try:
                    rtype, rpayload = await _read()
                except Exception:
                    break
                if rtype == RESP_CURRENT_STATE and rpayload:
                    state["state"] = improv_state_name(rpayload[0])
                    if state["state"] == STATE_HAVE_IP:
                        state["ok"] = True
                        break
    except Exception as err:
        return {"ok": False, "error": str(err), "mac_suffix": target[-6:]}

    return state


def is_provisionable(device_info: Any) -> bool:
    """True while the node accepts the zero-PSK provisioning handshake."""
    return bool(getattr(device_info, "api_encryption_provisionable", False))


async def provision_noise_key(client, device_info: Any) -> tuple[bool, Optional[str]]:
    """Install a fresh Noise PSK. Returns ``(success, base64 key)``.

    Idempotent: an already keyed node (``api_encryption_supported`` without a
    provisioning window) reports back without a new key.
    """
    from .config import encode_psk, new_noise_key

    if not is_provisionable(device_info):
        return False, None

    key = new_noise_key()
    try:
        success = await client.noise_encryption_set_key(key)
    except Exception:
        return False, None
    if not success:
        return False, None
    return True, encode_psk(key)


async def reconnect_with_psk(config: VoicePEConfig, host: str, port: int, psk: str):
    """Second connection attempt using the freshly installed key."""
    from .models import make_client

    client = make_client(
        host,
        port,
        psk,
        device_name=config.device_name,
        mac=(config.mac_address or "").replace(":", "").lower() or None,
    )
    await client.connect()
    return client
