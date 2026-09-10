"""``jarvis voice-pe ...`` command helpers in the existing hand-rolled style.

The Jarvis CLI reads ``sys.argv`` directly (no argparse), so this module
exposes :func:`handle` returning the process exit code. Output uses an emoji
per line plus indentation for hierarchy, matching the rest of the CLI.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from . import config as pe_config
from .discovery import discover
from .models import VoicePEConfig, make_client

_ACCENT = (0.55, 0.0, 1.0)


def _flag(argv: list[str], name: str) -> Optional[str]:
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def _print_health(snapshot: dict) -> None:
    print("  🛰️ Device", flush=True)
    print(f"     🔗 {snapshot.get('device', '?')} (API {snapshot.get('api_version', '?')})", flush=True)
    print(f"     🟢 state: {snapshot.get('device_state', '?')} / session: {snapshot.get('session_state', '?')}", flush=True)
    print(f"     🎚️  audio queue: {snapshot.get('audio_queue_ms', 0)} ms", flush=True)
    features = snapshot.get("voice_features") or []
    print(f"     🧩 features: {', '.join(features) or 'none'}", flush=True)
    print(
        f"     👂 wake words: {'disabled' if snapshot.get('wake_words_disabled') else 'enabled'}",
        flush=True,
    )
    if snapshot.get("error"):
        print(f"     ⚠️  last error: {snapshot['error']}", flush=True)


def handle(argv: list[str], settings: Any, manager: Any = None) -> int:
    """Run one ``voice-pe`` subcommand. Returns a process exit code.

    ``manager`` is the live :class:`VoicePEManager` when the daemon bundled it;
    a standalone invocation passes a transient one in (see ``run_cli``).
    """
    cfg = manager.config if manager is not None else pe_config.from_settings(settings)

    if not argv:
        print("🎙️ Voice PE commands:", flush=True)
        for line in (
            "jarvis voice-pe discover",
            "jarvis voice-pe pair",
            "jarvis voice-pe list",
            "jarvis voice-pe status <device>",
            "jarvis voice-pe set-led <device> --rgb 8c00ff --brightness 0.66",
            "jarvis voice-pe announce <device> \"text\"",
            "jarvis voice-pe play <device> <url>",
            "jarvis voice-pe pause <device>",
            "jarvis voice-pe resume <device>",
            "jarvis voice-pe volume <device> 0.66",
            "jarvis voice-pe mute <device> on|off",
            "jarvis voice-pe stop <device>",
            "jarvis voice-pe forget <device>",
        ):
            print(f"  ▫️ {line}", flush=True)
        return 0

    command = argv[0]
    rest = argv[1:]

    if command == "discover":
        found = asyncio.run(discover(cfg))
        if not found:
            print("🔍 No Voice PE found on the network", flush=True)
            print("   ▫️ Check power, Wi-Fi and the mute switch position", flush=True)
            return 0
        print(f"🔍 Found {len(found)} Voice PE node(s)", flush=True)
        for entry in found:
            print(f"  🛰️ {entry.get('node_name') or entry.get('host')} ({entry.get('source')})", flush=True)
            print(f"     📡 {entry.get('host')}:{entry.get('port')}", flush=True)
            print(f"     🏷️  MAC {entry.get('mac_address')}", flush=True)
            print(f"     📦 {entry.get('project_name')} v{entry.get('project_version')}", flush=True)
        return 0

    if command == "pair":
        return _pair(cfg)

    if command == "list":
        devices = manager.devices if manager is not None else []
        if not devices:
            print("📋 No Voice PE devices attached", flush=True)
            return 0
        print(f"📋 {len(devices)} device(s)", flush=True)
        for device in devices:
            view = device.ui_view()
            connection = view["connection"]
            features = connection.get("voice_features") or []
            print(f"  🛰️ {view['config']['host']} - {device.identity.get('friendly_name', '')}", flush=True)
            print(
                f"     🎙️ {view['audio']['input_channel']} ch, "
                f"TTS {connection['session_state']}, "
                f"wake words {view['wake_words']}",
                flush=True,
            )
            print(
                f"     🟢 {connection['device_state']}, "
                f"🧩 {', '.join(features) or 'none'}",
                flush=True,
            )
        return 1 if any(
            device.health_snapshot().get("device_state") == "error"
            for device in devices
        ) else 0

    if command == "status":
        key = rest[0] if rest else ""
        device = manager.device(key) if manager is not None else None
        if device is None:
            print("🛰️ No attached device for that name", flush=True)
            return 1
        print(f"🛰️ {device.identity.get('node_name') or device._host}", flush=True)
        _print_health(device.health_snapshot())
        return 0

    if command == "set-led":
        key = rest[0] if rest else ""
        rgb_text = _flag(rest, "--rgb") or ""
        brightness = _flag(rest, "--brightness")
        from .led import parse_hex_rgb

        rgb = parse_hex_rgb(rgb_text) or _ACCENT
        level = float(brightness) if brightness else cfg.led_brightness
        return _await(manager, manager.set_led(key, rgb, level)) if manager else 1

    if command == "announce":
        key = rest[0] if rest else ""
        text = rest[1] if len(rest) > 1 else ""
        return _await(manager, manager.announce(key, text)) if manager else 1

    if command == "play":
        key = rest[0] if rest else ""
        url = rest[1] if len(rest) > 1 else ""
        return _await(manager, manager.play(key, url)) if manager else 1

    if command == "pause":
        key = rest[0] if rest else ""
        return _await(manager, manager.pause_media(key)) if manager else 1

    if command == "resume":
        key = rest[0] if rest else ""
        return _await(manager, manager.resume_media(key)) if manager else 1

    if command == "volume":
        key = rest[0] if rest else ""
        try:
            level = float(rest[1]) if len(rest) > 1 else float(cfg.led_brightness)
        except (TypeError, ValueError):
            print("❓ volume expects a number between 0 and 1", flush=True)
            return 1
        return _await(manager, manager.set_volume(key, level)) if manager else 1

    if command == "mute":
        key = rest[0] if rest else ""
        wanted = str(rest[1] if len(rest) > 1 else "on").strip().lower()
        return (
            _await(manager, manager.set_muted(key, wanted in {"on", "1", "true"}))
            if manager
            else 1
        )

    if command == "stop":
        key = rest[0] if rest else ""
        return _await(manager, manager.stop_media(key)) if manager else 1

    if command == "forget":
        key = rest[0] if rest else ""
        mac = _mac_for(cfg, key)
        if not mac:
            print("🗑️ Nothing stored for that device", flush=True)
            return 0
        pe_config.forget_device(mac)
        print(f"🗑️ Forgot {mac} (metadata and local secret removed)", flush=True)
        print("   ▫️ The device itself keeps its firmware and Noise key", flush=True)
        return 0

    print(f"❓ Unknown voice-pe subcommand: {command}", flush=True)
    return 1


def _pair(cfg: VoicePEConfig) -> int:
    """Discover, provision the Noise key if needed, then store metadata.

    Metadata is stored for every matched node, plaintext ones included, so the
    next start can reconnect from the persisted address instead of scanning
    again; the integration flag is written as well, otherwise a paired unit
    would still leave the manager disabled. ``forget`` reverses both.
    """
    from .provisioning import provision_noise_key

    async def _run() -> int:
        found = await discover(cfg)
        if not found:
            print("🔗 No Voice PE to pair (check power and Wi-Fi)", flush=True)
            return 1
        entry = found[0]
        print(f"🔗 Pairing {entry.get('node_name') or entry.get('host')}", flush=True)
        mac = str(entry.get("mac_address") or "") or str(entry["host"])
        psk = entry.get("noise_psk") or pe_config.get_psk(cfg, entry.get("mac_address", ""))
        stored = False
        for attempt in range(2):
            client = make_client(
                entry["host"],
                int(entry["port"]),
                psk,
                device_name=entry.get("node_name") or None,
            )
            try:
                await client.connect(login=True, log_errors=True)
                info = await client.device_info()
                meta = {
                    "mac_address": str(getattr(info, "mac_address", "") or mac),
                    "node_name": str(getattr(info, "name", "") or entry.get("node_name", "")),
                    "friendly_name": str(getattr(info, "friendly_name", "") or ""),
                    "project_name": str(getattr(info, "project_name", "") or ""),
                    "project_version": str(getattr(info, "project_version", "") or ""),
                    "voice_feature_flags": int(
                        getattr(info, "voice_assistant_feature_flags", 0) or 0
                    ),
                    "addresses": [str(entry["host"])],
                    "port": int(entry["port"]),
                }
                if psk:
                    meta["noise_psk"] = str(psk)
                    print("  ✅ Existing key accepted, no new key generated", flush=True)
                elif getattr(info, "api_encryption_provisionable", False):
                    ok, encoded = await provision_noise_key(client, info)
                    if ok and encoded:
                        meta["noise_psk"] = encoded
                        psk = encoded
                        print("  🔑 Noise PSK installed and stored", flush=True)
                    else:
                        print("  ⚠️  Provisioning window closed, storing plaintext metadata", flush=True)
                else:
                    print("  ✅ Plaintext node (no Noise key on this firmware)", flush=True)
                # Same metadata path as the live device, so both stay in sync.
                stored = bool(pe_config.save_device_metadata(meta["mac_address"], meta))
                break
            except Exception as err:
                print(f"  ⚠️  Attempt {attempt + 1}: {err}", flush=True)
                psk = None
            finally:
                try:
                    await client.disconnect(True)
                except Exception:
                    pass
        if not stored:
            print("  ❌ Nothing stored - the device is not paired", flush=True)
            return 1
        enabled = pe_config.enable_integration(True)
        print(f"  💾 Stored metadata for {mac} (integration {'enabled' if enabled else 'NOT enabled'})", flush=True)
        return 0 if enabled else 1

    return asyncio.run(_run())


def _await(manager, coro) -> int:
    loop = getattr(manager, "_loop", None)
    if loop is None:
        return 1
    try:
        result = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=20.0)
    except Exception as err:
        print(f"⚠️  voice-pe: {err}", flush=True)
        return 1
    if result is False:
        print("⚠️  Device did not acknowledge the command", flush=True)
        return 1
    print("✅ Command delivered", flush=True)
    return 0


def _mac_for(cfg: VoicePEConfig, key: str) -> str:
    needle = str(key or "").strip().lower()
    for mac, meta in cfg.devices.items():
        if not needle:
            return mac
        if needle in {mac.lower(), str(meta.get("node_name", "")).lower()}:
            return mac
        if needle == str(mac).replace(":", "").lower():
            return mac
    return ""
