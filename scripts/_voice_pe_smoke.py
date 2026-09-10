"""Hardware smoke test for the Voice PE integration (stock retail firmware).

Run with the satellite powered and on the same LAN:

    python scripts/_voice_pe_smoke.py [host]

Eight checks, each printed as ``ok``/``FAIL`` with the decoded value. The exit
code is the number of failed checks, so CI can call it directly. No mock, no
Home Assistant server: the script is a plain Native API client on TCP 6053.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis.integrations.voice_pe import config as pe_config  # noqa: E402
from jarvis.integrations.voice_pe.capabilities import build_snapshot  # noqa: E402
from jarvis.integrations.voice_pe.discovery import discover, probe  # noqa: E402
from jarvis.integrations.voice_pe.models import (  # noqa: E402
    ANNOUNCEMENT_TIMEOUT_S,
    make_client,
)
from jarvis.integrations.voice_pe.tts_stream import (  # noqa: E402
    TtsHttpServer,
    lan_ip_for,
    synthesize_pcm,
)

RESULTS: list[tuple[int, str, bool, str]] = []


def _report(index: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((index, name, bool(ok), detail))
    print(f"{'ok  ' if ok else 'FAIL'} {index}. {name}" + (f" - {detail}" if detail else ""))


async def _smoke(host: str | None) -> int:
    from jarvis.config import load_settings

    settings = load_settings()
    values = {name: getattr(settings, name) for name in dir(settings)
              if name.startswith("voice_pe_")}
    if host:
        values["voice_pe_host"] = host
    values["voice_pe_enabled"] = True
    cfg = pe_config.from_settings(SimpleNamespace(**values))
    psk = pe_config.get_psk(cfg, cfg.mac_address or "") or None

    # 1. Discovery: one node in preference order.
    nodes = await discover(cfg) if cfg.discovery_enabled else []
    if not nodes and cfg.host:
        info = await probe(cfg.host, cfg.port, cfg, psk, cfg.device_name)
        if info is not None:
            nodes = [{"host": cfg.host, "port": cfg.port, "node_name": getattr(info, "name", "")}]
    _report(1, "discovery returned a Voice PE node", bool(nodes),
            f"{len(nodes)} node(s): {', '.join(n.get('node_name') or n['host'] for n in nodes)}")
    if not nodes:
        return 1
    entry = nodes[0]
    address = str(entry["host"])
    port = int(entry.get("port") or cfg.port)
    psk = entry.get("noise_psk") or psk

    # 2. Full handshake: Hello + ConnectRequest, then identity.
    client = make_client(address, port, psk, device_name=entry.get("node_name") or None)
    try:
        await client.connect(login=True, log_errors=True)
        info = await client.device_info()
        _report(2, "connect(login=True) + device_info", info is not None,
                f"{getattr(info, 'project_name', '')} v{getattr(info, 'project_version', '')}"
                f" / API {getattr(info, 'esphome_version', '')}")
        if info is None:
            return 1

        # 3. Enumeration: entity list is the base of every key lookup.
        entities, services = await client.list_entities_services()
        _report(3, "list_entities_services is non-empty", bool(entities),
                f"{len(entities)} entities, {len(services)} services")

        # 4. Capability decode from the real flags, not from a version string.
        try:
            caps_model = await client.device_capabilities_compat(info)
        except Exception:
            caps_model = None
        snapshot = build_snapshot(info, list(entities), list(services), caps_model)
        _report(4, "capability snapshot decoded", snapshot.voice_assistant,
                f"flags={snapshot.feature_flags} [{', '.join(snapshot.names()) or 'none'}]"
                f" api_audio_egress={int(snapshot.uses_api_audio)}")

        # 5. One persistent Voice Assistant subscription installs and stays.
        #    The subscription is what arms the message plumbing, so it has to
        #    cover the announcement check below and is released at the end.
        async def _start(*_args):
            return 0

        async def _stop(_abort):
            return None

        async def _audio(_data, _data2=None):
            return None

        states_seen: list = []
        # ``subscribe_states`` has no release handle in 46.x: it dies with the
        # connection, ``subscribe_voice_assistant`` returns one.
        client.subscribe_states(lambda msg: states_seen.append(msg))
        unsub = client.subscribe_voice_assistant(
            handle_start=_start,
            handle_stop=_stop,
            handle_audio=_audio if snapshot.api_audio else None,
        )
        installed = callable(unsub)
        _report(5, "state stream + one voice assistant subscription", installed)

        # 6. TTS WAV over LAN HTTP - the egress of every flag set without SPEAKER.
        server = TtsHttpServer()
        port_http = await server.start()
        key = "smoke"
        server.put(key, synthesize_pcm(None, "") or b"\x01\x02\x03\x04")
        url = f"http://{lan_ip_for(address, port) or address}:{port_http}/{key}"
        body = ""

        def _fetch() -> str:
            from urllib.request import urlopen

            with urlopen(url, timeout=3) as response:  # noqa: S310 - loopback/LAN only
                return response.read().decode("latin-1")

        try:
            body = await asyncio.get_running_loop().run_in_executor(None, _fetch)
        except Exception as err:
            url, body = url, f"error: {err}"
        await server.stop()
        _report(6, "TTS_END WAV is fetchable over LAN HTTP", body.startswith("RIFF"),
                f"{url} ({len(body)} bytes)")

        # 7. Announcement RPC closes with a finished reply (when supported).
        if snapshot.announce:
            try:
                finished = await client.send_voice_assistant_announcement_await_response(
                    url, min(ANNOUNCEMENT_TIMEOUT_S, 10.0),
                    text="smoke", start_conversation=snapshot.start_conversation,
                )
                _report(7, "announcement finished on the device",
                        bool(getattr(finished, "success", False)),
                        f"success={getattr(finished, 'success', False)}")
            except Exception as err:
                _report(7, "announcement finished on the device", False, str(err))
        else:
            _report(7, "announcement finished on the device", True, "no ANNOUNCE flag, skipped")

        # 8. Assistant configuration round-trip inside the same generation.
        try:
            va_config = await client.get_voice_assistant_configuration(5.0)
            _report(8, "voice assistant configuration read", va_config is not None,
                    f"{len(getattr(va_config, 'available_wake_words', []) or [])} wake words,"
                    f" active={list(getattr(va_config, 'active_wake_words', []) or [])}")
        except Exception as err:
            _report(8, "voice assistant configuration read", False, str(err))

        # Release the Voice Assistant subscription of this generation; the state
        # stream ends with the connection itself.
        try:
            if callable(unsub):
                unsub()
        except Exception as err:
            _report(9, "release voice_assistant subscription", False, str(err))
        media_states = [
            getattr(msg, "state", None)
            for msg in states_seen
            if hasattr(msg, "state") and hasattr(msg, "muted")
        ]
        print(
            f"info  {len(states_seen)} entity states seen; media player states: "
            f"{[int(s) for s in media_states if s is not None] or 'none'}"
        )
    finally:
        try:
            await client.disconnect(True)
        except Exception:
            pass

    failed = sum(1 for _i, _n, ok, _d in RESULTS if not ok)
    print(f"-- {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return failed


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else None
    return asyncio.run(_smoke(host))


if __name__ == "__main__":
    raise SystemExit(main())
