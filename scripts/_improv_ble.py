"""Improv BLE join for the Voice PE: plain reset (1+0) then RPC over BLE."""
import asyncio
import time

import serial
from bleak import BleakClient, BleakScanner

NAMES = {0: "stop", 1: "no_network", 2: "awaiting_enable", 3: "network_connecting",
         4: "have_ip", 5: "not_able_to_join", 6: "unable_to_join", 7: "unknown",
         13: "completed"}


def frame(command: int, payload: bytes = b"") -> bytes:
    body = bytes([1, command]) + payload
    return bytes([len(body)]) + body


def explain(raw: bytes) -> str:
    if len(raw) < 3:
        return f"raw={raw!r}"
    kind, payload = raw[2], raw[3:]
    if kind == 1:
        return (f"STATE error={payload[0] if payload else 0} "
                f"state={NAMES.get(payload[1] if len(payload) > 1 else 0, '?')}")
    if kind == 2:
        return f"ERROR {payload[0] if payload else '?'}"
    parts, i = [], 1 if payload[:1] == b"\x01" else 0
    while i < len(payload):
        size = payload[i]
        parts.append(payload[i + 1: i + 1 + size].decode("utf-8", "replace"))
        i += 1 + size
    return f"INFO {' | '.join(parts)}"


def reset_and_log(seconds: float = 1.6) -> str:
    """Classic ESP32 1+0 reset (DTR=1, negative RTS pulse) then read the log."""
    with serial.Serial("COM3", 115200, timeout=0.1) as ser:
        ser.dtr = True
        ser.rts = False
        ser.rts = True
        time.sleep(0.05)
        ser.rts = False
        end = time.time() + seconds
        buf = bytearray()
        while time.time() < end:
            chunk = ser.read(max(1, ser.in_waiting))
            if chunk:
                buf.extend(chunk)
    return bytes(buf).decode("utf-8", "replace")


async def main() -> int:
    for line in reset_and_log().splitlines():
        if line.strip():
            print("uart:", line.strip(), flush=True)

    devices = await BleakScanner.discover(timeout=6.0)
    target = next((d for d in devices
                   if (d.name or "").lower().startswith(("ha-voice-pe",
                                                        "home-assistant-voice"))),
                  None)
    if target is None:
        print("no Improv BLE device")
        return 1
    print(f"target {target.address} {target.name}", flush=True)

    async with BleakClient(target.address, timeout=15.0) as client:
        chars = {c.uuid.lower()[-4:]: c
                 for s in client.services for c in s.characteristics
                 if c.uuid.lower().startswith("00467768")}
        rpc_in, rpc_out, state_c = chars.get("8003"), chars.get("8004"), chars.get("8002")

        async def command(payload: bytes, label: str) -> int:
            await client.write_gatt_char(rpc_in, payload, response="write" in rpc_in.properties)
            got = 0
            for attempt in range(4):
                await asyncio.sleep(0.2)
                out = bytes(await client.read_gatt_char(rpc_out))
                state = bytes(await client.read_gatt_char(state_c)) if state_c else b""
                print(f"  {label} t{attempt}: rpc={explain(out)} state={state!r}", flush=True)
                if len(out) > 2:
                    got = out[4] if len(out) > 4 and out[2] == 1 else got
                    break
                if state and state[0] in (4, 5, 6):
                    got = state[0]
                    break
            return got

        await command(frame(1), "get_state")
        await command(frame(2), "get_info")
        await command(frame(3, b"DILNA21"), "ssid")
        await command(frame(4, b"nanotriko"), "credential")
        await command(frame(5), "apply")

        final = 0
        for _ in range(12):
            final = await command(frame(1), "poll")
            if final in (4, 5, 6):
                break
            await asyncio.sleep(1.0)
        print(f"final={NAMES.get(final, 'unknown')}", flush=True)
        return 0 if final == 4 else 1


raise SystemExit(asyncio.run(main()))
