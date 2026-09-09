"""Reset with the documented 1+0 sequence, then Improv-BLE within the window."""
import asyncio
import time

import serial
from bleak import BleakClient, BleakScanner

NAMES = {1: "no_network", 2: "awaiting_enable", 3: "network_connecting", 4: "have_ip",
         5: "not_able_to_join", 6: "unable_to_join", 13: "completed"}


def explain(raw: bytes) -> str:
    if len(raw) < 2:
        return f"raw={raw!r}"
    kind, payload = raw[1], raw[2:]
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


def read_uart(ser, seconds: float) -> str:
    end = time.time() + seconds
    buf = bytearray()
    while time.time() < end:
        chunk = ser.read(max(1, ser.in_waiting))
        if chunk:
            buf.extend(chunk)
    return bytes(buf).decode("utf-8", "replace")


def show(text: str) -> None:
    for line in text.splitlines():
        if line.strip():
            print("uart:", line.strip(), flush=True)


async def main() -> int:
    ser = serial.Serial("COM3", 115200, timeout=0.1)
    try:
        # 1+0: DTR=1 RTS=0, pulse RTS high, then DTR=0 RTS=1
        ser.dtr = True
        ser.rts = False
        time.sleep(0.05)
        ser.rts = True
        time.sleep(0.05)
        ser.dtr = False
        ser.rts = True
        show(read_uart(ser, 2.5))

        devices = await BleakScanner.discover(timeout=5.0)
        target = next((d for d in devices
                       if (d.name or "").lower().startswith(("ha-voice-pe",
                                                            "home-assistant-voice"))),
                      None)
        if target is None:
            show(read_uart(ser, 1.5))
            print("no BLE device")
            return 1
        print(f"ble: {target.address} {target.name}", flush=True)

        async with BleakClient(target.address, timeout=12.0) as client:
            chars = {c.uuid.lower()[-4:]: c
                     for s in client.services for c in s.characteristics
                     if c.uuid.lower().startswith("00467768")}
            print("chars:", sorted(chars), flush=True)
            if "8003" not in chars:
                show(read_uart(ser, 1.5))
                return 1

            async def cmd(number: int, payload: bytes = b"", label: str = "") -> None:
                await client.write_gatt_char(chars["8003"], bytes([number]) + payload,
                                             response=False)
                for _ in range(4):
                    await asyncio.sleep(0.2)
                    out = bytes(await client.read_gatt_char(chars.get("8004")))
                    if out:
                        print(f"  {label}: {explain(out)}", flush=True)
                        return
                state = (bytes(await client.read_gatt_char(chars["8002"]))
                         if "8002" in chars else b"")
                print(f"  {label}: state={state!r}", flush=True)

            await cmd(1, b"", "state")
            await cmd(3, b"DILNA21", "ssid")
            await cmd(4, b"nanotriko", "credential")
            await cmd(5, b"", "apply")
            for _ in range(8):
                await cmd(1, b"", "poll")
                await asyncio.sleep(1.0)
        show(read_uart(ser, 3.0))
        return 0
    finally:
        ser.close()


raise SystemExit(asyncio.run(main()))
