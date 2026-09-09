"""Longer Improv serial walk with slower pacing and per-request hex dump."""
import binascii
import time

from serial import Serial

NAMES = {1: "no_network", 2: "awaiting_enable", 3: "network_connecting",
         4: "have_ip", 5: "not_able_to_join", 6: "unable_to_join", 7: "unknown",
         13: "completed"}


def scan(buf):
    out, i = [], 0
    while i < len(buf) - 2:
        length = buf[i]
        if (2 <= length <= 250 and i + length + 1 <= len(buf)
                and buf[i + 1] == 1 and buf[i + 2] in (1, 2, 3)):
            out.append(bytes(buf[i: i + length + 1]))
            i += length + 1
        else:
            i += 1
    return out


def explain(raw):
    kind, payload = raw[2], raw[3:]
    if kind == 1:
        return f"STATE error={payload[0] if payload else 0} " \
               f"state={NAMES.get(payload[1] if len(payload) > 1 else 0, '?')}"
    if kind == 2:
        return f"ERROR {payload[0] if payload else '?'}"
    parts, i = [], 1 if payload[:1] == b"\x01" else 0
    while i < len(payload):
        size = payload[i]
        parts.append(payload[i + 1: i + 1 + size].decode("utf-8", "replace"))
        i += 1 + size
    return "INFO " + " | ".join(parts)


def main() -> int:
    with Serial("COM3", 115200, timeout=0.1) as ser:
        ser.reset_input_buffer()
        for round_no in range(6):
            body = bytes([1, 1])
            ser.write(bytes([len(body)]) + body)
            end = time.time() + 2.0
            buf = bytearray()
            while time.time() < end:
                chunk = ser.read(max(1, ser.in_waiting))
                if chunk:
                    buf.extend(chunk)
                else:
                    time.sleep(0.01)
            raw = bytes(buf)
            packets = scan(raw)
            print(f"round {round_no}: bytes={len(raw)} packets={len(packets)}", flush=True)
            for packet in packets:
                print("   ", explain(packet), flush=True)
            if not packets:
                print("    raw:", binascii.hexlify(raw[:48]).decode(), flush=True)
            time.sleep(1.0)
    return 0


raise SystemExit(main())
