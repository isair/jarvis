"""Hex-level Improv serial probe on COM3."""
import binascii
import time

from serial import Serial

with Serial("COM3", 115200, timeout=0.1, dsrdtr=True) as ser:
    ser.reset_input_buffer()
    # drain the boot log first
    time.sleep(0.2)
    ser.reset_input_buffer()
    for cmd in (6, 1, 2):
        body = bytes([1, cmd])
        ser.write(bytes([len(body)]) + body)
        end = time.time() + 1.2
        buf = bytearray()
        while time.time() < end:
            chunk = ser.read(max(1, ser.in_waiting))
            if chunk:
                buf.extend(chunk)
        raw = bytes(buf)
        print(f"cmd={cmd} n={len(raw)} hex={binascii.hexlify(raw[:80]).decode()}")
        print(f"   txt={raw[:120].decode('utf-8', 'replace')!r}")
