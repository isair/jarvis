"""Collect the Voice PE console log for a few seconds."""
import time

from serial import Serial

with Serial("COM3", 115200, timeout=0.2, dsrdtr=True) as ser:
    ser.reset_input_buffer()
    end = time.time() + 12.0
    buf = bytearray()
    while time.time() < end:
        chunk = ser.read(max(1, ser.in_waiting))
        if chunk:
            buf.extend(chunk)
print(bytes(buf).decode("utf-8", "replace"))
