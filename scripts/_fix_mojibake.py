"""Repair double-encoded (mojibake) literals in the source tree.

Per-character encoding: the double-encoded runs mix code points that only
cp1252 maps (U+0161 -> 0x9A) with ones only latin-1 maps (U+008F -> 0x8F), so
each character is encoded with cp1252 first and latin-1 as the fallback, then
the byte string is decoded as UTF-8. Correctly stored emoji (code points above
U+00FF that cp1252 cannot map) fail and stay untouched.

    python scripts/_fix_mojibake.py
"""

from __future__ import annotations

from pathlib import Path


def _encode_run(run):
    out = bytearray()
    for char in run:
        try:
            out += char.encode("cp1252")
        except Exception:
            try:
                out += char.encode("latin-1")
            except Exception:
                return None
    return bytes(out)


def _decode_run(run):
    raw = _encode_run(run)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except Exception:
        return None


def _fix_line(line):
    out = []
    fixed = 0
    index = 0
    size = len(line)
    while index < size:
        if ord(line[index]) > 127:
            end = index
            while end < size and ord(line[end]) > 127:
                end += 1
            run = line[index:end]
            decoded = _decode_run(run)
            if decoded is not None and decoded != run:
                out.append(decoded)
                fixed += 1
            else:
                out.append(run)
            index = end
        else:
            out.append(line[index])
            index += 1
    return "".join(out), fixed


def main():
    root = Path(__file__).resolve().parents[1]
    total = 0
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rebuilt = []
        changed = 0
        for line in text.splitlines(keepends=True):
            if line.endswith("\n"):
                body, eol = line[:-1], "\n"
            else:
                body, eol = line, ""
            new, count = _fix_line(body)
            rebuilt.append(new + eol)
            changed += count
        if changed:
            path.write_text("".join(rebuilt), encoding="utf-8")
            total += changed
            print("fixed " + str(path.relative_to(root)) + " (" + str(changed) + " runs)", flush=True)
    print("MOJIBAKE_REPAIRED " + str(total) + " runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
