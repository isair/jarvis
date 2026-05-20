#!/usr/bin/env python3
"""Download GeoLite2-City.mmdb into ~/.local/share/jarvis/geoip/

Official (preferred): set MAXMIND_LICENSE_KEY from https://www.maxmind.com/en/accounts/current/license-key

Optional mirror (no MaxMind account): set JARVIS_GEOLITE2_USE_MIRROR=1
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from jarvis.utils.location import _get_database_path  # noqa: E402

_MIRROR_URL = (
    "https://github.com/P3TERX/GeoLite.mmdb/raw/download/GeoLite2-City.mmdb"
)
_OFFICIAL_URL = (
    "https://download.maxmind.com/geoip/databases/GeoLite2-City/download"
    "?suffix=tar.gz&license_key={license_key}"
)


def _download_official(license_key: str, dest: Path) -> bool:
    url = _OFFICIAL_URL.format(license_key=license_key.strip())
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Jarvis-GeoLite2/1.0"})
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("GeoLite2-City.mmdb")),
            None,
        )
        if member is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        extracted = tar.extractfile(member)
        if extracted is None:
            return False
        dest.write_bytes(extracted.read())
    return dest.is_file() and dest.stat().st_size > 1_000_000


def _download_mirror(dest: Path) -> bool:
    resp = requests.get(
        _MIRROR_URL,
        timeout=180,
        headers={"User-Agent": "Jarvis-GeoLite2/1.0"},
        stream=True,
    )
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    return dest.is_file() and dest.stat().st_size > 1_000_000


def main() -> int:
    dest = _get_database_path()
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"✅ Already present: {dest} ({dest.stat().st_size // (1024 * 1024)} MB)")
        return 0

    license_key = os.environ.get("MAXMIND_LICENSE_KEY", "").strip()
    if license_key:
        print("📥 Downloading from MaxMind (official)...")
        try:
            if _download_official(license_key, dest):
                print(f"✅ Saved {dest}")
                return 0
        except Exception as exc:
            print(f"⚠️  Official download failed: {exc}")

    if os.environ.get("JARVIS_GEOLITE2_USE_MIRROR", "").strip() in ("1", "true", "yes"):
        print("📥 Downloading GeoLite2-City mirror...")
        try:
            if _download_mirror(dest):
                print(f"✅ Saved {dest}")
                print("💡 Prefer an official MaxMind download when you have a license key.")
                return 0
        except Exception as exc:
            print(f"❌ Mirror download failed: {exc}")
            return 1

    print("❌ No database downloaded.")
    print("   Set MAXMIND_LICENSE_KEY for official download, or")
    print("   JARVIS_GEOLITE2_USE_MIRROR=1 for the community mirror.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
