from __future__ import annotations
import socket
import ipaddress
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import threading
from ..debug import debug_log

try:
    import geoip2.database
    import geoip2.errors
    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False

try:
    import miniupnpc
    MINIUPNPC_AVAILABLE = True
except ImportError:
    MINIUPNPC_AVAILABLE = False

# Session flag to show location warning only once per session
_location_warning_shown = False

# Simple in-memory caches (module scoped)
# Cache for location lookups keyed by final resolved IP -> location_info dict
_location_cache: Dict[str, Dict[str, Any]] = {}

# Cache for CGNAT OpenDNS public IP resolution attempts keyed by original CGNAT IP.
# Value is tuple: (timestamp, resolved_public_ip or None). We avoid re-querying OpenDNS
# more than once per hour for the same CGNAT IP. This respects user privacy by
# minimizing external DNS queries.
_cgnat_resolution_cache: Dict[str, tuple[datetime, Optional[str]]] = {}

# TTL for CGNAT OpenDNS resolution attempts
_CGNAT_RESOLUTION_TTL = timedelta(hours=1)

# Disk cache paths (share directory with geoip DB for locality)
def _cache_base_dir() -> Path:
    return Path.home() / ".local" / "share" / "jarvis"

_LOCATION_CACHE_FILE = _cache_base_dir() / "location_cache.json"
_CGNAT_CACHE_FILE = _cache_base_dir() / "cgnat_cache.json"

_cache_lock = threading.RLock()

def _load_disk_caches() -> None:
    """Load caches from disk into memory (best-effort)."""
    try:
        base = _cache_base_dir()
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    now = datetime.utcnow()
    # Location cache
    try:
        if _LOCATION_CACHE_FILE.exists():
            with _LOCATION_CACHE_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # Expect mapping ip -> {data: {...}, ts: iso}
            for ip, payload in raw.items():
                data = payload.get("data") if isinstance(payload, dict) else None
                ts_str = payload.get("ts") if isinstance(payload, dict) else None
                if not isinstance(data, dict) or not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue
                # TTL for location cache can vary; default 60 minutes (aligned with config default). We'll store ttl minutes in payload optionally.
                ttl_minutes = payload.get("ttl") if isinstance(payload, dict) else None
                try:
                    ttl_minutes = int(ttl_minutes)
                except Exception:
                    ttl_minutes = 60
                if now - ts < timedelta(minutes=ttl_minutes):
                    _location_cache[ip] = data
    except Exception:
        pass
    # CGNAT resolution cache
    try:
        if _CGNAT_CACHE_FILE.exists():
            with _CGNAT_CACHE_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            for cgnat_ip, payload in raw.items():
                if not isinstance(payload, dict):
                    continue
                ts_str = payload.get("ts")
                resolved = payload.get("resolved")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue
                if now - ts < _CGNAT_RESOLUTION_TTL:
                    _cgnat_resolution_cache[cgnat_ip] = (ts, resolved)
    except Exception:
        pass

def _persist_disk_caches(location_cache_minutes: int = 60) -> None:
    """Persist in-memory caches to disk (best-effort)."""
    with _cache_lock:
        try:
            base = _cache_base_dir()
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        # Location cache serialization
        try:
            loc_out = {}
            now = datetime.utcnow().isoformat()
            for ip, data in _location_cache.items():
                loc_out[ip] = {"data": data, "ts": now, "ttl": int(location_cache_minutes)}
            with _LOCATION_CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(loc_out, f)
        except Exception:
            pass
        # CGNAT cache
        try:
            cgnat_out = {}
            for ip, (ts, resolved) in _cgnat_resolution_cache.items():
                cgnat_out[ip] = {"ts": ts.isoformat(), "resolved": resolved}
            with _CGNAT_CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(cgnat_out, f)
        except Exception:
            pass

# Load caches on module import
_load_disk_caches()


def _get_local_network_ip() -> Optional[str]:
    """
    Get the local network IP address without making external calls.
    This respects privacy by not contacting third-party services.

    Note: This returns the local IP, not the public IP, so geolocation
    will only work if the user manually configures their public IP.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connect to a non-routable address to determine local IP
            # This doesn't actually send any data
            s.connect(("10.254.254.254", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            s.close()

    except Exception:
        pass

    return None


def _get_external_ip_via_upnp() -> Optional[str]:
    """
    Get external IP address by querying the local router via UPnP.
    This is privacy-friendly as it only communicates with your local router.

    Returns:
        External IP address if successful, None otherwise.
    """
    if not MINIUPNPC_AVAILABLE:
        return None

    try:
        upnp = miniupnpc.UPnP()
        upnp.discoverdelay = 200  # milliseconds

        # Discover UPnP devices
        device_count = upnp.discover()
        if device_count == 0:
            return None

        # Select the Internet Gateway Device
        upnp.selectigd()

        # Get the external IP address
        external_ip = upnp.externalipaddress()

        # Validate the IP address and ensure it's not private
        if external_ip and not _is_private_ip(external_ip) and "." in external_ip:
            return external_ip

    except Exception:
        # UPnP might not be supported or enabled
        pass

    return None


def _is_private_ip(ip: str) -> bool:
    """Check if an IP address is private/local or special-use (non-geolocatable)."""
    if not ip:
        return True
    try:
        addr = ipaddress.ip_address(ip)
        # RFC 6598 shared address space (CGNAT) will be treated separately; don't mark as private here
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return True
        # 0.0.0.0 and unspecified
        if addr.is_unspecified:
            return True
    except ValueError:
        return True
    return False


def _is_cgnat_ip(ip: str) -> bool:
    """Return True if IP is in carrier-grade NAT (100.64.0.0/10)."""
    try:
        addr = ipaddress.ip_address(ip)
        return addr in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


def _get_external_ip_via_socket() -> Optional[str]:
    """
    Get external IP address by creating a socket connection to determine
    which local interface would be used for external communication.

    This method creates a connection to a well-known external server (Google DNS)
    to determine the local IP address used for external communication.
    No data is actually sent.

    Returns:
        IP address used for external communication, None if failed.
    """
    try:
        # Try multiple well-known servers to increase reliability
        servers = [
            ("8.8.8.8", 80),      # Google DNS
            ("1.1.1.1", 80),      # Cloudflare DNS
            ("208.67.222.222", 80), # OpenDNS
        ]

        for server_ip, port in servers:
            try:
                # Create a UDP socket (no actual data transmission)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Connect to determine which local interface would be used
                    s.connect((server_ip, port))
                    detected_ip = s.getsockname()[0]
                    s.close()

                    # Return the first non-private IP we find
                    if detected_ip and not _is_private_ip(detected_ip):
                        return detected_ip

                except Exception:
                    s.close()

            except Exception:
                continue

    except Exception:
        pass

    return None


def _get_external_ip_automatically() -> Optional[str]:
    """
    Attempt to automatically determine the external IP address using
    privacy-friendly methods in order of preference:

    1. UPnP (query local router) - most privacy-friendly
    2. Socket connection (determine external interface) - minimal external contact

    Returns:
        External IP address if successful, None otherwise.
    """
    # Try UPnP first (most privacy-friendly)
    ip = _get_external_ip_via_upnp()
    if ip:
        return ip

    # Fallback to socket method
    ip = _get_external_ip_via_socket()
    if ip:
        return ip

    return None


def _get_database_path() -> Path:
    """Get the path where the GeoLite2 database should be stored."""
    base_dir = Path.home() / ".local" / "share" / "jarvis" / "geoip"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "GeoLite2-City.mmdb"


def _print_location_setup_instructions(db_path: Path) -> None:
    """Print user-friendly location setup instructions with proper formatting."""
    global _location_warning_shown

    # Only show warning once per session
    if _location_warning_shown:
        return

    _location_warning_shown = True

    print("  📍 Location features are not available")
    print()
    print("     To enable location-based features:")
    print("     1. 🌐 Register for a free MaxMind account:")
    print("        https://www.maxmind.com/en/geolite2/signup")
    print()
    print("     2. 📥 Download the GeoLite2 City database (MMDB format)")
    print()
    print("     3. 📂 Save the database file as:")
    print(f"        {db_path}")


def _download_geolite2_database() -> bool:
    """
    Download the GeoLite2 City database from MaxMind.
    Note: This requires registration for a license key since 2019.
    For now, we'll provide instructions for manual download.
    """
    try:
        db_path = _get_database_path()

        # Check if database already exists and is recent (less than 30 days old)
        if db_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(db_path.stat().st_mtime)).days
            if age_days < 30:
                debug_log("GeoLite2 database found and up to date", "location")
                return True

        debug_log(f"GeoLite2 database not found or outdated at: {db_path}", "location")
        _print_location_setup_instructions(db_path)

        return False

    except Exception as e:
        debug_log(f"Error checking database: {e}", "location")
        return False


def _resolve_public_ip_via_opendns(timeout: float = 1.5) -> Optional[str]:
    """Resolve true public IP via a single DNS query to OpenDNS (myip.opendns.com)."""
    try:
        resolver_ip = ("208.67.222.222", 53)
        import random
        tid = random.randint(0, 0xFFFF)
        header = tid.to_bytes(2, 'big') + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
        labels = b"".join(len(part).to_bytes(1, 'big') + part.encode('ascii') for part in "myip.opendns.com".split('.')) + b"\x00"
        qtype_qclass = b"\x00\x01\x00\x01"
        packet = header + labels + qtype_qclass
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        try:
            s.sendto(packet, resolver_ip)
            data, _ = s.recvfrom(512)
        finally:
            s.close()
        if len(data) < 12 or data[0:2] != tid.to_bytes(2, 'big'):
            return None
        question_len = len(labels) + 4
        answer_start = 12 + question_len
        if len(data) < answer_start + 12:
            return None
        rdlength = int.from_bytes(data[answer_start + 10:answer_start + 12], 'big') if len(data) >= answer_start + 12 else 0
        rdata_start = answer_start + 12
        rdata_end = rdata_start + rdlength
        if rdlength == 4 and len(data) >= rdata_end:
            ip_bytes = data[rdata_start:rdata_end]
            return '.'.join(str(b) for b in ip_bytes)
    except Exception:
        return None
    return None


def get_location_info(
    ip_address: Optional[str] = None,
    *,
    config_ip: Optional[str] = None,
    auto_detect: bool = True,
    resolve_cgnat_public_ip: bool = True,
) -> Dict[str, Any]:
    """Get location information for an IP address.

    Args:
        ip_address: Direct IP address to look up. If provided it is used as-is.
        config_ip: Manually configured public IP (takes precedence over auto-detect when ip_address is None).
        auto_detect: Attempt to discover an external IP via UPnP / socket heuristics if neither ip_address nor config_ip given.
        resolve_cgnat_public_ip: If True and a CGNAT (100.64.0.0/10) address is detected, attempt a single DNS query via OpenDNS to discover the true public IP (privacy-light).
    """
    if not GEOIP2_AVAILABLE:
        return {"error": "geoip2 library not available"}

    # Get IP address to lookup (prioritize parameter, then config, then auto-detect)
    if ip_address is None:
        if config_ip:
            ip_address = config_ip
        elif auto_detect:
            # Try automatic detection using privacy-friendly methods
            ip_address = _get_external_ip_automatically()
            if not ip_address:
                # Final fallback to local IP (won't work for geolocation)
                ip_address = _get_local_network_ip()
        else:
            # Fall back to local IP without auto-detection
            ip_address = _get_local_network_ip()

    if not ip_address:
        return {"error": "No IP address available. Try enabling auto-detection or configure 'location_ip_address' in your config."}

    # Mark CGNAT and optionally resolve public IP via OpenDNS if enabled in settings
    cgnat_flag = _is_cgnat_ip(ip_address)
    if cgnat_flag and resolve_cgnat_public_ip:
        # Check CGNAT resolution cache first
        cache_entry = _cgnat_resolution_cache.get(ip_address)
        now = datetime.utcnow()
        if cache_entry:
            ts, cached_public = cache_entry
            if now - ts < _CGNAT_RESOLUTION_TTL:
                # Use cached result (even if None to avoid extra queries)
                if cached_public and not _is_cgnat_ip(cached_public) and not _is_private_ip(cached_public):
                    debug_log(f"CGNAT IP {ip_address} used cached public {cached_public}", "location")
                    ip_address = cached_public
            else:
                # Expired entry
                _cgnat_resolution_cache.pop(ip_address, None)
                cache_entry = None
        if not cache_entry:
            resolved = _resolve_public_ip_via_opendns()
            _cgnat_resolution_cache[ip_address] = (now, resolved)
            # Persist CGNAT cache change
            _persist_disk_caches()
            if resolved and not _is_cgnat_ip(resolved) and not _is_private_ip(resolved):
                debug_log(f"CGNAT IP {ip_address} resolved to public {resolved} via OpenDNS", "location")
                ip_address = resolved

    # Return cached location result if we already computed for this final ip_address
    if ip_address in _location_cache:
        cached = _location_cache[ip_address]
        # Ensure we always include ip key even if older cache missing it
        if 'ip' not in cached:
            cached['ip'] = ip_address
        return cached.copy()

    # Check if database is available
    db_path = _get_database_path()
    if not db_path.exists():
        if not _download_geolite2_database():
            return {"error": "GeoLite2 database not available"}

    try:
        with geoip2.database.Reader(str(db_path)) as reader:
            response = reader.city(ip_address)

            location_info = {
                "ip": ip_address,
                "country": response.country.name,
                "country_code": response.country.iso_code,
                "region": response.subdivisions.most_specific.name,
                "region_code": response.subdivisions.most_specific.iso_code,
                "city": response.city.name,
                "latitude": float(response.location.latitude) if response.location.latitude else None,
                "longitude": float(response.location.longitude) if response.location.longitude else None,
                "timezone": response.location.time_zone,
                "accuracy_radius": response.location.accuracy_radius,
            }

            # Clean up None values and empty strings
            cleaned_info = {k: v for k, v in location_info.items() if v is not None and v != ""}
            debug_log(f"Location detected: {cleaned_info.get('city', 'Unknown city')}, {cleaned_info.get('country', 'Unknown country')}", "location")
            # Cache successful lookup
            _location_cache[ip_address] = cleaned_info.copy()
            _persist_disk_caches()
            return cleaned_info

    except geoip2.errors.AddressNotFoundError:
        debug_log(f"IP address {ip_address} not found in database", "location")
        reason = "cgnat_not_found" if cgnat_flag else "not_found"
        advice = (
            "Detected CGNAT (100.64.0.0/10). Configure 'location_ip_address' with a real public IP or disable auto-detect."
            if cgnat_flag else
            "If this is CGNAT or a very new allocation, configure 'location_ip_address' manually."
        )
        result = {
            "error": f"IP address {ip_address} not found in database",
            "ip": ip_address,
            "reason": reason,
            "advice": advice,
        }
        # Cache negative result to avoid repeated DB lookups for same IP (could revisit TTL policy later)
        _location_cache[ip_address] = result.copy()
        _persist_disk_caches()
        return result
    except Exception as e:
        debug_log(f"Error looking up location: {e}", "location")
        result = {"error": f"Error looking up location: {e}", "ip": ip_address}
        _location_cache[ip_address] = result.copy()
        _persist_disk_caches()
        return result


def get_location_context(
    *,
    config_ip: Optional[str] = None,
    auto_detect: bool = True,
    resolve_cgnat_public_ip: bool = True,
) -> str:
    """Generate a concise location context string using explicit parameters."""
    location_info = get_location_info(
        config_ip=config_ip,
        auto_detect=auto_detect,
        resolve_cgnat_public_ip=resolve_cgnat_public_ip,
    )

    if "error" in location_info:
        return "Location: Unknown"

    parts = []

    # Add city and region if available
    if location_info.get("city"):
        if location_info.get("region"):
            parts.append(f"{location_info['city']}, {location_info['region']}")
        else:
            parts.append(location_info["city"])
    elif location_info.get("region"):
        parts.append(location_info["region"])

    # Add country
    if location_info.get("country"):
        parts.append(location_info["country"])

    # Add timezone if different from what we might expect
    if location_info.get("timezone"):
        parts.append(f"({location_info['timezone']})")

    if parts:
        return f"Location: {', '.join(parts)}"
    else:
        return "Location: Unknown"


def is_location_available() -> bool:
    """Check if location detection is available and working."""
    if not GEOIP2_AVAILABLE:
        return False

    db_path = _get_database_path()
    return db_path.exists()


def setup_location_database() -> bool:
    """
    Setup the location database. This will check for the database
    and provide instructions if it's not available.

    Returns:
        True if database is available and ready, False otherwise.
    """
    if not GEOIP2_AVAILABLE:
        print("📦 Location library not installed")
        print()
        print("   To install the required geoip2 library:")
        print("   pip install geoip2")
        print()
        debug_log("geoip2 library not available", "location")
        return False

    return _download_geolite2_database()


def get_detailed_location_info(
    ip_address: Optional[str] = None,
    *,
    config_ip: Optional[str] = None,
    auto_detect: bool = True,
    resolve_cgnat_public_ip: bool = True,
) -> Dict[str, Any]:
    """Get detailed location information including coordinates and formatted address."""
    location_info = get_location_info(
        ip_address,
        config_ip=config_ip,
        auto_detect=auto_detect,
        resolve_cgnat_public_ip=resolve_cgnat_public_ip,
    )

    if "error" in location_info:
        return location_info

    # Add computed fields
    if location_info.get("latitude") and location_info.get("longitude"):
        location_info["coordinates"] = f"{location_info['latitude']}, {location_info['longitude']}"

    # Add formatted address
    address_parts = []
    for field in ["city", "region", "country"]:
        if location_info.get(field):
            address_parts.append(location_info[field])

    if address_parts:
        location_info["formatted_address"] = ", ".join(address_parts)

    return location_info


# For testing and debugging
if __name__ == "__main__":
    print("Testing location detection...")

    # Test local IP detection (privacy-focused)
    ip = _get_local_network_ip()
    print(f"Local IP: {ip}")

    # Test location lookup (will likely fail without public IP)
    location = get_location_info()
    print(f"Location info: {location}")

    # Test context generation
    context = get_location_context()
    print(f"Location context: {context}")

    # Test detailed info
    detailed = get_detailed_location_info()
    print(f"Detailed location: {detailed}")

    print("\nNote: For accurate location detection, configure 'location_ip_address' in your config")
    print("or provide an IP address explicitly to respect privacy.")
