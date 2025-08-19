from __future__ import annotations
import sys
import socket
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

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
    """Check if an IP address is private/local."""
    if not ip:
        return True
    
    # Common private IP ranges
    private_ranges = [
        "127.",      # localhost
        "10.",       # Class A private
        "172.16.",   # Class B private (172.16.0.0 to 172.31.255.255)
        "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
        "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
        "192.168.",  # Class C private
        "169.254.",  # Link-local
        "0.0.0.0",   # Invalid
    ]
    
    return any(ip.startswith(prefix) for prefix in private_ranges)


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
                return True
        
        print(f"[location] GeoLite2 database not found or outdated at: {db_path}", file=sys.stderr)
        print("[location] To enable location features:", file=sys.stderr)
        print("[location] 1. Register for a free MaxMind account at: https://www.maxmind.com/en/geolite2/signup", file=sys.stderr)
        print("[location] 2. Download GeoLite2 City database (MMDB format)", file=sys.stderr)
        print(f"[location] 3. Save it as: {db_path}", file=sys.stderr)
        
        return False
        
    except Exception as e:
        print(f"[location] Error checking database: {e}", file=sys.stderr)
        return False


def get_location_info(ip_address: Optional[str] = None, config_ip: Optional[str] = None, auto_detect: bool = True) -> Dict[str, Any]:
    """
    Get location information for the given IP address.
    
    Args:
        ip_address: IP address to lookup. If None, tries other sources.
        config_ip: IP address from configuration. Used when ip_address is None.
        auto_detect: Whether to attempt automatic IP detection via UPnP/socket.
    
    Returns:
        Dictionary with location information or empty dict if unavailable.
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
            return {k: v for k, v in location_info.items() if v is not None and v != ""}
            
    except geoip2.errors.AddressNotFoundError:
        return {"error": f"IP address {ip_address} not found in database", "ip": ip_address}
    except Exception as e:
        return {"error": f"Error looking up location: {e}", "ip": ip_address}


def get_location_context(config_ip: Optional[str] = None, auto_detect: bool = True) -> str:
    """Generate location context string for the agent."""
    location_info = get_location_info(config_ip=config_ip, auto_detect=auto_detect)
    
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
        print("[location] geoip2 library not installed. Install with: pip install geoip2", file=sys.stderr)
        return False
    
    return _download_geolite2_database()


def get_detailed_location_info(ip_address: Optional[str] = None, config_ip: Optional[str] = None, auto_detect: bool = True) -> Dict[str, Any]:
    """
    Get detailed location information including coordinates.
    This is useful for more advanced location-based features.
    """
    location_info = get_location_info(ip_address, config_ip=config_ip, auto_detect=auto_detect)
    
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
