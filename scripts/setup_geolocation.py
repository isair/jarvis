#!/usr/bin/env python3
"""
Setup script for GeoLite2 geolocation database.

This script helps users set up the MaxMind GeoLite2 database required for
location-based features in Jarvis.

Since MaxMind requires registration for free access to GeoLite2 data (as of 2019),
this script provides instructions and utilities to help with the setup process.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add the src directory to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    # Location utilities live under utils.location after refactor.
    from jarvis.utils.location import (
        _get_database_path,
        is_location_available,
        get_location_info,
        setup_location_database,
        _get_local_network_ip,
        _get_external_ip_automatically,
    )
    from jarvis.config import load_settings
    SETTINGS = load_settings()
    JARVIS_AVAILABLE = True
except ImportError as e:
    print(
        "Warning: Could not import Jarvis location utilities from 'jarvis.utils.location'.\n"
        f"  Import error: {e}\n"
        "  Make sure you're running from the repository root and that 'src' is on PYTHONPATH.\n"
        "  Example (zsh/bash): export PYTHONPATH=\"$(pwd)/src:$PYTHONPATH\"\n"
        "  Or install the project in editable mode once packaging is set up (pip install -e .)."
    )
    JARVIS_AVAILABLE = False


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import geoip2
        return True
    except ImportError:
        return False


def install_dependencies() -> bool:
    """Install required dependencies."""
    print("Installing geoip2 dependency...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geoip2==4.8.0"])
        return True
    except subprocess.CalledProcessError:
        return False


def get_database_info() -> dict:
    """Get information about the database location and status."""
    if not JARVIS_AVAILABLE:
        base_dir = Path.home() / ".local" / "share" / "jarvis" / "geoip"
        db_path = base_dir / "GeoLite2-City.mmdb"
    else:
        db_path = _get_database_path()

    return {
        "path": db_path,
        "directory": db_path.parent,
        "exists": db_path.exists(),
        "size": db_path.stat().st_size if db_path.exists() else 0,
    }


def print_setup_instructions():
    """Print instructions for setting up the GeoLite2 database."""
    db_info = get_database_info()

    print("\n" + "="*60)
    print("📍 JARVIS GEOLOCATION SETUP")
    print("="*60)

    print(f"Database location: {db_info['path']}")
    print(f"Database exists: {'✅ Yes' if db_info['exists'] else '❌ No'}")

    if db_info['exists']:
        size_mb = db_info['size'] / (1024 * 1024)
        print(f"Database size: {size_mb:.1f} MB")

        if JARVIS_AVAILABLE:
            print("\n🧪 Testing location detection...")
            try:
                location = get_location_info(settings=SETTINGS)
                if "error" in location:
                    print(f"❌ Location test failed: {location['error']}")
                else:
                    print("✅ Location detection working!")
                    print(f"   Detected: {location.get('city', 'Unknown')}, {location.get('country', 'Unknown')}")
            except Exception as e:
                print(f"❌ Location test error: {e}")
    else:
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Register for a free MaxMind account:")
        print("   https://www.maxmind.com/en/geolite2/signup")
        print()
        print("2. Generate a license key in your account dashboard")
        print()
        print("3. Download GeoLite2 City database:")
        print("   - Go to: https://www.maxmind.com/en/accounts/current/geoip/downloads")
        print("   - Download: GeoLite2 City (MMDB format)")
        print("   - Extract the .tar.gz file")
        print()
        print("4. Copy the database file:")
        print(f"   cp GeoLite2-City_*/GeoLite2-City.mmdb {db_info['path']}")
        print()
        print("5. Location detection is automatic!")
        print("   Jarvis will attempt to detect your external IP using:")
        print("   - UPnP (queries your local router)")
        print("   - Socket routing (minimal external contact)")
        print("   - Optional single DNS query (OpenDNS) if behind CGNAT (config: location_cgnat_resolve_public_ip=true)")
        print()
        print("   If automatic detection fails, manually configure:")
        print("   Add to ~/.config/jarvis/config.json:")
        print('   {')
        print('     "location_auto_detect": false,')
        print('     "location_ip_address": "YOUR_PUBLIC_IP_HERE"')
        print('   }')
        print()
        print("   💡 To find your public IP: https://whatismyipaddress.com")
        print()
        print("6. Run this script again to test the setup")

        # Create directory if it doesn't exist
        db_info['directory'].mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Created directory: {db_info['directory']}")


def test_location_features():
    """Test the location detection features."""
    if not JARVIS_AVAILABLE:
        print("❌ Cannot test: Jarvis modules not available")
        return False

    print("\n🔍 Testing location features...")

    # Test if location is available
    if not is_location_available():
        print("❌ Location database not available")
        return False

    # Test automatic external IP detection
    print("Testing automatic external IP detection...")
    external_ip = _get_external_ip_automatically()
    if external_ip:
        print(f"✅ External IP automatically detected: {external_ip}")
    else:
        print("⚠️  Automatic IP detection failed")
        print("💡 You may need to manually configure 'location_ip_address'")

    # Test local IP detection (fallback)
    print("\nTesting local IP detection (fallback)...")
    local_ip = _get_local_network_ip()
    if local_ip:
        print(f"✅ Local IP detected: {local_ip}")
    else:
        print("⚠️  Could not detect local IP")

    # Test location detection
    try:
        location = get_location_info(settings=SETTINGS)
        if "error" in location:
            print(f"⚠️  Location detection result: {location['error']}")
            reason = location.get("reason")
            advice = location.get("advice")
            if reason == "cgnat_not_found":
                print("💡 Carrier-grade NAT (100.64.0.0/10) and IP not in GeoLite2. Cannot derive precise location.")
                print("   Configure a real public IP in ~/.config/jarvis/config.json:")
                print("   { 'location_ip_address': 'YOUR_PUBLIC_IP', 'location_auto_detect': false }")
            elif reason == "not_found":
                print("💡 IP not found in free GeoLite2 dataset. It may be new or CGNAT.")
            elif "No IP address available" in location['error']:
                print("💡 No IP available. Provide 'location_ip_address' in config.")
            if advice:
                print(f"   Advice: {advice}")
            return False

        print("✅ Location detection working!")
        print(f"   IP: {location.get('ip', 'Unknown')}")
        print(f"   Location: {location.get('city', 'Unknown')}, {location.get('region', '')}, {location.get('country', 'Unknown')}")

        if location.get('latitude') and location.get('longitude'):
            print(f"   Coordinates: {location['latitude']}, {location['longitude']}")

        if location.get('timezone'):
            print(f"   Timezone: {location['timezone']}")

        return True

    except Exception as e:
        print(f"❌ Location test error: {e}")
        return False


def create_test_config():
    """Create a test configuration file with location enabled."""
    config_path = Path.home() / ".config" / "jarvis" / "config.json"

    if config_path.exists():
        print(f"✅ Config file already exists: {config_path}")
        print("To enable location features, add to your config:")
        print('  "location_ip_address": "YOUR_PUBLIC_IP_HERE"')
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)

    test_config = {
        "location_enabled": True,
        "location_cache_minutes": 60,
        "location_ip_address": None,
        "location_auto_detect": True,
        "voice_debug": True
    }

    import json
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)

    print(f"✅ Created test config: {config_path}")
    print("💡 Location features will auto-detect your IP address")
    print("   If auto-detection fails, manually set 'location_ip_address'")


def main():
    """Main setup function."""
    print("🌍 Jarvis Geolocation Setup")

    # Check dependencies
    if not check_dependencies():
        print("❌ geoip2 library not found")
        print("Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            sys.exit(1)
        print("✅ Dependencies installed")
    else:
        print("✅ Dependencies available")

    # Print setup instructions
    print_setup_instructions()

    # Test if everything is working
    db_info = get_database_info()
    if db_info['exists']:
        test_success = test_location_features()

        if test_success:
            print("\n🎉 Geolocation setup complete!")
            print("Location metadata will now be included in agent context.")
        else:
            print("\n⚠️  Database exists but testing failed")
            print("Please check the database file is valid.")
    else:
        print("\n⏳ Database not found - follow the instructions above")

    print("\n💡 Privacy Note: Jarvis respects your privacy by:")
    print("   - Using UPnP (local router) and socket routing instead of third-party services")
    print("   - Working entirely with local databases")
    print("   - Giving you full control over IP detection methods")
    print("\n💡 Tip: Set JARVIS_VOICE_DEBUG=1 to see location info in debug output")


if __name__ == "__main__":
    main()
