#!/usr/bin/env python3
"""
Script to generate example configuration files from the default values in config.py.
This ensures config examples stay in sync with the actual defaults.
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import jarvis modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from jarvis.config import export_example_config


def generate_config_example() -> None:
    """Generate examples/config.json from defaults."""
    config = export_example_config(include_db_path=False)
    
    # Generate the config file
    config_path = project_root / "examples" / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")  # Add trailing newline
    
    print(f"Generated {config_path}")


def main() -> None:
    """Generate all example configuration files."""
    print("Generating configuration examples from defaults...")
    
    generate_config_example()
    
    print("\nDone! Example files are now in sync with config.py defaults.")


if __name__ == "__main__":
    main()
