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


def generate_stub_json() -> None:
    """Generate examples/config.stub.json from defaults."""
    config = export_example_config(include_db_path=False)
    
    # Generate the stub config file
    stub_path = project_root / "examples" / "config.stub.json"
    with stub_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")  # Add trailing newline
    
    print(f"Generated {stub_path}")


def generate_readme_example() -> str:
    """Generate the config JSON example for README.md."""
    config = export_example_config(include_db_path=False)
    
    # For README, we want a slightly shorter example
    # Remove some less commonly changed settings
    readme_config = {k: v for k, v in config.items() if k not in [
        "voice_max_collect_seconds", "vad_aggressiveness", "vad_frame_ms",
        "vad_pre_roll_ms", "endpoint_silence_ms", "max_utterance_ms", 
        "sample_rate", "tune_enabled", "hot_window_enabled", 
        "hot_window_seconds", "dialogue_memory_timeout", "voice_min_energy"
    ]}
    
    return json.dumps(readme_config, indent=2)


def main() -> None:
    """Generate all example configuration files."""
    print("Generating configuration examples from defaults...")
    
    # Generate stub JSON
    generate_stub_json()
    
    # Generate README example and print it for manual update
    readme_example = generate_readme_example()
    print("\nGenerated README example (update manually):")
    print("```json")
    print(readme_example)
    print("```")
    
    print("\nDone! Example files are now in sync with config.py defaults.")


if __name__ == "__main__":
    main()
