"""Cross-platform launcher for Claude Code preview_start.

Detects the OS and delegates to the appropriate platform-specific script
(bat on Windows, sh on macOS/Linux). Can be invoked with any Python 3.x.

Usage:
    python scripts/launch.py <script_name> [args...]

Examples:
    python scripts/launch.py run_desktop_app
    python scripts/launch.py run_desktop_app --voice-debug
    python scripts/launch.py run_evals
"""

import os
import platform
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/launch.py <script_name> [args...]")
        sys.exit(1)

    script_name = sys.argv[1]
    extra_args = sys.argv[2:]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(project_root, "scripts")

    if platform.system() == "Windows":
        script_path = os.path.join(scripts_dir, f"{script_name}.bat")
        if not os.path.isfile(script_path):
            print(f"ERROR: {script_path} not found")
            sys.exit(1)
        result = subprocess.run(
            [script_path] + extra_args,
            cwd=project_root,
            shell=True,
        )
    else:
        script_path = os.path.join(scripts_dir, f"{script_name}.sh")
        if not os.path.isfile(script_path):
            print(f"ERROR: {script_path} not found")
            sys.exit(1)
        result = subprocess.run(
            ["bash", script_path] + extra_args,
            cwd=project_root,
        )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
