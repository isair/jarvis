"""Conservative develop installer gate for GitHub Actions."""

import os
from pathlib import Path
import re
import subprocess


SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def requires_build(event: str, before: str, after: str) -> bool:
    """Skip only a complete push diff containing exclusively test paths."""
    if event != "push" or not SHA.fullmatch(before) or not SHA.fullmatch(after):
        return True
    try:
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
             "--name-only", "-z", before, after, "--"],
            capture_output=True, check=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return any(path and not path.startswith(b"tests/")
               for path in diff.stdout.split(b"\0"))


def main() -> None:
    required = requires_build(os.environ.get("GITHUB_EVENT_NAME", ""),
                              os.environ.get("BEFORE_SHA", ""),
                              os.environ.get("GITHUB_SHA", ""))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"required={str(required).lower()}\n")
    print("📦 Installer build required" if required else
          "⏭️ Installer build skipped: no changes outside tests")


if __name__ == "__main__":
    main()
