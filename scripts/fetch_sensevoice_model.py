"""
Fetch the SenseVoiceSmall model weights for bundling with the app.

Jarvis ships the speech-recognition weights inside the installer so users
never download them at runtime (no HuggingFace/ModelScope wrestle). This
script downloads the weights once into ``models/SenseVoiceSmall`` at the
repository root; the PyInstaller build (``jarvis_desktop.spec``) bundles
that directory, and the app resolves it via
``jarvis.listening.sensevoice.bundled_model_dir``.

CI (``.github/workflows/build-desktop.yml``) runs this before every
desktop build, restoring the weights from ``actions/cache`` (keyed on
this script) so the ~940 MB are downloaded once and reused across
builds. The script is idempotent: it skips the download when the model
directory already holds the essential files (``config.yaml`` +
``model.pt``), which is what a cache hit looks like.

Usage:
    python scripts/fetch_sensevoice_model.py            # HuggingFace (default)
    python scripts/fetch_sensevoice_model.py --hub ms   # ModelScope
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repository root: scripts/ is one level below the root.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "SenseVoiceSmall"

# FunAudioLLM/SenseVoiceSmall is the HuggingFace mirror; iic/SenseVoiceSmall
# is the ModelScope id. Both carry the same weights (234M params, ~940 MB).
DEFAULT_ID = "FunAudioLLM/SenseVoiceSmall"
MODELSCOPE_ID = "iic/SenseVoiceSmall"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub",
        choices=["hf", "ms"],
        default="hf",
        help="Download source: 'hf' = HuggingFace (default), 'ms' = ModelScope",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id (default: FunAudioLLM/SenseVoiceSmall "
        "on HuggingFace, iic/SenseVoiceSmall on ModelScope)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the funasr import check after download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the model directory already looks complete",
    )
    args = parser.parse_args()

    # Idempotent by default: CI restores the weights from actions/cache and
    # devs re-run this script without paying for a re-download. The essential
    # inference files are config.yaml + model.pt; anything less means an
    # interrupted download and we re-fetch.
    if not args.force and (MODEL_DIR / "config.yaml").exists() and (MODEL_DIR / "model.pt").exists():
        size_mb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"✅ SenseVoiceSmall already present in {MODEL_DIR} ({size_mb:.0f} MB), skipping download")
        print("   Use --force to re-download.")
        return 0

    model_id = args.model or (MODELSCOPE_ID if args.hub == "ms" else DEFAULT_ID)
    print(f"📥 Fetching SenseVoiceSmall weights ({model_id}) ...")
    print(f"   Target: {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.hub == "ms":
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("   ❌ modelscope not installed. Run: pip install modelscope", file=sys.stderr)
            return 1
        snapshot_download(model_id, local_dir=str(MODEL_DIR))
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("   ❌ huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
            return 1
        snapshot_download(repo_id=model_id, local_dir=str(MODEL_DIR))

    size_mb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"✅ Downloaded {size_mb:.0f} MB into {MODEL_DIR}")
    print("   The next build_installer run bundles these weights with the app.")

    if not args.no_verify:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        try:
            from jarvis.listening.sensevoice import is_available

            if is_available():
                print("✅ funasr is importable — speech recognition will be ready in the build.")
            else:
                print("   ⚠️ funasr is not installed in this environment (pip install funasr).")
        except Exception as exc:  # pragma: no cover - best-effort check
            print(f"   ⚠️ Could not verify funasr: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
