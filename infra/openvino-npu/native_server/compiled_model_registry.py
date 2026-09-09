from __future__ import annotations

import hashlib
import json
from pathlib import Path


def model_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_dir.glob("*.xml")) + sorted(model_dir.glob("*.bin")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def cache_root(base: Path, openvino_version: str, driver_version: str, model_dir: Path, shape: str, compile_properties: dict[str, str] | None = None) -> Path:
    properties = {"device": "NPU", "performance_hint": "LATENCY", "inference_precision": "default", "num_streams": "1", "execution_mode": "serialized", **(compile_properties or {})}
    property_hash = hashlib.sha256(json.dumps(properties, sort_keys=True).encode()).hexdigest()[:12]
    root = base / "compiled" / f"ov-{openvino_version}" / f"driver-{driver_version}" / f"{shape}-{model_hash(model_dir)}-props-{property_hash}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"openvino": openvino_version, "driver": driver_version, "model_hash": model_hash(model_dir), "shape": shape, "compile_properties": properties}, indent=2), encoding="utf-8")
    return root
