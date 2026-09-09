"""
VRAM detection and model recommendation.

Cross-platform GPU memory detection with a preferred DXGI path on Windows
and ``nvidia-smi`` fallback on other platforms.  Provides model
recommendations based on available VRAM so the setup wizard and startup
flow can warn users whose GPU doesn't meet the default model's requirements.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..debug import debug_log

# ---------------------------------------------------------------------------
# Supported models & their VRAM thresholds (in MB)
# ---------------------------------------------------------------------------
# Each entry: (model_id, display_name, vram_mb_required, is_low_vram_option)
#
# Derived from SUPPORTED_CHAT_MODELS in ``jarvis.config`` plus the explicit
# low-VRAM fallback below — when a new model is added to the config, it
# appears here automatically (to keep the two in sync, every model in
# SUPPORTED_CHAT_MODELS must list a parseable ``vram`` string like "8GB+").
# The ``is_low_vram_option`` flag is set only for entries whose VRAM is below
# the default model's requirement.

def _build_vram_table() -> list[tuple[str, str, int, bool]]:
    """Build the VRAM table from ``jarvis.config.SUPPORTED_CHAT_MODELS``."""
    import re
    from jarvis.config import SUPPORTED_CHAT_MODELS, DEFAULT_CHAT_MODEL

    def _parse_vram_mb(vram_str: str) -> int:
        match = re.search(r"(\d+)", vram_str)
        if not match:
            return 99999  # unknown → highest tier
        return int(match.group(1)) * 1024  # "8GB+" → 8192

    default_vram = _parse_vram_mb(
        SUPPORTED_CHAT_MODELS.get(DEFAULT_CHAT_MODEL, {}).get("vram", "8GB+")
    )

    table: list[tuple[str, str, int, bool]] = []
    for model_id, info in SUPPORTED_CHAT_MODELS.items():
        vram_mb = _parse_vram_mb(info.get("vram", "8GB+"))
        name = info.get("name", model_id)
        is_low = vram_mb < default_vram
        table.append((model_id, name, vram_mb, is_low))

    # Sort by VRAM ascending so the module-level table is predictable;
    # get_recommended_model_id iterates from highest to lowest.
    table.sort(key=lambda x: x[2])
    return table


_MODEL_VRAM_TABLE: list[tuple[str, str, int, bool]] = _build_vram_table()

# Model to recommend when VRAM is below the default requirement.
# ``qwen3.5:0.8b`` is a tiny 873M-parameter agentic model that runs
# comfortably on 2 GB+ VRAM or CPU.  It is already included in SUPPORTED_CHAT_MODELS
# and thus in _MODEL_VRAM_TABLE above — this constant is a convenience
# reference so callers can compare against a known stable name.
_LOW_VRAM_OPTION: tuple[str, str, int, bool] | None = None
for _entry in _MODEL_VRAM_TABLE:
    if _entry[3]:  # is_low_vram_option
        _LOW_VRAM_OPTION = _entry
        break

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_total_vram_mb() -> Optional[int]:
    """Return total dedicated video memory in MB for the primary GPU, or
    ``None`` when detection is unavailable / fails.

    Resolution order:
    1. Windows → DXGI (``dxgi.dll`` COM factory → adapter description).
    2. Any platform with ``nvidia-smi`` on ``PATH``.
    3. Linux → ``/proc/driver/nvidia/gpus/*/information``.
    """
    if sys.platform == "win32":
        mb = _detect_via_dxgi()
        if mb is not None:
            return mb
        # Fall through to nvidia-smi on Windows too.

    mb = _detect_via_nvidia_smi()
    if mb is not None:
        return mb

    if sys.platform == "linux":
        mb = _detect_via_proc_nvidia()
        if mb is not None:
            return mb

    return None


def _detect_via_dxgi() -> Optional[int]:
    """Query the primary DXGI adapter's dedicated video memory via the
    COM ``IDXGIFactory1`` / ``IDXGIAdapter1::GetDesc1`` API.

    Uses raw ``ctypes`` — no third-party packages required.
    """
    # Guard: only Windows and only when the DLL exists.
    if sys.platform != "win32":
        return None
    try:
        return _dxgi_adapter_vram_mb()
    except Exception as exc:
        debug_log(f"DXGI VRAM detection failed: {exc}", "vram")
        return None


def _dxgi_adapter_vram_mb() -> Optional[int]:
    """Core DXGI COM call: enumerate the first adapter and read
    ``DedicatedVideoMemory`` from ``DXGI_ADAPTER_DESC1``.

    COM interface hierarchy (indices are vtable offsets):

    IUnknown:            [0] QueryInterface  [1] AddRef  [2] Release
    IDXGIObject (IUnknown):  + [3] SetPrivateData  [4] SetPrivateDataInterface
                              [5] GetPrivateData  [6] GetParent
    IDXGIFactory (IDXGIObject): + [7] EnumAdapters  [8] MakeWindowAssociation
                                 [9] GetWindowAssociation  [10] CreateSwapChain
                                 [11] CreateSoftwareAdapter
    IDXGIFactory1 (IDXGIFactory): + [12] EnumAdapters1  [13] IsCurrent

    IDXGIAdapter (IDXGIObject): + [7] CheckInterfaceSupport  [8] EnumOutputs
                                 [9] GetDesc
    IDXGIAdapter1 (IDXGIAdapter): + [10] GetDesc1
    """
    from ctypes import (windll, wintypes, Structure, POINTER, c_void_p,
                        c_size_t, byref, c_uint32, WINFUNCTYPE, addressof)

    class GUID(Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    # IID_IDXGIFactory1 = {7706F476-3C83-4E51-BFE0-5E143C7E1A66}
    _IID_IDXGIFACTORY1 = GUID(
        0x7706F476, 0x3C83, 0x4E51,
        (wintypes.BYTE * 8)(0xBF, 0xE0, 0x5E, 0x14, 0x3C, 0x7E, 0x1A, 0x66),
    )

    class DXGI_ADAPTER_DESC1(Structure):
        _fields_ = [
            ("Description", wintypes.WCHAR * 128),
            ("VendorId", wintypes.UINT),
            ("DeviceId", wintypes.UINT),
            ("SubSysId", wintypes.UINT),
            ("Revision", wintypes.UINT),
            ("DedicatedVideoMemory", c_size_t),
            ("DedicatedSystemMemory", c_size_t),
            ("SharedSystemMemory", c_size_t),
            ("AdapterLuid", wintypes.LARGE_INTEGER),
        ]

    # COM method type aliases
    ReleaseFunc = WINFUNCTYPE(wintypes.ULONG, c_void_p)
    EnumAdapters1Func = WINFUNCTYPE(
        wintypes.HRESULT, c_void_p, c_uint32, POINTER(c_void_p),
    )
    GetDesc1Func = WINFUNCTYPE(
        wintypes.HRESULT, c_void_p, POINTER(DXGI_ADAPTER_DESC1),
    )

    dxgi = windll.dxgi
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.restype = wintypes.HRESULT
    create_factory.argtypes = [POINTER(GUID), POINTER(c_void_p)]

    factory_ptr = c_void_p()
    hr = create_factory(byref(_IID_IDXGIFACTORY1), byref(factory_ptr))
    if hr != 0:  # S_OK
        debug_log(f"DXGI CreateDXGIFactory1 failed: HRESULT={hr:#x}", "vram")
        return None

    # SAFETY: factory_ptr is alive until we Release() it below.
    factory_vtable = POINTER(c_void_p).from_address(
        POINTER(c_void_p).from_address(factory_ptr)[0]
    )
    # Access the vtable array. Since we only need indices 2 (Release)
    # and 12 (EnumAdapters1), cast to a fixed-size array of known length.
    # We need at least 13 entries for the IDXGIFactory1 layout.
    factory_vtable_arr = (c_void_p * 14).from_address(
        addressof(factory_vtable)
    )

    release_fn = ReleaseFunc(factory_vtable_arr[2])

    # EnumAdapters1 is at vtable offset 12
    enum_adapters1_fn = EnumAdapters1Func(factory_vtable_arr[12])

    adapter_ptr = c_void_p()
    hr = enum_adapters1_fn(factory_ptr, 0, byref(adapter_ptr))
    if hr != 0 or not adapter_ptr:
        release_fn(factory_ptr)
        return None

    adapter_vtable_arr = (c_void_p * 11).from_address(
        addressof(
            POINTER(c_void_p).from_address(adapter_ptr)[0]
        )
    )
    adapter_release_fn = ReleaseFunc(adapter_vtable_arr[2])

    # GetDesc1 is at vtable offset 10
    get_desc1_fn = GetDesc1Func(adapter_vtable_arr[10])

    desc = DXGI_ADAPTER_DESC1()
    hr = get_desc1_fn(adapter_ptr, byref(desc))

    adapter_release_fn(adapter_ptr)
    release_fn(factory_ptr)

    if hr != 0:
        return None

    # DedicatedVideoMemory is in bytes → convert to MB
    return desc.DedicatedVideoMemory // (1024 * 1024)


def _detect_via_nvidia_smi() -> Optional[int]:
    """Parse ``nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits``."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return None
        return int(lines[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        return None


def _detect_via_proc_nvidia() -> Optional[int]:
    """Parse ``/proc/driver/nvidia/gpus/*/information`` on Linux."""
    import glob
    try:
        info_files = glob.glob("/proc/driver/nvidia/gpus/*/information")
        if not info_files:
            return None
        for path in info_files:
            with open(path, "r") as f:
                text = f.read()
            # "Total Memory: 12288 MiB"
            m = re.search(r"Total Memory:\s*(\d+)\s*MiB", text)
            if m:
                return int(m.group(1))
    except (OSError, IOError):
        pass
    return None


# ---------------------------------------------------------------------------
# Model recommendation
# ---------------------------------------------------------------------------


def get_recommended_model_id(total_vram_mb: Optional[int]) -> str:
    """Return the model ID best suited for the given VRAM.

    When VRAM is unknown (``None``), or is at least 8 GB, the default
    ``gemma4:e2b`` is returned.  When VRAM is below 8 GB the low-VRAM
    option (``qwen3.5:0.8b``) is returned instead.
    """
    if total_vram_mb is None:
        return "gemma4:e2b"  # safe default — user can still override

    # Scan from highest-VRAM to lowest-VRAM so we prefer the most
    # capable model that fits.  The table is sorted ascending, so
    # iterate in reverse.
    for model_id, _name, vram_mb, _low_vram in reversed(_MODEL_VRAM_TABLE):
        if total_vram_mb >= vram_mb:
            return model_id

    # Not enough VRAM for any tier → suggest the lowest-VRAM option
    if _LOW_VRAM_OPTION is not None:
        return _LOW_VRAM_OPTION[0]
    # Fallback: the model with the lowest requirement
    return _MODEL_VRAM_TABLE[0][0] if _MODEL_VRAM_TABLE else "gemma4:e2b"


def format_vram_warning(total_vram_mb: Optional[int],
                        model_id: str) -> Optional[str]:
    """Return a user-facing warning string when ``model_id`` exceeds
    the available VRAM, or ``None`` if everything fits.

    The message mentions the low-VRAM alternative when one exists for
    the user's VRAM tier.
    """
    if total_vram_mb is None:
        return None  # can't judge

    required = required_vram_mb(model_id)
    if required is None:
        return None  # unknown model — no warning

    if total_vram_mb >= required:
        return None

    if _LOW_VRAM_OPTION is not None:
        low_vram_id, low_vram_name, low_vram_req, _ = _LOW_VRAM_OPTION
        return (
            f"⚠️ Your GPU has {total_vram_mb} MB VRAM, but "
            f"{model_id} recommends {required} MB. "
            f"Consider switching to {low_vram_id} ({low_vram_name}) which "
            f"fits in {low_vram_req} MB."
        )
    return (
        f"⚠️ Your GPU has {total_vram_mb} MB VRAM, but "
        f"{model_id} recommends {required} MB."
    )


def required_vram_mb(model_id: str) -> Optional[int]:
    """Look up the VRAM requirement for a model ID.

    Returns MB or ``None`` for unknown models.

    The low-VRAM model is already included in ``_MODEL_VRAM_TABLE``
    (derived from ``SUPPORTED_CHAT_MODELS``), so the loop catches
    every registered model. Only completely unknown IDs return ``None``.
    """
    for mid, _name, vram_mb, _low_vram in _MODEL_VRAM_TABLE:
        if mid == model_id:
            return vram_mb
    return None


# ---------------------------------------------------------------------------
# Shared-GPU VRAM budget: chat LLM + Whisper resident on one CUDA device
# ---------------------------------------------------------------------------
# The chat model and Whisper share a single CUDA context pool, so the useful
# figure is not each model's own requirement but the *sum* that has to fit.
# These helpers build that sum from real numbers where possible: the on-disk
# GGUF size for the LLM and the shipped CTranslate2 weight size for Whisper.

#: CTranslate2 ``int8`` weight sizes (MB) — the ``model.bin`` of each shipped
#: faster-whisper snapshot. ``float16`` / ``float32`` are scaled from these.
_WHISPER_INT8_WEIGHTS_MB: Dict[str, int] = {
    "tiny": 75,
    "base": 143,
    "small": 243,
    "medium": 397,
    "large-v3": 631,
    "large-v3-turbo": 484,
}

#: Weight-size multipliers relative to ``int8`` (8-bit + fp16 scales).
_WEIGHTS_FACTOR: Dict[str, float] = {
    "int8": 1.0,
    "int8_float16": 1.1,
    "float16": 1.9,
    "bfloat16": 1.9,
    "float32": 3.8,
}

#: Static per-process overhead of the resident runtime (CUDA context plus
#: decoder KV / activation buffers). Kept separate from the weights so the
#: arithmetic stays traceable.
_WHISPER_RUNTIME_MB = 128
_LLM_RUNTIME_MB = 256

#: KV-cache + activation slack on top of the on-disk GGUF weights.
_LLM_KV_SLACK = 1.10

#: Extra slack kept on top of the summed requirement.
_BUDGET_MARGIN_MB = 256


def _fmt_mb(mb: Optional[int]) -> str:
    """Human-readable MB/GB string (``None`` → ``?``)."""
    if mb is None:
        return "?"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def _weights_factor(compute_type: str) -> float:
    return _WEIGHTS_FACTOR.get(str(compute_type or "int8").strip().lower(), 1.0)


def whisper_weights_mb(model_id: str, compute_type: str = "int8") -> Optional[int]:
    """Weight footprint of a Whisper model in MB for ``compute_type``."""
    key = str(model_id or "").strip()
    if not key:
        return None
    base = _WHISPER_INT8_WEIGHTS_MB.get(key)
    if base is None:
        # Strip a trailing language tag: "small.en" / "medium.en" share the
        # multilingual checkpoint size.
        base = _WHISPER_INT8_WEIGHTS_MB.get(key.split(".")[0])
    if base is None:
        return None
    return int(round(base * _weights_factor(compute_type)))


def _chat_weights_mb(chat_model: str) -> tuple[Optional[int], bool]:
    """Return ``(weights_mb, is_catalog_total)`` for the configured chat model.

    A filesystem path (LM Studio / llama.cpp GGUF) is measured directly off
    disk, so the figure is exact. A pull-style name is resolved through
    ``SUPPORTED_CHAT_MODELS``, whose value is already a whole-model budget —
    the ``is_catalog_total`` flag stops us from double-counting the runtime.
    """
    name = str(chat_model or "").strip()
    if not name:
        return None, False
    try:
        path = Path(name)
        if path.suffix.lower() in (".gguf", ".bin") and path.exists():
            size = path.stat().st_size
            if size:
                return max(1, size // (1024 * 1024)), False
    except OSError:
        pass
    table_mb = required_vram_mb(name)
    if table_mb is not None:
        return table_mb, True
    return None, False


def detect_used_vram_mb() -> Optional[int]:
    """Currently occupied VRAM in MB (``nvidia-smi``), or ``None``.

    DXGI's adapter descriptor exposes only the total, so the measured "used"
    value comes from ``nvidia-smi`` when it is on ``PATH``. Fail-open.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return None
        return int(lines[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        return None


def estimate_cuda_vram_plan(
    chat_model: str,
    whisper_model: str,
    compute_type: str = "int8",
) -> Dict[str, Any]:
    """Sum the resident VRAM demand of the chat LLM and Whisper model.

    Returns a dict with, in MB: the detected ``total_mb``, each model split
    into weights plus runtime, the summed ``required_mb``, the ``keep_free_mb``
    figure (requirement + margin) to leave unallocated, and the resulting
    ``headroom_mb`` against ``total_mb`` plus a ``fits`` flag. Unknown inputs
    stay ``None`` so the caller can print whatever *is* known.
    """
    plan: Dict[str, Any] = {
        "total_mb": detect_total_vram_mb(),
        "used_mb": detect_used_vram_mb(),
        "llm_weights_mb": None,
        "llm_runtime_mb": 0,
        "llm_total_mb": 0,
        "whisper_weights_mb": None,
        "whisper_runtime_mb": 0,
        "whisper_total_mb": 0,
        "required_mb": 0,
        "margin_mb": _BUDGET_MARGIN_MB,
        "keep_free_mb": 0,
        "headroom_mb": None,
        "fits": None,
        "notes": [],
    }

    weights, from_catalog = _chat_weights_mb(chat_model)
    if weights is not None:
        if from_catalog:
            # Catalogue value is already a whole-model budget.
            plan["llm_weights_mb"] = weights
            plan["llm_total_mb"] = weights
            plan["notes"].append("chat figure from model catalogue (incl. runtime)")
        else:
            runtime = _LLM_RUNTIME_MB + int(
                round(weights * (_LLM_KV_SLACK - 1.0))
            )
            plan["llm_weights_mb"] = weights
            plan["llm_runtime_mb"] = runtime
            llm_total = weights + runtime
            plan["llm_total_mb"] = llm_total
            plan["notes"].append(
                f"chat weights measured from GGUF file; +{runtime} MB KV/runtime"
            )

    w_weights = whisper_weights_mb(whisper_model, compute_type)
    if w_weights is not None:
        plan["whisper_weights_mb"] = w_weights
        plan["whisper_runtime_mb"] = _WHISPER_RUNTIME_MB
        plan["whisper_total_mb"] = w_weights + _WHISPER_RUNTIME_MB
        plan["notes"].append(
            f"whisper {compute_type} weights + {_WHISPER_RUNTIME_MB} MB CT2 runtime"
        )

    required = int(plan["llm_total_mb"]) + int(plan["whisper_total_mb"])
    plan["required_mb"] = required
    if required:
        plan["keep_free_mb"] = required + _BUDGET_MARGIN_MB

    total = plan["total_mb"]
    if total and required:
        plan["headroom_mb"] = int(total) - required
        plan["fits"] = bool((total - required) >= 0)
    return plan


def format_cuda_vram_budget(plan: Dict[str, Any]) -> str:
    """Render :func:`estimate_cuda_vram_plan` output as indented log lines."""
    if not plan:
        return ""

    lines: list[str] = []
    total = plan.get("total_mb")
    used = plan.get("used_mb")
    if total:
        suffix = f" · nvidia-smi used: {_fmt_mb(int(used))}" if used else ""
        lines.append(f"     💾 GPU VRAM total: {_fmt_mb(int(total))}{suffix}")

    llm_weights = plan.get("llm_weights_mb")
    if llm_weights:
        llm_runtime = int(plan.get("llm_runtime_mb") or 0)
        detail = f"{_fmt_mb(int(llm_weights))} weights"
        if llm_runtime:
            detail += f" + {_fmt_mb(llm_runtime)} KV/runtime (LLM)"
        lines.append(f"     🧠 Chat model resident: {detail} = "
                     f"{_fmt_mb(int(plan.get('llm_total_mb') or 0))}")

    w_weights = plan.get("whisper_weights_mb")
    if w_weights:
        lines.append(
            f"     🎤 Whisper resident: {_fmt_mb(int(w_weights))} weights + "
            f"{_fmt_mb(int(plan.get('whisper_runtime_mb') or 0))} CT2 runtime = "
            f"{_fmt_mb(int(plan.get('whisper_total_mb') or 0))}"
        )

    required = int(plan.get("required_mb") or 0)
    keep_free = int(plan.get("keep_free_mb") or 0)
    if required:
        lines.append(
            f"     📐 Combined CUDA demand: {_fmt_mb(required)} "
            f"(keep at least {_fmt_mb(keep_free)} unallocated)"
        )

    headroom = plan.get("headroom_mb")
    if headroom is not None:
        fits = plan.get("fits")
        marker = "✅" if fits else "⚠️"
        if fits:
            lines.append(
                f"     {marker} Both models fit on this GPU — "
                f"{_fmt_mb(int(headroom))} spare after both are resident"
            )
        else:
            lines.append(
                f"     {marker} Combined demand is {_fmt_mb(required)} but only "
                f"{_fmt_mb(int(total))} is installed — expect a CUDA fallback to "
                f"CPU or an out-of-memory load; use a smaller chat quant or "
                f"whisper model"
            )

    note = "; ".join(str(n) for n in plan.get("notes") or [])
    if note:
        lines.append(f"     ℹ️  {note}")
    return "\n".join(lines)

