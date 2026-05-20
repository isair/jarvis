"""Whisper model initialisation for VoiceListener."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from ..debug import debug_log

if TYPE_CHECKING:
    from .listener import VoiceListener

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .listener import (
    FASTER_WHISPER_AVAILABLE,
    MLX_WHISPER_AVAILABLE,
    WhisperModel,
    _clear_corrupted_whisper_cache,
    _get_mlx_model_repo,
    _is_faster_whisper_turbo_supported,
    _print_cuda_unavailable_hint,
    _probe_windows_cuda_libraries,
    mlx_whisper,
)


def initialise_whisper(listener: "VoiceListener") -> bool:
    """Load Whisper on listener. Returns False on fatal error."""
    if getattr(listener, "_whisper_ready", False):
        return True
    listener._whisper_backend = listener._determine_whisper_backend()
    model_name = getattr(listener.cfg, "whisper_model", "small")

    # Validate large-v3-turbo support for faster-whisper backend
    if model_name == "large-v3-turbo" and listener._whisper_backend != "mlx":
        if not _is_faster_whisper_turbo_supported():
            debug_log(
                "faster-whisper does not support large-v3-turbo, "
                "falling back to large-v3", "voice",
            )
            print(
                "  ⚠️  large-v3-turbo is not supported by the installed Whisper engine, "
                "using large-v3 instead", flush=True,
            )
            model_name = "large-v3"

    if listener._whisper_backend == "mlx":
        if not MLX_WHISPER_AVAILABLE:
            debug_log("MLX Whisper not available", "voice")
            print("  ❌ MLX Whisper not available. Install with: pip install mlx-whisper", flush=True)
            return False

        listener._mlx_model_repo = _get_mlx_model_repo(model_name)
        print(f"     🎤 Loading MLX Whisper '{model_name}' (Apple Silicon GPU)...", flush=True)

        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                # Pre-load the model by doing a warmup transcription.
                # Use low-amplitude noise (not silence) so the decoder actually runs —
                # silent audio trips the no-speech short-circuit and leaves the decode
                # path cold, so the first real utterance still pays the full cost.
                if np is not None:
                    rng = np.random.default_rng(0)
                    warmup_audio = rng.standard_normal(listener._samplerate).astype(np.float32) * 0.01
                    _ = mlx_whisper.transcribe(
                        warmup_audio,
                        path_or_hf_repo=listener._mlx_model_repo,
                        language=None,
                    )
                    debug_log(f"MLX Whisper model pre-loaded: repo={listener._mlx_model_repo}", "voice")

                print(f"     🎤 MLX Whisper '{model_name}' ready (Apple Silicon GPU)", flush=True)
                break
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limited = (
                    any(x in error_str for x in ["429", "too many requests", "rate limit"])
                    or getattr(getattr(e, "response", None), "status_code", None) == 429
                )
                if is_rate_limited and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    debug_log(f"rate limited loading MLX Whisper (attempt {attempt + 1}): {e}", "voice")
                    print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait)
                    continue
                debug_log(f"failed to initialise MLX Whisper: {e}", "voice")
                print(f"  ❌ Failed to initialise MLX Whisper: {e}", flush=True)
                if is_rate_limited:
                    print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                return False
    else:
        # faster-whisper backend
        if not FASTER_WHISPER_AVAILABLE:
            debug_log("faster-whisper not available", "voice")
            print("  ❌ faster-whisper not available. Install with: pip install faster-whisper", flush=True)
            return False

        device = getattr(listener.cfg, "whisper_device", "auto")
        compute = getattr(listener.cfg, "whisper_compute_type", "int8")

        # On Windows, probe for CUDA runtime libraries before trying to
        # use them. faster-whisper/CTranslate2 lazily loads cuBLAS and
        # cuDNN during transcription, so without this check a model
        # that loaded fine on cuda will crash on the first audio chunk.
        resolved_device, missing_libs = _probe_windows_cuda_libraries(device)
        if missing_libs:
            _print_cuda_unavailable_hint(missing_libs)
        device = resolved_device

        # Build list of (device, compute_type) combinations to try
        # This handles both compute type fallbacks and CUDA -> CPU fallbacks
        configs_to_try = []

        # Start with preferred config
        compute_types = [compute]
        if compute == "int8":
            compute_types.extend(["float16", "float32"])
        elif compute == "float16":
            compute_types.append("float32")

        # Add preferred device with all compute types
        for ct in compute_types:
            configs_to_try.append((device, ct))

        # If device is "auto" or "cuda", add CPU fallback configs
        # This handles Windows without CUDA libraries
        if device in ("auto", "cuda"):
            for ct in compute_types:
                configs_to_try.append(("cpu", ct))

        last_error = None
        used_device = device
        used_compute = compute
        for try_device, try_compute in configs_to_try:
            try:
                cpu_threads = (os.cpu_count() or 4) if try_device in ("cpu", "auto") else 0
                print(f"     🎤 Loading Whisper '{model_name}' (device={try_device}, compute={try_compute})...", flush=True)
                listener.model = WhisperModel(
                    model_name, device=try_device, compute_type=try_compute,
                    cpu_threads=cpu_threads,
                )
                listener._apply_whisper_load_success(
                    model_name, try_device, try_compute,
                    device, compute, cpu_threads,
                )
                used_device = try_device
                used_compute = try_compute
                last_error = None
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a CUDA/GPU-related error that we should fall back from
                is_cuda_error = any(x in error_str for x in [
                    "cuda", "cublas", "cudnn", "gpu", "nvidia",
                    ".dll is not found", "library", "ctypes"
                ])
                is_compute_error = any(x in error_str for x in [
                    "compute type", "int8", "float16"
                ])

                if is_cuda_error or is_compute_error:
                    debug_log(f"config ({try_device}, {try_compute}) failed, trying fallback: {e}", "voice")
                    continue

                # Check for corrupted model cache (e.g. interrupted download)
                is_corrupted_cache = "unable to open file" in error_str

                if is_corrupted_cache:
                    debug_log(f"detected corrupted Whisper model cache: {e}", "voice")
                    print("  ⚠️  Whisper model cache appears corrupted, attempting recovery...", flush=True)

                    cache_cleared = _clear_corrupted_whisper_cache(str(e))
                    if cache_cleared:
                        try:
                            print(f"     🎤 Re-downloading Whisper '{model_name}'...", flush=True)
                            listener.model = WhisperModel(
                                model_name, device=try_device, compute_type=try_compute,
                                cpu_threads=cpu_threads,
                            )
                            listener._apply_whisper_load_success(
                                model_name, try_device, try_compute,
                                device, compute, cpu_threads,
                                context="recovered",
                            )
                            used_device = try_device
                            used_compute = try_compute
                            last_error = None
                            break
                        except Exception as retry_e:
                            debug_log(f"retry after cache clear also failed: {retry_e}", "voice")
                            print(f"  ❌ Failed to load Whisper model after cache recovery: {retry_e}", flush=True)
                            return
                    else:
                        debug_log("could not clear corrupted cache automatically", "voice")
                        print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                        print("  💡 Try manually deleting the Whisper model cache directory and restarting", flush=True)
                        return False
                # Check for rate limiting (HTTP 429) — check string and response status code
                # (HfHubHTTPError may carry the status on .response without "429" in str(e))
                is_rate_limited = (
                    any(x in error_str for x in ["429", "too many requests", "rate limit"])
                    or getattr(getattr(e, "response", None), "status_code", None) == 429
                )

                if is_rate_limited:
                    _max_retries = 4
                    _backoff = 2
                    debug_log(f"rate limited loading Whisper model: {e}", "voice")
                    retry_succeeded = False
                    for retry_num in range(1, _max_retries + 1):
                        wait = _backoff ** retry_num
                        print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({retry_num}/{_max_retries})...", flush=True)
                        time.sleep(wait)
                        try:
                            listener.model = WhisperModel(
                                model_name, device=try_device, compute_type=try_compute,
                                cpu_threads=cpu_threads,
                            )
                            listener._apply_whisper_load_success(
                                model_name, try_device, try_compute,
                                device, compute, cpu_threads,
                                context="rate-limit retry",
                            )
                            used_device = try_device
                            used_compute = try_compute
                            last_error = None
                            retry_succeeded = True
                            break
                        except Exception as retry_e:
                            debug_log(f"rate-limit retry {retry_num} failed: {retry_e}", "voice")
                            last_error = retry_e
                    if retry_succeeded:
                        break
                    debug_log(f"gave up after {_max_retries} rate-limit retries", "voice")
                    print(f"  ❌ Failed to load Whisper model after {_max_retries} retries: {last_error}", flush=True)
                    print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                    return
                else:
                    # For other errors (model not found, etc.), don't try fallbacks
                    debug_log(f"failed to initialise faster-whisper: {e}", "voice")
                    print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                    return

        if last_error is not None:
            debug_log(f"failed to initialise faster-whisper with any config: {last_error}", "voice")
            print(f"  ❌ Failed to load Whisper model: {last_error}", flush=True)
            return False

        # Warm up faster-whisper so the first real utterance doesn't pay
        # the cold-decode cost. Use low-amplitude noise rather than pure
        # silence — silence trips faster-whisper's no-speech short-circuit
        # and the decoder never actually runs. Mirror the real transcribe
        # parameters so beam search, language detection, and the timestamp
        # path are all exercised here instead of on the user's first word.
        if np is not None and listener.model is not None:
            try:
                cpu_mode = listener._whisper_device == "cpu"
                rng = np.random.default_rng(0)
                warmup_audio = rng.standard_normal(listener._samplerate).astype(np.float32) * 0.01
                try:
                    segments_iter, _ = listener.model.transcribe(
                        warmup_audio,
                        language=None,
                        vad_filter=False,
                        condition_on_previous_text=not cpu_mode,
                        without_timestamps=cpu_mode,
                    )
                except TypeError:
                    segments_iter, _ = listener.model.transcribe(warmup_audio, language=None)
                for _ in segments_iter:
                    pass
                debug_log("faster-whisper warmup transcription complete", "voice")
            except Exception as e:
                debug_log(f"faster-whisper warmup failed: {e}", "voice")


    listener._whisper_ready = True
    return True
