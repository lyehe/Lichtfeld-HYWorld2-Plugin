"""Self-calibrating VRAM model for hyworld2 inference.

Replaces the fixed-constant bytes-per-pixel-per-frame estimate with values
learned from the user's actual machine. After each successful inference,
we record:

  model_bytes        : VRAM held by the WorldMirror weights (measured once
                       at pipeline construction).
  activation_bytes   : peak_bytes - model_bytes  (transient forward-pass
                       memory).
  pixels             : frames * H * W (where H, W are the model's inference
                       tensor dims, *not* the disk image dims).
  activation_bpp     : activation_bytes / pixels

The profile is EWMA-smoothed (alpha=0.3) across runs so a single outlier
doesn't skew the estimate. Stored at ``<plugin>/models/vram_profile.json``
so it survives across sessions on the same machine. Profile is keyed by
precision (``bf16`` vs ``fp32``) since activation size depends heavily on
that.
"""
from __future__ import annotations

import json
import threading

from . import downloads  # for the plugin-local models dir

_LOCK = threading.Lock()
_PROFILE_FILE = downloads.MODELS_DIR / "vram_profile.json"

# Conservative starting point if no calibration exists yet. Doubled from
# the initial empirical guess after we saw real-world OOMs with 1600/3200.
_DEFAULT_BPP_BF16 = 3200
_DEFAULT_BPP_FP32 = 6400

# EWMA factor applied when merging a new measurement into the saved value.
_EWMA_ALPHA = 0.3

# Multiply the learned value by this before budgeting, to absorb spikes
# not captured by max_memory_allocated() (e.g. fragmentation, pool
# overhead, transient kernels).
_SAFETY_FACTOR = 1.15


def _load() -> dict:
    with _LOCK:
        try:
            return json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _save(profile: dict) -> None:
    with _LOCK:
        try:
            _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        except Exception:
            pass


def get_bpp(bf16: bool) -> float:
    """Return calibrated bytes-per-pixel-per-frame, or the default."""
    key = "bf16" if bf16 else "fp32"
    profile = _load()
    entry = profile.get(key)
    if entry and "bpp" in entry and entry["bpp"] > 0:
        return float(entry["bpp"])
    return _DEFAULT_BPP_BF16 if bf16 else _DEFAULT_BPP_FP32


def get_model_bytes(bf16: bool) -> int:
    """Calibrated resident model weight bytes (0 if unknown)."""
    key = "bf16" if bf16 else "fp32"
    profile = _load()
    entry = profile.get(key)
    return int(entry.get("model_bytes", 0)) if entry else 0


def safety_factor() -> float:
    return _SAFETY_FACTOR


def record_model_bytes(bf16: bool, model_bytes: int) -> None:
    """Call once after pipeline construction with measured weight memory."""
    if model_bytes <= 0:
        return
    key = "bf16" if bf16 else "fp32"
    profile = _load()
    entry = profile.get(key, {})
    entry["model_bytes"] = int(model_bytes)
    profile[key] = entry
    _save(profile)


def record_run(bf16: bool, frames: int, h: int, w: int,
               peak_bytes: int, model_bytes: int,
               pre_alloc_bytes: int = 0, log=None) -> None:
    """Update the calibrated profile with a run's actual peak memory.

    ``peak_bytes``     : torch.cuda.max_memory_allocated() at end of inference.
    ``model_bytes``    : bytes held by the pipeline weights (constant).
    ``pre_alloc_bytes``: bytes already allocated right before the forward pass
                         — isolates this run's activation footprint from
                         accumulated scene state / other plugins / LFS buffers.
                         Without this, the measured peak drifts upward over a
                         session as more splats accumulate in the scene.
    """
    if frames <= 0 or h <= 0 or w <= 0:
        return
    activation = max(0, int(peak_bytes) - int(model_bytes) - int(pre_alloc_bytes))
    if activation <= 0:
        return
    pixels = frames * h * w
    bpp = activation / pixels

    key = "bf16" if bf16 else "fp32"
    profile = _load()
    entry = profile.get(key, {})
    prev = entry.get("bpp")
    if prev and prev > 0:
        blended = _EWMA_ALPHA * bpp + (1.0 - _EWMA_ALPHA) * prev
    else:
        blended = bpp
    entry["bpp"] = float(blended)
    entry["last_measurement"] = {
        "frames": int(frames),
        "h": int(h),
        "w": int(w),
        "peak_bytes": int(peak_bytes),
        "model_bytes": int(model_bytes),
        "bpp_raw": float(bpp),
        "bpp_blended": float(blended),
    }
    profile[key] = entry
    _save(profile)
    if log is not None:
        log(f"vram-profile: {key} bpp {prev or '?'} -> {blended:.0f} "
            f"(run: frames={frames} {w}x{h} peak={peak_bytes / 1e9:.2f}GB)")


def reset() -> None:
    """Wipe the calibration profile (e.g. after a hardware change)."""
    with _LOCK:
        try:
            _PROFILE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
