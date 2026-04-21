"""Lichtfeld-HYWorld2-Plugin — HY-World-Mirror-2 reconstruction for LichtFeld Studio.

All model data (WorldMirror weights + skyseg ONNX) is scoped to this plugin
directory via ``HF_HOME`` / ``HF_HUB_CACHE`` env vars set below. Nothing
leaks to ``~/.cache/huggingface/``.

Lifecycle:
  on_load   -> register classes, kick off background model download.
  on_unload -> unregister, stop downloads, drop pipeline. Cached models
               persist across sessions; they're removed only by the
               uninstall script or the panel's "Clear cached models" button.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# LFS's embedded Python on Windows hands plugins a sys.stderr whose
# .flush() raises OSError(EINVAL). tqdm calls flush() on construction,
# so huggingface_hub's download progress bar (and any other transitive
# tqdm user) crashes on first import. Raw plugin stderr has no consumer
# here — LFS surfaces output through lf.log — so repoint it at a real
# writable sink whose flush() actually works.
if sys.stderr is not None:
    try:
        sys.stderr.flush()
    except OSError:
        sys.stderr = open(os.devnull, "w", buffering=1)

# --- Plugin-local HF cache (must be set BEFORE any huggingface_hub import) ---
_PLUGIN_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _PLUGIN_DIR / "models"
_HF_HOME = _MODELS_DIR / "huggingface"
_HF_HUB = _HF_HOME / "hub"
os.environ["HF_HOME"] = str(_HF_HOME)
os.environ["HF_HUB_CACHE"] = str(_HF_HUB)
# Keep the legacy var in sync for older HF client versions
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HF_HUB)
# Avoid polluting cwd with skyseg.onnx if any code path bypasses our cache.
os.environ.setdefault("HYWORLD2_PLUGIN_CACHE_DIR", str(_MODELS_DIR))

# Plugin-local caches for torch.compile (FX graph + Triton kernels).
# Persists compiled artifacts across sessions so the 30-60s first-forward
# cost from torch.compile only hits the very first time for each unique
# (frames, H, W) shape. Subsequent launches reload the cached graph.
_COMPILE_CACHE = _PLUGIN_DIR / "cache" / "torch_compile"
_TRITON_CACHE = _PLUGIN_DIR / "cache" / "triton"
_COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
_TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_COMPILE_CACHE))
os.environ.setdefault("TRITON_CACHE_DIR", str(_TRITON_CACHE))

# --- Linux: auto-point gsplat's JIT at a system CUDA Toolkit if the user
# hasn't already set CUDA_HOME. gsplat 1.5.3 is a pure-Python wheel that
# JIT-compiles CUDA kernels on first import, so nvcc must be resolvable.
# We check the standard install locations; first hit wins. Windows users
# get nvcc via the standalone CUDA Toolkit installer's PATH entry; we
# don't touch CUDA_HOME on Windows.
if sys.platform.startswith("linux") and "CUDA_HOME" not in os.environ:
    import glob as _glob
    for _cand in sorted(_glob.glob("/usr/local/cuda*"), reverse=True) + ["/opt/cuda"]:
        if Path(_cand, "bin", "nvcc").exists():
            os.environ["CUDA_HOME"] = _cand
            os.environ["PATH"] = f"{_cand}/bin" + os.pathsep + os.environ.get("PATH", "")
            break

# Expose the vendored `hyworld2` Python package on sys.path.
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import lichtfeld as lf  # noqa: E402

from .core import downloads, pipeline_loader  # noqa: E402
from .panels.main_panel import HYWorld2Panel  # noqa: E402

_classes = [HYWorld2Panel]


_last_training_state = False


def _on_training_state_changed(new):
    """When LFS enters training, free the WorldMirror model from VRAM.

    Signal callbacks receive only the new value. We track the previous
    state in a module-level flag to detect the rising edge.
    """
    global _last_training_state
    import threading
    is_now = bool(new)
    rising_edge = is_now and not _last_training_state
    _last_training_state = is_now
    if rising_edge and pipeline_loader.is_loaded():
        lf.log.info(
            "[hyworld2] Training started — unloading WorldMirror model "
            "to free VRAM for the trainer."
        )
        threading.Thread(target=pipeline_loader.unload, daemon=True).start()


def _apply_inference_perf_flags() -> None:
    """Bit-equivalent torch/CUDA flags lifted from the Playground backend.

    - expandable_segments: less allocator fragmentation (helps OOM-retry).
    - cudnn.benchmark: picks fastest conv algo for the static DPT head shapes.
    - TF32 on Ampere+: sub-bit drift on fp32 matmuls, real speedup.

    Gated by the HYWORLD_PERF_OFF=1 env var just like the upstream impl.
    """
    if os.environ.get("HYWORLD_PERF_OFF") == "1":
        lf.log.info("[hyworld2] HYWORLD_PERF_OFF=1, skipping inference perf flags")
        return
    applied = []
    # expandable_segments is Linux-only in torch's CUDA caching allocator;
    # on Windows it's silently ignored but emits a UserWarning. Skip there.
    if sys.platform != "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        applied.append("expandable_segments")
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        applied += ["cudnn.benchmark", "TF32"]
        lf.log.info(f"[hyworld2] perf flags: {', '.join(applied)}")
    except Exception as exc:
        lf.log.warn(f"[hyworld2] couldn't apply perf flags: {exc}")


def on_load():
    downloads.set_logger(lambda msg: lf.log.info(f"[hyworld2] {msg}"))
    _apply_inference_perf_flags()
    for cls in _classes:
        lf.register_class(cls)
    downloads.start_background_download()
    # Auto-unload pipeline when the trainer starts running. subscribe_as
    # auto-cleans this subscription on plugin unload.
    try:
        global _last_training_state
        from lfs_plugins.ui.state import AppState
        _last_training_state = bool(AppState.is_training.value)
        AppState.is_training.subscribe_as("hyworld2_plugin", _on_training_state_changed)
    except Exception as exc:
        lf.log.warn(f"hyworld2_plugin: couldn't subscribe to is_training ({exc}).")
    lf.log.info("hyworld2_plugin loaded (models caching to plugin dir)")


def on_unload():
    import gc
    import time

    # (1) If training is live, stop it so the trainer thread settles before
    # CUDA teardown. Otherwise a still-running training step can hit a
    # shutdown-time `cudaHostAlloc failed: illegal memory access` inside
    # LFS's PinnedMemoryAllocator.
    try:
        from lfs_plugins.ui.state import AppState
        if getattr(AppState, "is_training", None) is not None and AppState.is_training.value:
            lf.log.info("hyworld2_plugin: stopping training before unload.")
            try:
                lf.stop_training()
            except Exception:
                pass
            # Give the trainer up to 2s to actually stop.
            for _ in range(20):
                if not AppState.is_training.value:
                    break
                time.sleep(0.1)
    except Exception as exc:
        lf.log.warn(f"hyworld2_plugin: stop_training on unload failed: {exc}")

    # (2) Cancel background download (if still fetching).
    try:
        downloads.cancel_download()
        downloads.join(timeout=2.0)
    except Exception:
        pass

    # (3) Synchronously drop the cached pipeline + skyseg session, collect
    # any GPU tensors, sync + empty the CUDA cache. This gives LFS's C++
    # side a clean slate before it tears down its own CUDA context.
    try:
        pipeline_loader.unload()
    except Exception as exc:
        lf.log.warn(f"hyworld2_plugin: pipeline_loader.unload() failed: {exc}")
    for _ in range(2):  # two passes — first frees wrappers, second frees buffers
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    for cls in reversed(_classes):
        lf.unregister_class(cls)
    # Cached models intentionally persist across sessions. Removal is the
    # responsibility of uninstall.ps1 / uninstall.sh or the panel's
    # "Clear cached models" button.
    lf.log.info("hyworld2_plugin unloaded (cached models kept in plugin dir)")
