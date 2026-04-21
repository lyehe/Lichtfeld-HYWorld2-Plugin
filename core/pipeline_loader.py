"""Cached WorldMirrorPipeline singleton + plugin-local skyseg session.

Both live in ``<plugin>/models/`` via ``core.downloads`` — nothing ends up in
the user's home cache.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from . import downloads

_lock = threading.Lock()
_pipeline: Any = None
_skyseg_session: Any = None
_loaded_bf16: bool = False
_loaded_compile: bool = False
_loaded_compile_mode: str = ""
_compile_active: bool = False     # True iff torch.compile wrap took effect
_warmup_complete: bool = False
_warmup_time_s: float = 0.0
_warmup_failed: bool = False

# Persistent flag: set when torch.compile's Inductor backend fails during
# warmup with a CalledProcessError / cl.exe / InductorError. Future loads
# auto-disable compile until the user explicitly clears it. Users who
# install / fix MSVC can delete the file or call clear_compile_block().
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_COMPILE_BLOCKED_FILE = _PLUGIN_DIR / "cache" / "compile_blocked.flag"


def _compile_is_blocked() -> bool:
    return _COMPILE_BLOCKED_FILE.is_file()


def _mark_compile_blocked(reason: str) -> None:
    try:
        _COMPILE_BLOCKED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _COMPILE_BLOCKED_FILE.write_text(reason, encoding="utf-8")
    except OSError:
        pass  # Best-effort: flag is optional, runtime state already set.


def clear_compile_block() -> None:
    """Remove the persistent compile-blocked flag so the next load tries again."""
    try:
        _COMPILE_BLOCKED_FILE.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _looks_like_compile_failure(exc: BaseException) -> bool:
    """Heuristic match for the MSVC/Inductor/Triton compile error chain."""
    s = f"{type(exc).__name__}: {exc}"
    return (
        "InductorError" in s
        or "CalledProcessError" in s
        or "cl.exe" in s
        or "Failed to compile" in s
    )


def get_pipeline(model_id: str = "tencent/HY-World-2.0",
                 *, enable_bf16: bool = False,
                 enable_compile: bool = False,
                 compile_mode: str = "reduce-overhead",
                 **kwargs) -> Any:
    """Return a cached WorldMirrorPipeline, constructing on first call.

    If the cached pipeline was built with different ``enable_bf16`` /
    ``enable_compile`` / ``compile_mode`` than requested, it's dropped and
    rebuilt — those flags are baked into the model at construction time.
    ``HYWORLD_COMPILE`` + ``HYWORLD_COMPILE_MODE`` env vars are set here
    before ``from_pretrained`` reads them.
    """
    global _pipeline, _loaded_bf16, _loaded_compile, _loaded_compile_mode

    # Compile-path gate. If a prior warmup tripped the MSVC/Inductor
    # compile error (Windows + LFS GUI-host embedded Python), silently
    # fall back to eager. The flag is persistent — it's cleared only by
    # clear_compile_block() or by deleting the file. The call to
    # msvc_env.apply() is the *real* fix: it injects the VS Developer
    # Prompt env vars so cl.exe can find the Windows SDK headers. We
    # only drop to the fallback if that injection also fails to make
    # compile work.
    if enable_compile:
        try:
            from . import msvc_env
            applied, reason = msvc_env.apply()
            downloads._log_fn(f"MSVC env: {reason}")
            del applied
        except Exception as exc:
            downloads._log_fn(f"MSVC env setup skipped: {exc}")
    if enable_compile and _compile_is_blocked():
        downloads._log_fn(
            "torch.compile disabled — prior Inductor compile failed on this "
            "host. Delete <plugin>/cache/compile_blocked.flag (or call "
            "pipeline_loader.clear_compile_block()) to retry."
        )
        enable_compile = False

    with _lock:
        needs_reload = _pipeline is not None and (
            _loaded_bf16 != bool(enable_bf16)
            or _loaded_compile != bool(enable_compile)
            or (enable_compile and _loaded_compile_mode != compile_mode)
        )
    if needs_reload:
        unload()  # reacquires the lock; drops model + GC + CUDA sync

    # Env vars must be set BEFORE the pipeline constructor runs.
    import os
    os.environ["HYWORLD_COMPILE"] = "1" if enable_compile else "0"
    if enable_compile:
        os.environ["HYWORLD_COMPILE_MODE"] = compile_mode

    _compile_failure = ""  # Populated only when warmup trips the MSVC/Inductor path.
    with _lock:
        if _pipeline is None:
            # Measure the model's resident VRAM so the auto-fit heuristic
            # can subtract it from the budget.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_allocated()
                else:
                    baseline = 0
            except Exception:
                baseline = 0
            from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
            _pipeline = WorldMirrorPipeline.from_pretrained(
                model_id, enable_bf16=bool(enable_bf16), **kwargs,
            )
            _loaded_bf16 = bool(enable_bf16)
            _loaded_compile = bool(enable_compile)
            _loaded_compile_mode = compile_mode if enable_compile else ""
            try:
                import torch

                from . import vram_profile
                if torch.cuda.is_available():
                    after = torch.cuda.memory_allocated()
                    vram_profile.record_model_bytes(bool(enable_bf16), max(0, after - baseline))
            except Exception:
                pass
            # Warm up cuDNN kernel selection, PTX -> SASS JIT, and (if
            # enabled) torch.compile Inductor compile. A 2-frame 280x280
            # forward is cheap but exercises the same graph a real Run
            # will use. If torch.compile is on, the 30-60s compile cost
            # is paid here instead of on the user's first Run.
            global _compile_active, _warmup_complete, _warmup_time_s, _warmup_failed
            _compile_active = _detect_compile_active(_pipeline)
            _warmup_time_s, _warmup_failed, _compile_failure = _warmup_pipeline(
                _pipeline, enable_compile=bool(enable_compile),
            )
            _warmup_complete = True

    # If warmup tripped an Inductor/cl.exe failure, persist the flag
    # and rebuild the pipeline in eager mode so the caller gets a
    # usable model instead of one that'll crash on first forward.
    # Done outside the lock to allow unload()+reconstruct to reacquire.
    if enable_compile and _compile_failure:
        _mark_compile_blocked(_compile_failure)
        downloads._log_fn(
            f"torch.compile failed during warmup ({_compile_failure[:120]}); "
            "reloading in eager mode and disabling compile for future runs."
        )
        unload()
        return get_pipeline(
            model_id=model_id,
            enable_bf16=enable_bf16,
            enable_compile=False,
            compile_mode=compile_mode,
            **kwargs,
        )
    return _pipeline


def _warmup_pipeline(pipeline, enable_compile: bool = False) -> tuple[float, bool, str]:
    """Run a dummy forward to trigger JIT + (optional) torch.compile.

    Returns ``(elapsed_seconds, failed, compile_failure_reason)``.
    ``compile_failure_reason`` is a short diagnostic string when the
    failure is clearly MSVC/Inductor/Triton-compile-related (so the
    caller can flip the persistent compile-blocked flag), and ``""``
    otherwise.
    """
    import sys
    import time
    try:
        import torch
        if getattr(pipeline.device, "type", "") != "cuda":
            return (0.0, False, "")
        t0 = time.perf_counter()
        with torch.inference_mode():
            # 280 = _OOM_RETRY_MIN_TARGET_SIZE, divisible by 14 (patch_size).
            imgs = torch.zeros(1, 2, 3, 280, 280, device=pipeline.device)
            views = {"img": imgs}
            inner = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
            model_bf16 = getattr(inner, "enable_bf16", False)
            use_amp = torch.cuda.is_bf16_supported()
            with torch.amp.autocast(
                "cuda",
                enabled=(not model_bf16 and use_amp),
                dtype=torch.bfloat16,
            ):
                pipeline.model(views=views, cond_flags=[0, 0, 0], is_inference=True)
            torch.cuda.synchronize(pipeline.device)
        elapsed = time.perf_counter() - t0
        tag = " (torch.compile warmed)" if enable_compile else ""
        print(f"[hyworld2 warmup] forward: {elapsed:.2f}s{tag}", file=sys.stderr)
        return (elapsed, False, "")
    except Exception as exc:
        print(f"[hyworld2 warmup] failed (non-fatal): {type(exc).__name__}: {exc}",
              file=sys.stderr)
        reason = ""
        if enable_compile and _looks_like_compile_failure(exc):
            reason = f"{type(exc).__name__}: {exc}"
        return (0.0, True, reason)


def is_loaded() -> bool:
    with _lock:
        return _pipeline is not None


def loaded_bf16() -> bool:
    with _lock:
        return _loaded_bf16 if _pipeline is not None else False


def get_status() -> dict:
    """Snapshot of build/warmup state for UI display.

    Returns fields (all safe to read at any time):
      loaded:            bool  — pipeline constructed
      bf16:              bool  — current precision
      compile_requested: bool  — user asked for torch.compile
      compile_active:    bool  — torch.compile wrap actually applied
      compile_mode:      str   — "" if compile inactive
      warmup_complete:   bool  — warmup forward finished
      warmup_time_s:     float — how long it took (cuDNN + JIT + compile)
      warmup_failed:     bool  — warmup threw (non-fatal)
    """
    with _lock:
        loaded = _pipeline is not None
        return {
            "loaded": loaded,
            "bf16": _loaded_bf16 if loaded else False,
            "compile_requested": _loaded_compile if loaded else False,
            "compile_active": _compile_active if loaded else False,
            "compile_mode": _loaded_compile_mode if (loaded and _loaded_compile) else "",
            "warmup_complete": _warmup_complete if loaded else False,
            "warmup_time_s": _warmup_time_s if loaded else 0.0,
            "warmup_failed": _warmup_failed if loaded else False,
        }


def _detect_compile_active(pipeline) -> bool:
    """Best-effort check: is the cached pipeline's model torch.compile-wrapped?

    Checks for attributes that Dynamo adds to OptimizedModule wrappers.
    """
    try:
        model = getattr(pipeline, "model", None)
        if model is None:
            return False
        # Fast-path: name check first (no import cost if False)
        cls_name = type(model).__name__
        if cls_name in ("OptimizedModule", "_TorchDynamoContext"):
            return True
        # Dynamo wrappers expose `_orig_mod` / `_torchdynamo_orig_callable`.
        return hasattr(model, "_orig_mod") or hasattr(model, "_torchdynamo_orig_callable")
    except Exception:
        return False


def get_skyseg_session(progress_cb=None) -> Any:
    """Load the skyseg ONNX session from the plugin-local cache.

    Raises ``FileNotFoundError`` if the model hasn't been downloaded yet —
    callers should check ``downloads.is_skyseg_cached()`` first or disable
    sky masking.
    """
    global _skyseg_session
    with _lock:
        if _skyseg_session is not None:
            return _skyseg_session

    if not downloads.is_skyseg_cached():
        raise FileNotFoundError(
            f"skyseg.onnx not found at {downloads.SKYSEG_PATH}. "
            "Wait for the plugin to finish downloading models, or disable "
            "'Apply sky mask' in the settings."
        )
    if progress_cb:
        progress_cb(f"Loading skyseg session from {downloads.SKYSEG_PATH}")
    import onnxruntime
    session = onnxruntime.InferenceSession(str(downloads.SKYSEG_PATH))
    with _lock:
        _skyseg_session = session
    return session


def unload() -> None:
    """Drop the cached pipeline + skyseg session and free CUDA memory.

    Two gc passes + explicit sync ensures the WorldMirror model's torch
    tensors are actually released to the CUDA allocator before LFS or the
    Python interpreter tears down the CUDA context — which otherwise
    causes a shutdown-time 'illegal memory access' inside LFS.
    """
    global _pipeline, _skyseg_session, _loaded_bf16, _loaded_compile, _loaded_compile_mode
    global _compile_active, _warmup_complete, _warmup_time_s, _warmup_failed
    with _lock:
        _pipeline = None
        _skyseg_session = None
        _loaded_bf16 = False
        _loaded_compile = False
        _loaded_compile_mode = ""
        _compile_active = False
        _warmup_complete = False
        _warmup_time_s = 0.0
        _warmup_failed = False
    import gc
    for _ in range(2):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
