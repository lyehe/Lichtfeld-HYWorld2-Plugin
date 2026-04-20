"""Plugin-local model downloads + progress state.

All model assets live under ``<plugin>/models/`` so the plugin owns its
storage end-to-end. Nothing leaks into ``~/.cache/``.

The module sets ``HF_HOME`` / ``HF_HUB_CACHE`` BEFORE any ``huggingface_hub``
import (done in ``__init__.py``) so ``snapshot_download`` — called both
here and inside ``hyworld2.worldrecon.pipeline._resolve_model_dir`` — lands
weights in the plugin dir automatically.
"""
from __future__ import annotations

import shutil
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PLUGIN_DIR / "models"
HF_CACHE_DIR = MODELS_DIR / "huggingface"
HF_HUB_DIR = HF_CACHE_DIR / "hub"
SKYSEG_PATH = MODELS_DIR / "skyseg" / "skyseg.onnx"

WORLDMIRROR_REPO = "tencent/HY-World-2.0"
WORLDMIRROR_SUBFOLDER = "HY-WorldMirror-2.0"
WORLDMIRROR_APPROX_BYTES = 4_800_000_000  # ~4.8 GB, for progress estimation
SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
SKYSEG_APPROX_BYTES = 168_000_000

# ----- Module-level state (thread-safe) ----------------------------------

_lock = threading.Lock()
_state = {
    "stage": "idle",        # idle | checking | downloading_weights | downloading_skyseg | ready | error
    "progress": 0.0,        # 0.0 .. 1.0 overall
    "message": "",
    "error": "",
    "cancelled": False,
    "bytes_downloaded": 0,
    "bytes_total": 0,
}
_thread: threading.Thread | None = None


def _noop_log(_msg: str) -> None:
    return None


_log_fn: Callable[[str], None] = _noop_log


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _is_cancelled() -> bool:
    with _lock:
        return _state["cancelled"]


def set_logger(fn: Callable[[str], None]) -> None:
    global _log_fn
    _log_fn = fn


# ----- Public API --------------------------------------------------------

def worldmirror_local_dir() -> Path:
    """Directory where the WorldMirror snapshot lands after download."""
    # snapshot_download returns <hub>/models--tencent--HY-World-2.0/snapshots/<hash>/
    return HF_HUB_DIR / f"models--{WORLDMIRROR_REPO.replace('/', '--')}"


def is_weights_cached() -> bool:
    """Best-effort check that WorldMirror's safetensors + config are on disk."""
    snap_root = worldmirror_local_dir() / "snapshots"
    if not snap_root.is_dir():
        return False
    for snap in snap_root.iterdir():
        sub = snap / WORLDMIRROR_SUBFOLDER
        if (sub / "model.safetensors").is_file() and (
            (sub / "config.json").is_file() or (sub / "config.yaml").is_file()
        ):
            return True
    return False


def is_skyseg_cached() -> bool:
    return SKYSEG_PATH.is_file() and SKYSEG_PATH.stat().st_size > 1_000_000


def is_ready() -> bool:
    return get_state()["stage"] == "ready"


def start_background_download() -> None:
    """Kick off the download thread (no-op if already running or complete)."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        if _state["stage"] == "ready":
            return
        _state.update(
            stage="checking",
            progress=0.0,
            message="Checking model cache...",
            error="",
            cancelled=False,
            bytes_downloaded=0,
            bytes_total=0,
        )
    _thread = threading.Thread(target=_run, name="hyworld2-model-dl", daemon=True)
    _thread.start()


def cancel_download() -> None:
    _set(cancelled=True)


def join(timeout: float = 2.0) -> None:
    t = _thread
    if t and t.is_alive():
        t.join(timeout=timeout)


def delete_models() -> None:
    """Remove every byte the plugin has cached. Idempotent, non-fatal on locks."""
    cancel_download()
    join(timeout=3.0)
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
    _set(stage="idle", progress=0.0, message="", error="", bytes_downloaded=0, bytes_total=0)


# ----- Worker ------------------------------------------------------------

def _run() -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        HF_HUB_DIR.mkdir(parents=True, exist_ok=True)
        SKYSEG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not is_weights_cached():
            _set(stage="downloading_weights",
                 message=f"Downloading WorldMirror weights (~{WORLDMIRROR_APPROX_BYTES // 1_000_000_000} GB)...",
                 progress=0.01,
                 bytes_total=WORLDMIRROR_APPROX_BYTES + SKYSEG_APPROX_BYTES)
            _log_fn("Downloading WorldMirror weights to plugin-local cache...")
            _download_worldmirror()
            if _is_cancelled():
                _set(stage="error", error="Cancelled", message="Download cancelled")
                return

        if not is_skyseg_cached():
            _set(stage="downloading_skyseg",
                 message="Downloading skyseg.onnx (~168 MB)...")
            _log_fn("Downloading skyseg.onnx to plugin-local cache...")
            _download_skyseg()
            if _is_cancelled():
                _set(stage="error", error="Cancelled", message="Download cancelled")
                return

        _set(stage="ready", progress=1.0, message="Models ready")
        _log_fn("All models cached in plugin dir.")

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _log_fn(f"Model download failed: {msg}")
        _set(stage="error", error=msg, message="Download failed")


def _download_worldmirror() -> None:
    # Parallel progress-tracker thread that watches bytes on disk.
    stop_flag = threading.Event()

    def watch():
        target = WORLDMIRROR_APPROX_BYTES
        total_budget = target + SKYSEG_APPROX_BYTES
        while not stop_flag.is_set():
            try:
                total = sum(
                    f.stat().st_size for f in HF_HUB_DIR.rglob("*") if f.is_file()
                )
            except OSError:
                total = 0
            frac = min(0.98, total / target) if target > 0 else 0.0
            # Reserve last 5% for the skyseg download.
            overall = frac * (target / total_budget)
            _set(progress=overall,
                 bytes_downloaded=total,
                 bytes_total=total_budget)
            if _is_cancelled():
                return
            time.sleep(0.5)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=WORLDMIRROR_REPO,
            allow_patterns=[f"{WORLDMIRROR_SUBFOLDER}/*"],
            cache_dir=str(HF_HUB_DIR),
        )
    finally:
        stop_flag.set()
        watcher.join(timeout=1.0)


def _download_skyseg() -> None:
    tmp = SKYSEG_PATH.with_suffix(".onnx.part")
    budget = WORLDMIRROR_APPROX_BYTES + SKYSEG_APPROX_BYTES

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if _is_cancelled():
            raise _Cancelled()
        dl = block_num * block_size
        if total_size > 0:
            sky_frac = min(1.0, dl / total_size)
            base = WORLDMIRROR_APPROX_BYTES / budget
            _set(progress=base + (1.0 - base) * sky_frac,
                 bytes_downloaded=WORLDMIRROR_APPROX_BYTES + dl,
                 bytes_total=budget)

    try:
        urllib.request.urlretrieve(SKYSEG_URL, str(tmp), hook)
    except _Cancelled:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return
    tmp.replace(SKYSEG_PATH)


class _Cancelled(Exception):
    pass
