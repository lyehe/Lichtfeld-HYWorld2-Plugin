"""Threaded runner for a single HY-World-2 reconstruction job.

Wraps ``WorldMirrorPipeline.__call__`` with:
  * coarse stage tracking (preparing / inference / saving / post-processing)
  * a rolling log buffer for the panel
  * cancellation (best-effort; the inference forward pass itself is not
    interruptible, so cancel takes effect at stage boundaries)
  * a post-processing step that populates ``<outdir>/images/`` with frames
    resized and renamed to match the synthetic filenames
    ``hyworld2._save_colmap_lightweight`` writes into
    ``<outdir>/sparse/0/images.txt`` — making the output a self-contained
    dataset that ``lf.load_file(..., is_dataset=True)`` can import.
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import lichtfeld as lf


class JobStage(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    LOADING_MODEL = "loading_model"
    INFERENCE = "inference"
    POST_PROCESSING = "post_processing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


_RUNNING_STAGES = {
    JobStage.PREPARING,
    JobStage.LOADING_MODEL,
    JobStage.INFERENCE,
    JobStage.POST_PROCESSING,
}


@dataclass
class JobConfig:
    input_path: str
    output_dir: str

    # Output mode: "dataset" writes files + auto-loads as COLMAP; "direct"
    # pushes tensors straight into the active LFS scene (fastest, no files
    # written at all); "both" does direct output AND writes files.
    output_mode: str = "dataset"

    # Inference
    target_size: int = 952
    fps: int = 1
    video_min_frames: int = 1
    video_max_frames: int = 32

    # Save toggles (ignored in "direct" mode)
    save_gs: bool = True
    save_points: bool = True
    save_depth: bool = False
    save_normal: bool = False
    apply_sky_mask: bool = True

    # Precision
    enable_bf16: bool = False

    # Advanced perf (all three gated by env vars inside hyworld2):
    #   HYWORLD_COMPILE       - wrap model in torch.compile (30-60s first forward)
    #   HYWORLD_COMPILE_MODE  - 'default' | 'reduce-overhead' | 'max-autotune'
    #   HYWORLD_FP32_HEADS    - blanket fp32 autocast on heads (slower, more stable)
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    enable_fp32_heads: bool = False

    # Priors
    prior_cam_path: str = ""
    prior_depth_path: str = ""

    # Post-processing (dataset mode)
    copy_images_for_dataset: bool = True

    # VRAM hygiene
    auto_unload_model_after_run: bool = False
    # Adapts target_size down when too many frames would blow VRAM.
    # Empirical: ~1500 B/pixel/frame in bf16, ~3000 B/pixel/frame in fp32.
    auto_fit_target_size: bool = True

    # Gating
    require_cuda: bool = True


@dataclass
class JobResult:
    success: bool
    output_dir: str = ""
    elapsed_s: float = 0.0
    num_frames: int = 0
    gaussians_ply: str = ""
    points_ply: str = ""
    camera_params_json: str = ""
    images_dir: str = ""
    sparse_dir: str = ""
    scene_node_id: int = -1        # "direct" / "both" modes: root group id in scene
    splat_node_name: str = ""      # name of the per-run splat node in scene
    points_node_name: str = ""     # name of the per-run point cloud node in scene
    points_data: object = None     # (pts_np, cols_np) tuple kept for train-from-points
    error: str = ""


_PATCH_SIZE = 14
_MIN_TARGET_SIZE = _PATCH_SIZE * 2  # 28
# Empirical bytes per (frame × pixel) for WorldMirror-2 forward pass + gsplat
# rasterizer on Ada Lovelace. Doubled from the initial calibration after
# seeing real OOMs with the earlier numbers — peak during inference is
# substantially higher than the average activation footprint would suggest.
_BYTES_PER_PIXEL_FRAME_BF16 = 3200
_BYTES_PER_PIXEL_FRAME_FP32 = 6400


def _fit_target_size_to_vram(user_target: int, num_frames: int, bf16: bool,
                             log=None) -> int | None:
    """Return a patch-aligned target_size that should fit in free VRAM.

    Uses the self-calibrating profile from ``vram_profile``: bytes-per-
    pixel-per-frame is an EWMA over actual peak measurements from previous
    runs on this machine. Falls back to conservative defaults before the
    first run completes.

    Returns ``None`` on probing failure (caller keeps user_target).
    """
    try:
        import math

        import torch

        from . import vram_profile
        if not torch.cuda.is_available():
            return None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return None

    # Budget = free MINUS the resident model (always allocated) AND some
    # headroom for scene tensors / viewport / transient spikes.
    model_bytes = vram_profile.get_model_bytes(bf16)
    effective_free = max(0, free_bytes - model_bytes) if model_bytes else free_bytes
    headroom = 0.70 if model_bytes else 0.60  # tighter when we've accounted for model
    budget = int(effective_free * headroom)

    bpp = vram_profile.get_bpp(bf16) * vram_profile.safety_factor()
    if num_frames <= 0 or bpp <= 0:
        return None
    max_pixels_per_frame = budget / (num_frames * bpp)
    if max_pixels_per_frame <= 0:
        return _MIN_TARGET_SIZE
    max_edge = int(math.sqrt(max_pixels_per_frame))
    max_edge = (max_edge // _PATCH_SIZE) * _PATCH_SIZE
    if max_edge < _MIN_TARGET_SIZE:
        max_edge = _MIN_TARGET_SIZE
    if log is not None:
        log(f"vram-fit: free={free_bytes / 1e9:.1f}GB, "
            f"model={model_bytes / 1e9:.1f}GB, "
            f"budget={budget / 1e9:.1f}GB, frames={num_frames}, "
            f"bf16={bf16}, bpp={bpp:.0f}, max_edge={max_edge}")
    return min(int(user_target), max_edge)


class HyWorld2Job:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancelled = False

        self._stage = JobStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._result: JobResult | None = None
        self._log: deque[str] = deque(maxlen=48)

    # ------------------------------------------------------------------ props
    @property
    def stage(self) -> JobStage:
        with self._lock:
            return self._stage

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def result(self) -> JobResult | None:
        with self._lock:
            return self._result

    @property
    def log_text(self) -> str:
        with self._lock:
            return "\n".join(self._log)

    def is_running(self) -> bool:
        return self.stage in _RUNNING_STAGES

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Job already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._status = "Cancelling..."

    # ------------------------------------------------------------ internals
    def _set(self, stage: JobStage, progress: float, status: str) -> None:
        with self._lock:
            self._stage = stage
            self._progress = progress
            self._status = status

    def _log_line(self, msg: str) -> None:
        line = str(msg).rstrip()
        if not line:
            return
        with self._lock:
            self._log.append(line)
        lf.log.info(f"[hyworld2] {line}")

    def _log_error(self, msg: str) -> None:
        with self._lock:
            self._log.append(f"ERROR: {msg}")
        lf.log.error(f"[hyworld2] {msg}")

    def _is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def _check_cancel(self) -> None:
        if self._is_cancelled():
            raise _Cancelled()

    def _run(self) -> None:
        t0 = time.time()
        try:
            self._run_pipeline()
        except _Cancelled:
            self._set(JobStage.CANCELLED, self.progress, "Cancelled")
            with self._lock:
                self._result = JobResult(success=False, error="Cancelled",
                                         elapsed_s=time.time() - t0)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._log_error(msg)
            self._log_error(traceback.format_exc())
            self._set(JobStage.ERROR, self.progress, msg)
            with self._lock:
                self._result = JobResult(success=False, error=msg,
                                         elapsed_s=time.time() - t0)

    # ------------------------------------------------------------- core
    def _run_pipeline(self) -> None:
        cfg = self.cfg
        t_start = time.time()

        # -------- Preparing --------------------------------------------
        self._set(JobStage.PREPARING, 2.0, "Checking inputs")

        input_path = Path(cfg.input_path).expanduser()
        if not input_path.exists():
            raise RuntimeError(f"Input path does not exist: {input_path}")

        outdir = Path(cfg.output_dir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        if cfg.require_cuda:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. HY-World-Mirror-2 requires a CUDA GPU.")

        self._check_cancel()

        # Resolve the list of frames up-front so we can both:
        #   1. Hand the pipeline a stable input directory.
        #   2. Know the source paths for the post-processing image copy.
        try:
            from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import prepare_input
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import hyworld2 ({exc}). The plugin vendors this "
                "package inline at <plugin>/hyworld2/; if the import fails "
                "here it means LFS's plugin venv hasn't finished resolving "
                "dependencies. Check the LFS log for 'uv sync' errors."
            ) from exc

        self._log_line(f"Preparing input: {input_path}")
        img_paths, _subdir = prepare_input(
            str(input_path),
            target_size=cfg.target_size,
            fps=cfg.fps,
            video_strategy="new",
            min_frames=cfg.video_min_frames,
            max_frames=cfg.video_max_frames,
        )
        if not img_paths:
            raise RuntimeError(f"No frames extracted from {input_path}")
        self._log_line(f"Prepared {len(img_paths)} frame(s)")
        self._check_cancel()

        # Avoid re-encoding the user's original JPEGs. The pipeline will
        # internally call prepare_input(input_path=...) again — we want
        # that second call to see the same files without re-extracting
        # video. Solution: hand the pipeline the DIRECTORY containing the
        # already-resolved img_paths. For image folders that's the user's
        # folder (pipeline re-globs, hits cache). For video it's the
        # extract dir our prepare_input created (pipeline sees it as an
        # image folder, skips re-extraction).
        staged_dir = outdir / "_frames"
        # staged_paths hold REFERENCES only — they point at the original
        # files until the post-inference resave step creates _frames/
        # copies at inference resolution for direct-mode scene cameras.
        staged_paths: list[Path] = [Path(p) for p in img_paths]
        pipeline_input = str(Path(img_paths[0]).parent) if img_paths else str(input_path)
        self._log_line(f"Skip-staging: feeding pipeline {pipeline_input} ({len(img_paths)} frames)")
        self._check_cancel()

        # -------- Load model (cached) ----------------------------------
        from . import pipeline_loader
        if not pipeline_loader.is_loaded():
            self._set(JobStage.LOADING_MODEL, 10.0, "Loading WorldMirror model")
            self._log_line("Loading WorldMirror weights (first run may take a while)...")
        # HYWORLD_FP32_HEADS is read inside forward(), so it takes effect
        # on the next run without a pipeline reload.
        import os
        os.environ["HYWORLD_FP32_HEADS"] = "1" if cfg.enable_fp32_heads else "0"

        pipeline = pipeline_loader.get_pipeline(
            enable_bf16=cfg.enable_bf16,
            enable_compile=cfg.enable_compile,
            compile_mode=cfg.compile_mode,
        )
        self._check_cancel()

        # Reset peak counter so the post-inference measurement only
        # captures this job's forward pass. Also snapshot the allocated
        # baseline BEFORE the pipeline runs so the VRAM profile can
        # subtract accumulated scene state (splats from earlier runs,
        # viewport buffers, etc.) and isolate the true activation cost.
        pre_alloc_bytes = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                pre_alloc_bytes = int(torch.cuda.memory_allocated())
        except Exception:
            pass

        # -------- Auto-fit target_size to free VRAM --------------------
        effective_target = int(cfg.target_size)
        if cfg.auto_fit_target_size:
            fitted = _fit_target_size_to_vram(
                user_target=cfg.target_size,
                num_frames=len(staged_paths),
                bf16=cfg.enable_bf16,
                log=self._log_line,
            )
            if fitted and fitted < effective_target:
                self._log_line(
                    f"Auto-fit: lowering target_size {cfg.target_size} -> {fitted} "
                    f"for {len(staged_paths)} frames to fit VRAM."
                )
                effective_target = fitted

        # Preload skyseg ONNX session from the plugin's local cache so the
        # 168MB model doesn't download into LFS's working directory.
        sky_session = None
        if cfg.apply_sky_mask:
            try:
                sky_session = pipeline_loader.get_skyseg_session(self._log_line)
            except Exception as exc:
                self._log_line(f"Warning: skyseg session unavailable ({exc}); continuing without sky mask.")
                sky_session = None

        # -------- Inference --------------------------------------------
        status_target = (
            f"target_size={effective_target}"
            + (f" (auto-fit from {cfg.target_size})" if effective_target != int(cfg.target_size) else "")
        )
        self._set(JobStage.INFERENCE, 25.0, f"Running WorldMirror inference · {status_target}")
        self._log_line(f"Running inference at {status_target}...")

        prior_cam = cfg.prior_cam_path.strip() or None
        prior_depth = cfg.prior_depth_path.strip() or None
        if prior_cam:
            self._log_line(f"Using camera prior: {prior_cam}")
        if prior_depth:
            self._log_line(f"Using depth prior: {prior_depth}")

        # In "direct" mode we monkey-patch hyworld2's save_results to capture
        # predictions in memory and skip file I/O entirely. In "both" mode we
        # do both: capture predictions AND let the original save_results run.
        mode = cfg.output_mode
        capture_direct = mode in ("direct", "both")
        write_files = mode in ("dataset", "both")

        import hyworld2.worldrecon.pipeline as _hyp
        captured = {}
        orig_save = _hyp.save_results

        def _intercept(predictions, imgs, img_paths, outdir_arg, **kwargs):
            captured["predictions"] = predictions
            captured["imgs"] = imgs
            captured["img_paths"] = list(img_paths)
            # Pipeline also passes filter_mask / gs_filter_mask (computed from
            # sky segmentation + confidence) — capture these so direct mode can
            # apply the same filtering save_results does.
            captured["filter_mask"] = kwargs.get("filter_mask")
            captured["gs_filter_mask"] = kwargs.get("gs_filter_mask")
            if write_files:
                return orig_save(predictions, imgs, img_paths, outdir_arg, **kwargs)
            return {}

        if capture_direct:
            _hyp.save_results = _intercept
        try:
            pipeline(
                input_path=pipeline_input,
                output_path=str(outdir),
                strict_output_path=str(outdir),
                target_size=effective_target,
                fps=cfg.fps,
                video_min_frames=cfg.video_min_frames,
                video_max_frames=cfg.video_max_frames,
                # In pure direct mode, skip every save inside save_results;
                # in dataset/both modes, apply the user's file toggles.
                save_gs=cfg.save_gs if write_files else False,
                save_points=cfg.save_points if write_files else False,
                save_depth=cfg.save_depth if write_files else False,
                save_normal=cfg.save_normal if write_files else False,
                save_camera=write_files,
                save_colmap=write_files,
                apply_sky_mask=cfg.apply_sky_mask and sky_session is not None,
                sky_mask_session=sky_session,
                prior_cam_path=prior_cam,
                prior_depth_path=prior_depth,
                log_time=False,
            )
        finally:
            if capture_direct:
                _hyp.save_results = orig_save
        self._check_cancel()

        # -------- VRAM calibration: record peak usage ------------------
        try:
            import torch

            from . import vram_profile
            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
                model_bytes = vram_profile.get_model_bytes(cfg.enable_bf16)
                imgs_tensor = captured.get("imgs")
                if imgs_tensor is not None and imgs_tensor.dim() == 5:
                    _, S, _, H, W = imgs_tensor.shape
                    vram_profile.record_run(
                        bf16=cfg.enable_bf16,
                        frames=int(S), h=int(H), w=int(W),
                        peak_bytes=peak,
                        model_bytes=int(model_bytes),
                        pre_alloc_bytes=pre_alloc_bytes,
                        log=self._log_line,
                    )
        except Exception as exc:
            self._log_line(f"vram-profile update failed (non-fatal): {exc}")

        # -------- Post-process --------------------------------------------
        self._set(JobStage.POST_PROCESSING, 85.0, "Finalising outputs")
        images_dir_str = ""
        sparse_dir = outdir / "sparse" / "0"
        scene_node_id = -1

        # ORDER MATTERS: in "both" mode we first populate the dataset's
        # images/ dir from the ORIGINAL-resolution staged frames (so the
        # trainer sees full-quality supervision). Only after that do we
        # overwrite the staged frames at inference resolution for the
        # direct-mode scene cameras.
        if write_files and cfg.copy_images_for_dataset and sparse_dir.is_dir():
            images_dir_str = str(self._populate_images_dir(outdir, staged_paths))
            # hyworld2's _save_colmap_lightweight writes an empty points3D.txt
            # (comments only) which LFS's colmap loader rejects as corrupted
            # data. Fill it from points.ply so the dataset round-trips.
            self._populate_points3d(outdir)
        elif write_files and not sparse_dir.is_dir():
            self._log_line("No sparse/0/ directory was produced by the pipeline.")

        # Re-save staged frames at inference resolution for direct-mode
        # camera references. Runs AFTER images/ so we don't pollute the
        # dataset. Writes to outdir/_frames (not the user's input folder)
        # and redirects staged_paths list in-place.
        if capture_direct and captured.get("imgs") is not None and staged_paths:
            self._resave_staged_at_inference_resolution(
                captured["imgs"], staged_paths, staged_dir
            )

        # (A) Direct-to-scene: push tensors straight into the LFS scene.
        direct_info = None
        if capture_direct and captured.get("predictions") is not None:
            self._log_line("Applying predictions to LFS scene...")
            try:
                from . import direct_output
                # Staged frames live under outdir/_frames and are kept across
                # the run so cameras have valid image_path references.
                # Give LFS's trainer an output_path so checkpoint auto-save
                # works — even in direct mode the user has picked an
                # output_dir, use that as the checkpoint/snapshot target.
                # ``output_path`` is a read-only property; go via the
                # generic ``.set(name, value)`` hook which routes through
                # LFS's property registry (writes when .can_edit()).
                try:
                    params = lf.dataset_params()
                    current = getattr(params, "output_path", "") or ""
                    if not current or not Path(current).is_dir():
                        outdir.mkdir(parents=True, exist_ok=True)
                        if hasattr(params, "set"):
                            params.set("output_path", str(outdir))
                        self._log_line(f"Training output_path set to {outdir}")
                except Exception as exc:
                    self._log_line(f"Could not set dataset_params.output_path: {exc}")

                direct_info = direct_output.apply_predictions_to_scene(
                    captured["predictions"],
                    captured["img_paths"],
                    imgs=captured.get("imgs"),
                    filter_mask=captured.get("filter_mask"),
                    gs_filter_mask=captured.get("gs_filter_mask"),
                    node_name=f"HY-World-Mirror-2 ({outdir.name})",
                    attach_images=True,
                    add_point_cloud=True,
                    log=self._log_line,
                )
                if direct_info:
                    scene_node_id = int(direct_info.get("parent_id") or -1)
                    # Auto-select the run's splat node for training so
                    # "Start training" picks up our output without extra clicks.
                    splat = direct_info.get("splat_node_name")
                    if splat:
                        try:
                            direct_output.set_training_node(splat, log=self._log_line)
                        except Exception as exc:
                            self._log_line(f"set_training_node: {exc}")
            except Exception as exc:
                self._log_line(f"Direct output failed: {exc}")

        # Clean up staging dir in pure "dataset" mode (no direct-output
        # cameras reference it). For direct/both modes, _frames/ was only
        # created (inside _resave) if direct output ran; scene cameras
        # reference it, so leave it.
        if (
            mode == "dataset"
            and write_files
            and images_dir_str
            and cfg.copy_images_for_dataset
            and staged_dir.exists()
        ):
            shutil.rmtree(staged_dir, ignore_errors=True)

        # -------- VRAM hygiene -------------------------------------------
        # Splats pushed into the scene via DLPack keep their GPU buffers
        # alive independently (scene's lf.Tensor holds its own refcount
        # on the shared storage). Everything else — predictions for depth
        # / normals / raw camera params, and the inference imgs tensor
        # — is safe to drop, which lets torch return a substantial chunk
        # of activation memory to the CUDA allocator.
        try:
            captured.clear()
        except Exception:
            pass
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        if cfg.auto_unload_model_after_run:
            self._log_line("Auto-unloading WorldMirror model (free VRAM).")
            try:
                pipeline_loader.unload()
            except Exception as exc:
                self._log_line(f"pipeline_loader.unload() failed: {exc}")

        # -------- Done --------------------------------------------------
        elapsed = time.time() - t_start
        splat_name = direct_info.get("splat_node_name", "") if direct_info else ""
        points_name = direct_info.get("points_node_name", "") if direct_info else ""
        points_data = None
        if direct_info:
            pn = direct_info.get("points_np")
            cn = direct_info.get("colors_np")
            if pn is not None and cn is not None:
                points_data = (pn, cn)
        result = JobResult(
            success=True,
            output_dir=str(outdir),
            elapsed_s=elapsed,
            num_frames=len(staged_paths),
            gaussians_ply=str(outdir / "gaussians.ply") if (outdir / "gaussians.ply").is_file() else "",
            points_ply=str(outdir / "points.ply") if (outdir / "points.ply").is_file() else "",
            camera_params_json=str(outdir / "camera_params.json") if (outdir / "camera_params.json").is_file() else "",
            images_dir=images_dir_str,
            sparse_dir=str(sparse_dir) if sparse_dir.is_dir() else "",
            scene_node_id=scene_node_id,
            splat_node_name=splat_name,
            points_node_name=points_name,
            points_data=points_data,
        )
        with self._lock:
            self._result = result
        self._set(JobStage.DONE, 100.0, "Complete")
        self._log_line(f"Done in {elapsed:.1f}s")

    # -------- helpers -------------------------------------------------
    def _populate_images_dir(self, outdir: Path, staged_paths: list[Path]) -> Path:
        """Resize staged frames into ``outdir/images/`` to match cameras.txt.

        ``hyworld2._save_colmap_lightweight`` writes ``cameras.txt`` with a
        single (W, H) derived from ``save_results``'s ``new_w, new_h`` — we
        resize each frame to that size so the dataset dimensions match the
        intrinsics.
        """
        from PIL import Image
        sparse_dir = outdir / "sparse" / "0"
        cameras_txt = sparse_dir / "cameras.txt"
        if not cameras_txt.is_file():
            raise RuntimeError(f"Expected {cameras_txt} after pipeline run")

        target_w, target_h = _read_first_camera_size(cameras_txt)
        images_dir = outdir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(staged_paths):
            dst = images_dir / f"image_{i + 1:04d}.jpg"
            with Image.open(src) as im:
                im = im.convert("RGB")
                if im.size != (target_w, target_h):
                    im = im.resize((target_w, target_h), Image.LANCZOS)
                im.save(dst, "JPEG", quality=95)

        self._log_line(f"Wrote {len(staged_paths)} images -> {images_dir} ({target_w}x{target_h})")
        return images_dir

    def _resave_staged_at_inference_resolution(self, imgs, staged_paths,
                                                 staged_dir: Path) -> None:
        """Write the inference tensor to fresh JPGs under ``staged_dir``.

        ``imgs`` is [B, S, C, H, W] float in [0, 1]. We create
        ``staged_dir`` fresh and replace ``staged_paths`` (list) in place
        with paths into that dir. Original input files are never touched.
        """
        from PIL import Image
        try:
            tensor = imgs.detach().cpu().float().clamp(0.0, 1.0)
        except Exception as exc:
            self._log_line(f"resave staged: couldn't move imgs to cpu ({exc}); skipping.")
            return
        if tensor.dim() != 5:
            return
        _, S, _, H, W = tensor.shape
        n = min(S, len(staged_paths))

        # Fresh dir — remove any stale content from a previous run.
        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)
        staged_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            arr = (tensor[0, i].permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")
            dst = staged_dir / f"image_{i + 1:04d}.jpg"
            try:
                Image.fromarray(arr).save(str(dst), "JPEG", quality=95)
            except Exception as exc:
                self._log_line(f"resave staged[{i}] failed: {exc}")
                return
            staged_paths[i] = dst  # redirect list to the new file
        self._log_line(f"Wrote {n} inference-res frames to {staged_dir} ({W}x{H}).")

    def _populate_points3d(self, outdir: Path, max_points: int = 500_000) -> None:
        """Rewrite ``sparse/0/points3D.txt`` with real points from ``points.ply``.

        hyworld2 writes only header comments into points3D.txt, which LFS's
        COLMAP loader rejects as "File is empty". We read the voxel-pruned
        points.ply that the pipeline already emits and serialise up to
        ``max_points`` entries so LFS can ingest the dataset directly.
        """
        points_ply = outdir / "points.ply"
        target = outdir / "sparse" / "0" / "points3D.txt"
        if not points_ply.is_file():
            self._log_line(f"No points.ply - leaving {target.name} empty (LFS may reject).")
            return
        try:
            from plyfile import PlyData
        except ImportError as exc:
            self._log_line(f"plyfile not available ({exc}) - can't populate points3D.txt.")
            return

        data = PlyData.read(str(points_ply))
        if "vertex" not in [e.name for e in data.elements]:
            self._log_line("points.ply has no 'vertex' element.")
            return
        v = data["vertex"]
        total = len(v)
        if total == 0:
            return

        # Subsample deterministically to keep points3D.txt under ~40 MB.
        if total > max_points:
            import numpy as np
            idx = np.linspace(0, total - 1, max_points, dtype=np.int64)
            xs, ys, zs = v["x"][idx], v["y"][idx], v["z"][idx]
            rs, gs_, bs = v["red"][idx], v["green"][idx], v["blue"][idx]
            n = max_points
        else:
            xs, ys, zs = v["x"], v["y"], v["z"]
            rs, gs_, bs = v["red"], v["green"], v["blue"]
            n = total

        # Stream-write to avoid buffering ~40 MB in one big string.
        with open(target, "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {n}, mean track length: 0\n")
            for i in range(n):
                f.write(
                    f"{i + 1} {float(xs[i]):.6f} {float(ys[i]):.6f} {float(zs[i]):.6f} "
                    f"{int(rs[i])} {int(gs_[i])} {int(bs[i])} 1.0\n"
                )
        self._log_line(f"Wrote {n} points -> {target.name} (from {total} in points.ply)")


class _Cancelled(Exception):
    pass


def _read_first_camera_size(cameras_txt: Path) -> tuple[int, int]:
    for raw in cameras_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[...]
        return int(parts[2]), int(parts[3])
    raise RuntimeError(f"No camera entry found in {cameras_txt}")
