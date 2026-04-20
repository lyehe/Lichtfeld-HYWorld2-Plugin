"""HY-World-Mirror-2 reconstruction panel."""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import lichtfeld as lf

from ..core import colmap_io, direct_output, downloads, pipeline_loader, vram_profile
from ..core.job import HyWorld2Job, JobConfig, JobResult

_INPUT_TYPES = ["images", "video", "colmap"]
_DEFAULT_OUTPUT_ROOT = str(Path.home() / "hyworld2_out")

# Named dirty-field groups — call-sites get a label instead of a long
# string list. Order within a group doesn't matter; `handle.dirty(name)`
# is idempotent per name.
_DIRTY_MODEL_UI = (
    "models_ready", "models_downloading", "models_error",
    "model_stage_text", "model_progress_value", "model_progress_pct",
    "model_error_text", "model_bytes_line", "can_run",
    "model_status_line", "build_status_line",
    "show_download_btn", "show_load_btn", "show_unload_btn", "model_loading",
)
_DIRTY_RESULT = (
    "show_results", "show_error", "error_text",
    "result_output", "result_frames", "result_time",
    "result_has_gs", "result_has_points",
    "splats_already_loaded", "points_already_loaded",
    "has_scene_splats", "has_scene_points", "training_node_text",
    "can_run",
)
_DIRTY_RUN = (
    "stage_text", "progress_value", "progress_pct", "progress_status",
)
_DIRTY_RUNNING = ("show_idle", "show_running", "can_run")
_DIRTY_LOG = ("show_logs", "live_log_text")


def _supports_bf16() -> bool:
    """Return True if the host GPU has native bf16 support (compute >= 8.0)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


class HYWorld2Panel(lf.ui.Panel):
    id = "hyworld2_plugin.main"
    label = "HY-World-Mirror-2"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 150

    # ------------------------------------------------------------------
    def __init__(self):
        self._doc = None
        self._handle = None

        # Inputs
        self.input_path = ""
        self.input_type = "images"
        self.prior_cam_path = ""
        self.prior_depth_path = ""

        # Output
        self.output_dir = self._default_output_dir()

        # Parameters
        self.target_size = 952
        self.fps = 1
        self.video_min_frames = 1
        self.video_max_frames = 32

        # Output mode:
        #   "direct"  - fastest, tensors -> scene + in-memory trainer.
        #               No files, no disk I/O. Training works via
        #               lf.prepare_training_from_scene().
        #   "dataset" - writes COLMAP files + calls lf.load_file(is_dataset=True,
        #               init_path=gaussians.ply). Use when you want the
        #               dataset persisted to disk for re-training later.
        #   "both"    - direct + dataset.
        self.output_mode = "direct"

        self.save_gs = True
        self.save_points = True
        self.save_depth = False
        self.save_normal = False
        self.apply_sky_mask = True
        self.auto_load_dataset = True
        # bfloat16 inference — ~half VRAM, marginal quality hit. Requires
        # compute capability >= 8.0 (Ampere or newer; RTX 30xx/40xx/50xx).
        # Auto-enabled when the host GPU supports it.
        self.enable_bf16 = _supports_bf16()
        # Release ~2.4 GB bf16 / ~4.7 GB fp32 of WorldMirror weights back
        # to the CUDA allocator after each Run. Costs ~9s to reload on
        # the next run. Leave off unless VRAM is tight.
        self.auto_unload_model_after_run = False
        # Automatically lower target_size for high frame counts so the
        # forward pass doesn't OOM. Respects the user's target_size as
        # the upper bound.
        self.auto_fit_target_size = True
        # Advanced perf (env-var gated inside hyworld2 — changes affect the
        # NEXT run; compile changes force a pipeline reload).
        self.enable_compile = False
        self.compile_mode = "reduce-overhead"
        self.enable_fp32_heads = False

        # Job
        self._job: HyWorld2Job | None = None
        self._last_result: JobResult | None = None
        self._loaded_result_key: object = None

        # Post-job secondary-load state (shown after success)
        self._splats_loaded = False
        self._points_loaded = False

        # Model load state: "idle" | "loading" | "ready" | "error"
        self._model_load_state = "idle"
        self._model_load_error = ""
        self._model_load_thread: threading.Thread | None = None

        # UI diff state for on_update
        self._last_stage = ""
        self._last_progress = -1.0
        self._last_status = ""
        self._last_log_text = ""
        self._last_running = False
        self._last_result_key = None
        self._last_model_state = ("", -1.0, "")

        self._collapsed = {"advanced"}

    # ------------------------------------------------------------------
    # Helpers / state
    @staticmethod
    def _default_output_dir() -> str:
        base = Path(_DEFAULT_OUTPUT_ROOT)
        return str(base / time.strftime("run_%Y%m%d_%H%M%S"))

    def _dirty(self, *fields: str) -> None:
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for name in fields:
            self._handle.dirty(name)

    def _is_running(self) -> bool:
        return self._job is not None and self._job.is_running()

    def _has_output(self) -> bool:
        return bool(self.output_dir.strip())

    def _has_input(self) -> bool:
        return bool(self.input_path.strip())

    def _can_run(self) -> bool:
        return (
            not self._is_running()
            and self._has_input()
            and self._has_output()
            and downloads.is_ready()
        )

    # ------ Model load state helpers ----------------------------------
    def _refresh_model_load_state(self) -> str:
        """Sync our own state cache from pipeline_loader (which is the truth)."""
        try:
            if pipeline_loader.is_loaded():
                return "ready"
        except Exception:
            pass
        return self._model_load_state if self._model_load_state == "loading" else "idle"

    def _model_loading(self) -> bool:
        return self._refresh_model_load_state() == "loading"

    def _show_download_btn(self) -> bool:
        # Show when we have no cached weights (or an error), and not currently fetching.
        s = downloads.get_state()
        stage = s["stage"]
        if stage in ("ready",):
            return not downloads.is_weights_cached()
        if stage in ("downloading_weights", "downloading_skyseg", "checking"):
            return False
        # "idle" or "error" → expose the action.
        return True

    def _show_load_btn(self) -> bool:
        # Only meaningful once weights are on disk, and model isn't loaded/loading.
        if not downloads.is_ready():
            return False
        state = self._refresh_model_load_state()
        return state not in ("ready", "loading")

    def _show_unload_btn(self) -> bool:
        return self._refresh_model_load_state() == "ready"

    def _model_status_line(self) -> str:
        # Models-on-disk part
        s = downloads.get_state()
        stage = s["stage"]
        if stage in ("downloading_weights", "downloading_skyseg", "checking"):
            models_part = f"Models: {int(s['progress'] * 100)}% downloading"
        elif stage == "error":
            models_part = "Models: error"
        elif downloads.is_weights_cached() and downloads.is_skyseg_cached():
            models_part = "Models: downloaded"
        elif downloads.is_weights_cached():
            models_part = "Models: partial (skyseg missing)"
        else:
            models_part = "Models: not downloaded"
        # GPU part
        state = self._refresh_model_load_state()
        if state == "ready":
            gpu_part = "Model: ready in VRAM"
        elif state == "loading":
            gpu_part = "Model: loading..."
        elif state == "error":
            gpu_part = f"Model: error ({self._model_load_error or 'unknown'})"
        else:
            gpu_part = "Model: not loaded"
        return f"{models_part} · {gpu_part}"

    def _build_status_line(self) -> str:
        """Shows current precision / compile / warmup state of the loaded model."""
        st = pipeline_loader.get_status()
        if not st["loaded"]:
            return "Build: (model not loaded)"
        parts: list[str] = []
        parts.append("bf16" if st["bf16"] else "fp32")
        if st["compile_requested"]:
            if st["compile_active"]:
                parts.append(f"torch.compile ON ({st['compile_mode']})")
            else:
                parts.append("torch.compile requested but inactive")
        if st["warmup_complete"]:
            if st["warmup_failed"]:
                parts.append("warmup failed (continuing)")
            else:
                parts.append(f"warmed in {st['warmup_time_s']:.1f}s")
        else:
            parts.append("warming up…")
        return "Build: " + " · ".join(parts)

    def _models_ready(self) -> bool:
        return downloads.is_ready()

    def _models_downloading(self) -> bool:
        stage = downloads.get_state()["stage"]
        return stage in ("checking", "downloading_weights", "downloading_skyseg")

    def _models_error(self) -> bool:
        return downloads.get_state()["stage"] == "error"

    def _model_stage_text(self) -> str:
        s = downloads.get_state()
        return str(s.get("message") or s.get("stage", ""))

    def _model_progress_value(self) -> str:
        p = downloads.get_state()["progress"]
        return f"{max(0.0, min(1.0, p)):.4f}"

    def _model_progress_pct(self) -> str:
        p = downloads.get_state()["progress"]
        return f"{int(p * 100)}%"

    def _model_error_text(self) -> str:
        return downloads.get_state()["error"] or ""

    def _model_bytes_line(self) -> str:
        s = downloads.get_state()
        dl = s.get("bytes_downloaded", 0)
        total = s.get("bytes_total", 0)
        if not total:
            return ""
        return f"{dl / 1_000_000_000:.2f} / {total / 1_000_000_000:.2f} GB"

    @staticmethod
    def _result_key(r: JobResult | None):
        if r is None:
            return None
        return (r.success, r.output_dir, r.elapsed_s, r.error, r.num_frames)

    # ------------------------------------------------------------------
    # Lifecycle
    def on_mount(self, doc):
        self._doc = doc
        self._sync_section_states()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("hyworld2")
        if model is None:
            return

        # Two-way bindings
        model.bind("input_path", lambda: self.input_path, self._set_input_path)
        model.bind("input_type", lambda: self.input_type, self._set_input_type)
        model.bind("prior_cam_path", lambda: self.prior_cam_path, self._set_prior_cam_path)
        model.bind("prior_depth_path", lambda: self.prior_depth_path, self._set_prior_depth_path)
        model.bind("output_dir", lambda: self.output_dir, self._set_output_dir)
        model.bind("output_mode", lambda: self.output_mode, self._set_output_mode)

        model.bind("target_size", lambda: str(self.target_size),
                   lambda v: self._set_int("target_size", v, 280, 1680))
        model.bind("fps", lambda: str(self.fps),
                   lambda v: self._set_int("fps", v, 1, 30))
        model.bind("video_min_frames", lambda: str(self.video_min_frames),
                   lambda v: self._set_int("video_min_frames", v, 1, 64))
        model.bind("video_max_frames", lambda: str(self.video_max_frames),
                   lambda v: self._set_int("video_max_frames", v, 1, 64))

        model.bind("save_gs", lambda: self.save_gs, lambda v: self._set_bool("save_gs", v))
        model.bind("save_points", lambda: self.save_points, lambda v: self._set_bool("save_points", v))
        model.bind("save_depth", lambda: self.save_depth, lambda v: self._set_bool("save_depth", v))
        model.bind("save_normal", lambda: self.save_normal, lambda v: self._set_bool("save_normal", v))
        model.bind("apply_sky_mask", lambda: self.apply_sky_mask,
                   lambda v: self._set_bool("apply_sky_mask", v))
        model.bind("auto_load_dataset", lambda: self.auto_load_dataset,
                   lambda v: self._set_bool("auto_load_dataset", v))
        model.bind("enable_bf16", lambda: self.enable_bf16,
                   lambda v: self._set_bf16(v))
        model.bind("auto_unload_model_after_run",
                   lambda: self.auto_unload_model_after_run,
                   lambda v: self._set_bool("auto_unload_model_after_run", v))
        model.bind("auto_fit_target_size",
                   lambda: self.auto_fit_target_size,
                   lambda v: self._set_bool("auto_fit_target_size", v))
        model.bind("enable_compile",
                   lambda: self.enable_compile,
                   lambda v: self._set_bool("enable_compile", v))
        model.bind("compile_mode",
                   lambda: self.compile_mode,
                   self._set_compile_mode)
        model.bind("enable_fp32_heads",
                   lambda: self.enable_fp32_heads,
                   lambda v: self._set_bool("enable_fp32_heads", v))

        # Read-only computed
        model.bind_func("has_input", self._has_input)
        model.bind_func("has_output", self._has_output)
        model.bind_func("input_summary", self._input_summary)
        model.bind_func("output_summary", lambda: self.output_dir or "(no folder)")

        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("can_run", self._can_run)

        model.bind_func("stage_text", self._stage_text)
        model.bind_func("progress_value", self._progress_value)
        model.bind_func("progress_pct", self._progress_pct)
        model.bind_func("progress_status", self._progress_status)

        model.bind_func("show_logs", self._show_logs)
        model.bind_func("live_log_text", self._live_log_text)

        model.bind_func("show_results", self._show_results)
        model.bind_func("show_error", self._show_error)
        model.bind_func("error_text", self._error_text)
        model.bind_func("result_output", self._result_output)
        model.bind_func("result_frames", self._result_frames)
        model.bind_func("result_time", self._result_time)
        model.bind_func("result_has_gs", self._result_has_gs)
        model.bind_func("result_has_points", self._result_has_points)
        model.bind_func("splats_already_loaded", lambda: self._splats_loaded)
        model.bind_func("points_already_loaded", lambda: self._points_loaded)
        model.bind_func("has_scene_splats", self._result_has_scene_splats)
        model.bind_func("has_scene_points", self._result_has_scene_points)
        model.bind_func("training_node_text", self._training_node_text)

        model.bind_func("model_status_line", self._model_status_line)
        model.bind_func("build_status_line", self._build_status_line)
        model.bind_func("show_download_btn", self._show_download_btn)
        model.bind_func("show_load_btn", self._show_load_btn)
        model.bind_func("show_unload_btn", self._show_unload_btn)
        model.bind_func("model_loading", self._model_loading)

        model.bind_func("show_use_scene_btn", self._can_use_scene)
        model.bind_func("is_video_input", lambda: self.input_type == "video")
        model.bind_func("is_image_input", lambda: self.input_type == "images")
        model.bind_func("is_colmap_input", lambda: self.input_type == "colmap")
        model.bind_func("browse_input_label", self._browse_input_label)

        # Model download state
        model.bind_func("models_ready", self._models_ready)
        model.bind_func("models_downloading", self._models_downloading)
        model.bind_func("models_error", self._models_error)
        model.bind_func("model_stage_text", self._model_stage_text)
        model.bind_func("model_progress_value", self._model_progress_value)
        model.bind_func("model_progress_pct", self._model_progress_pct)
        model.bind_func("model_error_text", self._model_error_text)
        model.bind_func("model_bytes_line", self._model_bytes_line)

        # Events
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("retry_download", self._on_retry_download)
        model.bind_event("clear_models", self._on_clear_models)
        model.bind_event("browse_input", self._on_browse_input)
        model.bind_event("browse_input_folder", self._on_browse_input_folder)
        model.bind_event("browse_input_video", self._on_browse_input_video)
        model.bind_event("browse_colmap", self._on_browse_colmap)
        model.bind_event("use_current_scene", self._on_use_current_scene)
        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("browse_prior_cam", self._on_browse_prior_cam)
        model.bind_event("browse_prior_depth", self._on_browse_prior_depth)
        model.bind_event("clear_prior_cam", lambda _h, _e, _a: self._set_prior_cam_path(""))
        model.bind_event("clear_prior_depth", lambda _h, _e, _a: self._set_prior_depth_path(""))
        model.bind_event("do_start", self._on_start)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("load_gaussians_ply", self._on_load_gaussians)
        model.bind_event("load_points_ply", self._on_load_points)
        model.bind_event("unload_model", self._on_unload_model)
        model.bind_event("train_from_splats", self._on_train_from_splats)
        model.bind_event("train_from_points", self._on_train_from_points)
        model.bind_event("download_models", self._on_download_models)
        model.bind_event("load_model", self._on_load_model)
        model.bind_event("reset_vram_profile", self._on_reset_vram_profile)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        dirty = False

        model_state = (
            downloads.get_state()["stage"],
            round(downloads.get_state()["progress"], 3),
            downloads.get_state()["message"],
            self._refresh_model_load_state(),
        )
        if model_state != self._last_model_state:
            self._last_model_state = model_state
            self._dirty(*_DIRTY_MODEL_UI)
            dirty = True

        job = self._job
        if job:
            stage = job.stage.value
            progress = job.progress
            status = job.status
            if (stage != self._last_stage or progress != self._last_progress
                    or status != self._last_status):
                self._last_stage = stage
                self._last_progress = progress
                self._last_status = status
                self._dirty(*_DIRTY_RUN)
                dirty = True

            log_text = job.log_text
            if log_text != self._last_log_text:
                self._last_log_text = log_text
                self._dirty(*_DIRTY_LOG)
                dirty = True

            running = job.is_running()
            if running != self._last_running:
                self._last_running = running
                self._dirty(*_DIRTY_RUNNING)
                dirty = True

            result = job.result
            rk = self._result_key(result)
            if rk is not None and rk != self._last_result_key:
                self._last_result = result
                self._last_result_key = rk
                self._handle_job_finished(result)
                self._dirty(*_DIRTY_RESULT)
                dirty = True

        return dirty

    def on_unmount(self, doc):
        if self._job and self._job.is_running():
            self._job.cancel()
        doc.remove_data_model("hyworld2")
        self._doc = None
        self._handle = None

    # ------------------------------------------------------------------
    # Computed bindings
    def _input_summary(self) -> str:
        if not self.input_path:
            return "(no input selected)"
        return self.input_path

    def _stage_text(self) -> str:
        if not self._job:
            return "Idle"
        return self._job.stage.value.replace("_", " ").title()

    def _progress_value(self) -> str:
        if not self._job:
            return "0"
        return f"{max(0.0, min(1.0, self._job.progress / 100.0)):.4f}"

    def _progress_pct(self) -> str:
        if not self._job:
            return "0%"
        return f"{int(self._job.progress)}%"

    def _progress_status(self) -> str:
        return self._job.status if self._job else ""

    def _show_logs(self) -> bool:
        return bool(self._live_log_text())

    def _live_log_text(self) -> str:
        return self._job.log_text if self._job else ""

    def _show_results(self) -> bool:
        return self._last_result is not None and self._last_result.success

    def _show_error(self) -> bool:
        return self._last_result is not None and not self._last_result.success

    def _error_text(self) -> str:
        if self._last_result and not self._last_result.success:
            return self._last_result.error or "Unknown error"
        return ""

    def _result_output(self) -> str:
        return self._last_result.output_dir if self._last_result and self._last_result.success else ""

    def _result_frames(self) -> str:
        return str(self._last_result.num_frames) if self._last_result and self._last_result.success else "0"

    def _result_time(self) -> str:
        if self._last_result and self._last_result.success:
            return f"{self._last_result.elapsed_s:.1f}s"
        return ""

    def _result_has_gs(self) -> bool:
        return bool(self._last_result and self._last_result.success and self._last_result.gaussians_ply)

    def _result_has_points(self) -> bool:
        return bool(self._last_result and self._last_result.success and self._last_result.points_ply)

    def _result_has_scene_splats(self) -> bool:
        return bool(self._last_result and self._last_result.success and self._last_result.splat_node_name)

    def _result_has_scene_points(self) -> bool:
        return bool(self._last_result and self._last_result.success
                    and self._last_result.points_data is not None)

    def _training_node_text(self) -> str:
        try:
            scene = lf.get_scene()
            if scene is None:
                return ""
            return scene.training_model_node_name or ""
        except Exception:
            return ""

    def _can_use_scene(self) -> bool:
        try:
            return bool(lf.has_scene())
        except Exception:
            return False

    def _browse_input_label(self) -> str:
        return {
            "images": "Browse image folder",
            "video": "Browse video file",
            "colmap": "Browse COLMAP workspace",
        }.get(self.input_type, "Browse")

    # ------------------------------------------------------------------
    # Setters
    def _set_input_path(self, value) -> None:
        self.input_path = str(value or "").strip()
        self._dirty("input_path", "input_summary", "has_input", "can_run")

    def _set_input_type(self, value) -> None:
        v = str(value or "").strip().lower()
        if v in _INPUT_TYPES and v != self.input_type:
            self.input_type = v
            self._dirty(
                "input_type", "is_video_input", "is_image_input",
                "is_colmap_input", "browse_input_label",
            )

    def _set_prior_cam_path(self, value) -> None:
        self.prior_cam_path = str(value or "").strip()
        self._dirty("prior_cam_path")

    def _set_prior_depth_path(self, value) -> None:
        self.prior_depth_path = str(value or "").strip()
        self._dirty("prior_depth_path")

    def _set_output_dir(self, value) -> None:
        self.output_dir = str(value or "").strip()
        self._dirty("output_dir", "output_summary", "has_output", "can_run")

    def _set_output_mode(self, value) -> None:
        v = str(value or "").strip().lower()
        if v in ("direct", "dataset", "both") and v != self.output_mode:
            self.output_mode = v
            self._dirty("output_mode")

    def _set_int(self, name: str, value, lo: int, hi: int) -> None:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return
        parsed = max(lo, min(hi, parsed))
        if getattr(self, name) != parsed:
            setattr(self, name, parsed)
            self._dirty(name)

    def _set_bool(self, name: str, value) -> None:
        parsed = bool(value)
        if getattr(self, name) != parsed:
            setattr(self, name, parsed)
            self._dirty(name)

    def _set_compile_mode(self, value) -> None:
        v = str(value or "").strip().lower()
        if v in ("default", "reduce-overhead", "max-autotune") and v != self.compile_mode:
            self.compile_mode = v
            self._dirty("compile_mode")

    def _set_bf16(self, value) -> None:
        parsed = bool(value)
        if self.enable_bf16 == parsed:
            return
        self.enable_bf16 = parsed
        self._dirty("enable_bf16")
        # If a pipeline is already loaded with the opposite precision, warn
        # the user that the next Run / Load will rebuild the model.
        try:
            if pipeline_loader.is_loaded() and pipeline_loader.loaded_bf16() != parsed:
                lf.log.info(
                    f"[hyworld2] bf16 {'ON' if parsed else 'OFF'} — "
                    "the cached model will be reloaded on the next Run or Load."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sections
    def _sync_section_states(self) -> None:
        if not self._doc:
            return
        for name in ("advanced", "priors"):
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                content.set_class("collapsed", name in self._collapsed)
            if arrow:
                arrow.set_class("is-expanded", name not in self._collapsed)

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        self._sync_section_states()

    # ------------------------------------------------------------------
    # Event handlers
    def _on_browse_input(self, handle, event, args):
        """Context-aware browse: dispatches based on input_type."""
        if self.input_type == "video":
            self._on_browse_input_video(handle, event, args)
        elif self.input_type == "colmap":
            self._on_browse_colmap(handle, event, args)
        else:
            self._on_browse_input_folder(handle, event, args)

    def _on_browse_input_folder(self, handle, event, args):
        del handle, event, args
        start = self.input_path if os.path.isdir(self.input_path or "") else os.getcwd()
        picked = lf.ui.open_folder_dialog("Select image folder", start)
        if picked:
            self.input_type = "images"
            self._set_input_path(picked)
            self._dirty("input_type")

    def _on_browse_input_video(self, handle, event, args):
        del handle, event, args
        try:
            picked = lf.ui.open_video_file_dialog()
        except Exception as exc:
            lf.log.error(f"[hyworld2] Video dialog failed: {exc}")
            return
        if picked:
            self.input_type = "video"
            self._set_input_path(picked)
            self._dirty("input_type")

    def _on_browse_colmap(self, handle, event, args):
        del handle, event, args
        start = self.input_path if os.path.isdir(self.input_path or "") else os.getcwd()
        picked = lf.ui.open_folder_dialog("Select COLMAP workspace (with sparse/ and images/)", start)
        if not picked:
            return
        self._use_colmap_workspace(picked)

    def _on_use_current_scene(self, handle, event, args):
        del handle, event, args
        try:
            from lfs_plugins.ui.state import AppState
            scene_path = AppState.scene_path.value
        except Exception:
            scene_path = ""
        if not scene_path:
            lf.log.warn("[hyworld2] No current scene path available.")
            return
        p = Path(scene_path)
        if p.is_file():
            p = p.parent
        self._use_colmap_workspace(str(p))

    def _use_colmap_workspace(self, workspace: str) -> None:
        ws = Path(workspace)
        images_dir = ws / "images"
        if not images_dir.is_dir():
            lf.log.warn(f"[hyworld2] {images_dir} not found — select a workspace with an images/ subfolder.")
            return
        try:
            tmp = Path(tempfile.gettempdir()) / f"hyworld2_prior_{int(time.time())}.json"
            n = colmap_io.colmap_workspace_to_prior_json(ws, tmp)
        except Exception as exc:
            lf.log.error(f"[hyworld2] Failed to parse COLMAP workspace: {exc}")
            return
        self.input_type = "colmap"
        self._set_input_path(str(images_dir))
        self._set_prior_cam_path(str(tmp))
        lf.log.info(f"[hyworld2] Using COLMAP workspace {ws} ({n} cameras) as prior.")
        self._dirty("input_type")

    def _on_browse_output(self, handle, event, args):
        del handle, event, args
        start = self.output_dir if self.output_dir else str(Path.home())
        picked = lf.ui.open_folder_dialog("Select output folder", start)
        if picked:
            self._set_output_dir(picked)

    def _on_browse_prior_cam(self, handle, event, args):
        del handle, event, args
        start = os.path.dirname(self.prior_cam_path) if self.prior_cam_path else os.getcwd()
        picked = lf.ui.open_folder_dialog(
            "Select COLMAP workspace (with sparse/) or folder containing camera_params.json",
            start,
        )
        if not picked:
            return
        self._resolve_prior_source(picked)

    def _resolve_prior_source(self, path: str) -> None:
        """Turn a user-picked path into a hyworld2-ready prior_cam_path.

        Accepts three kinds of input:
          1. A COLMAP workspace (dir containing sparse/0/cameras.txt) →
             converted to a temp prior JSON via ``colmap_io``.
          2. A folder containing ``camera_params.json`` → used directly.
          3. A raw path (the user will have to point at a valid JSON
             themselves).
        """
        p = Path(path)
        # (1) COLMAP workspace
        sparse = colmap_io.find_sparse_dir(p)
        if sparse is not None:
            try:
                tmp = Path(tempfile.gettempdir()) / f"hyworld2_prior_{int(time.time())}.json"
                n = colmap_io.colmap_workspace_to_prior_json(p, tmp)
            except Exception as exc:
                lf.log.error(f"[hyworld2] Failed to parse COLMAP workspace: {exc}")
                return
            self._set_prior_cam_path(str(tmp))
            lf.log.info(f"[hyworld2] Using COLMAP workspace {p} ({n} cameras) as camera prior.")
            return
        # (2) hyworld2 native camera_params.json sitting next to the folder
        candidate = p / "camera_params.json"
        if candidate.is_file():
            self._set_prior_cam_path(str(candidate))
            lf.log.info(f"[hyworld2] Using camera_params.json at {candidate} as camera prior.")
            return
        # (3) raw
        self._set_prior_cam_path(str(p))
        lf.log.warn(
            f"[hyworld2] {p} has neither sparse/ nor camera_params.json — "
            "prior path set verbatim; hyworld2 will fail if it's not a valid JSON."
        )

    def _on_browse_prior_depth(self, handle, event, args):
        del handle, event, args
        start = self.prior_depth_path if os.path.isdir(self.prior_depth_path or "") else os.getcwd()
        picked = lf.ui.open_folder_dialog("Select prior depth folder", start)
        if picked:
            self._set_prior_depth_path(picked)

    def _on_start(self, handle, event, args):
        del handle, event, args
        if self._is_running():
            return
        if not self._can_run():
            lf.log.warn("[hyworld2] Need both input and output paths.")
            return

        # Reset per-job state
        self._last_result = None
        self._last_result_key = None
        self._loaded_result_key = None
        self._splats_loaded = False
        self._points_loaded = False
        self._last_log_text = ""
        self._last_stage = ""
        self._last_progress = -1.0
        self._last_status = ""

        cfg = JobConfig(
            input_path=self.input_path,
            output_dir=self.output_dir,
            output_mode=self.output_mode,
            target_size=self.target_size,
            fps=self.fps,
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            save_gs=self.save_gs,
            save_points=self.save_points,
            save_depth=self.save_depth,
            save_normal=self.save_normal,
            apply_sky_mask=self.apply_sky_mask,
            enable_bf16=self.enable_bf16,
            auto_unload_model_after_run=self.auto_unload_model_after_run,
            auto_fit_target_size=self.auto_fit_target_size,
            enable_compile=self.enable_compile,
            compile_mode=self.compile_mode,
            enable_fp32_heads=self.enable_fp32_heads,
            prior_cam_path=self.prior_cam_path,
            prior_depth_path=self.prior_depth_path,
        )
        self._job = HyWorld2Job(cfg)
        self._job.start()

        self._dirty(*_DIRTY_RUNNING, *_DIRTY_LOG, *_DIRTY_RESULT, *_DIRTY_RUN)

    def _on_cancel(self, handle, event, args):
        del handle, event, args
        if self._job and self._job.is_running():
            self._job.cancel()

    def _on_load_gaussians(self, handle, event, args):
        del handle, event, args
        r = self._last_result
        if not (r and r.success and r.gaussians_ply):
            return
        try:
            lf.load_file(r.gaussians_ply)
            self._splats_loaded = True
            self._dirty("splats_already_loaded")
        except Exception as exc:
            lf.log.error(f"[hyworld2] Failed to load gaussians: {exc}")

    def _on_load_points(self, handle, event, args):
        del handle, event, args
        r = self._last_result
        if not (r and r.success and r.points_ply):
            return
        try:
            lf.load_file(r.points_ply)
            self._points_loaded = True
            self._dirty("points_already_loaded")
        except Exception as exc:
            lf.log.error(f"[hyworld2] Failed to load points: {exc}")

    def _on_unload_model(self, handle, event, args):
        del handle, event, args
        def _task():
            pipeline_loader.unload()
            self._model_load_state = "idle"
            self._dirty("model_status_line", "show_load_btn", "show_unload_btn", "can_run")
        threading.Thread(target=_task, daemon=True).start()
        lf.log.info("[hyworld2] Unloading model in background.")

    def _on_download_models(self, handle, event, args):
        del handle, event, args
        downloads.start_background_download()

    def _on_reset_vram_profile(self, handle, event, args):
        del handle, event, args
        try:
            vram_profile.reset()
            lf.log.info("[hyworld2] VRAM calibration reset — next run re-probes from defaults.")
        except Exception as exc:
            lf.log.error(f"[hyworld2] Failed to reset VRAM profile: {exc}")

    def _on_load_model(self, handle, event, args):
        del handle, event, args
        if self._model_load_thread is not None and self._model_load_thread.is_alive():
            return
        # Allow reload if precision flag changed since load.
        if pipeline_loader.is_loaded() and pipeline_loader.loaded_bf16() == self.enable_bf16:
            return
        if not downloads.is_ready():
            lf.log.warn("[hyworld2] Models not downloaded yet; cannot load into GPU.")
            return
        self._model_load_state = "loading"
        self._model_load_error = ""
        self._dirty("model_status_line", "show_load_btn", "show_unload_btn", "model_loading")

        bf16_flag = self.enable_bf16

        def _task():
            try:
                pipeline_loader.get_pipeline(enable_bf16=bf16_flag)  # slow: ~9s
                self._model_load_state = "ready"
                lf.log.info(f"[hyworld2] Model loaded into GPU (bf16={'on' if bf16_flag else 'off'}).")
            except Exception as exc:
                self._model_load_state = "error"
                self._model_load_error = f"{type(exc).__name__}: {exc}"
                lf.log.error(f"[hyworld2] Model load failed: {exc}")
            finally:
                self._dirty("model_status_line", "show_load_btn", "show_unload_btn",
                            "model_loading", "can_run")
        self._model_load_thread = threading.Thread(target=_task, daemon=True)
        self._model_load_thread.start()

    def _on_train_from_splats(self, handle, event, args):
        del handle, event, args
        r = self._last_result
        if not (r and r.success and r.splat_node_name):
            lf.log.warn("[hyworld2] No in-scene splats from this run to train from.")
            return
        if direct_output.set_training_node(r.splat_node_name,
                                            log=lambda m: lf.log.info(f"[hyworld2] {m}")):
            self._dirty("training_node_text")

    def _on_train_from_points(self, handle, event, args):
        del handle, event, args
        r = self._last_result
        if not (r and r.success and r.points_data is not None):
            lf.log.warn("[hyworld2] No point cloud from this run to seed training.")
            return
        pts_np, cols_np = r.points_data
        init_name = f"HY-World-Mirror-2 ({Path(r.output_dir).name}) / train_init"
        created = direct_output.add_splats_from_points(
            pts_np, cols_np,
            node_name=init_name,
            parent_id=r.scene_node_id if r.scene_node_id >= 0 else -1,
            log=lambda m: lf.log.info(f"[hyworld2] {m}"),
        )
        if not created:
            return
        if direct_output.set_training_node(created,
                                            log=lambda m: lf.log.info(f"[hyworld2] {m}")):
            self._dirty("training_node_text")

    def _on_retry_download(self, handle, event, args):
        del handle, event, args
        downloads.start_background_download()

    def _on_clear_models(self, handle, event, args):
        del handle, event, args
        threading.Thread(target=self._clear_and_restart, daemon=True).start()

    def _clear_and_restart(self):
        if self._job and self._job.is_running():
            self._job.cancel()
        pipeline_loader.unload()
        downloads.delete_models()
        lf.log.info("[hyworld2] Cleared cached models.")

    # ------------------------------------------------------------------
    # Post-job
    def _handle_job_finished(self, result: JobResult | None) -> None:
        if result is None or not result.success:
            return
        if self._loaded_result_key == self._last_result_key:
            return
        self._loaded_result_key = self._last_result_key
        # "direct" mode pushes tensors into the scene but does NOT create an
        # LFS trainer — that requires a dataset on disk. Without a dataset,
        # the training panel shows "no trainer loaded". For training, use
        # "dataset" or "both".
        if self.output_mode == "direct":
            return
        if self.auto_load_dataset and result.output_dir and result.sparse_dir:
            try:
                # init_path seeds the trainer's starting splats from the
                # pretrained gaussians.ply so training refines them rather
                # than starting from scratch.
                init_ply = result.gaussians_ply or ""
                lf.log.info(f"[hyworld2] Loading dataset: {result.output_dir}"
                            + (f" (init={init_ply})" if init_ply else ""))
                lf.load_file(result.output_dir, is_dataset=True, init_path=init_ply)
            except Exception as exc:
                lf.log.error(f"[hyworld2] Failed to load dataset: {exc}")
