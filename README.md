# HY-World-Mirror-2.0 · LichtFeld Studio Plugin

Runs Tencent's [HunyuanWorld-Mirror 2](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) 3D reconstruction model inside LichtFeld Studio — feed it images, a video, or a COLMAP workspace and get Gaussian splats + poses + a point cloud either pushed straight into the scene (no files written) or saved as a trainable COLMAP dataset.

## What you get

- **Input:** image folder, video file, or existing COLMAP workspace.
- **Optional priors:** camera JSON and/or per-frame depth folder.
- **"Use current LFS scene as prior"** — if the current scene was loaded from a COLMAP-style dataset, the plugin converts its poses + intrinsics into a prior and feeds them to inference.
- **Three output modes:**
  - **Direct** (default) — tensors go straight into the scene via `scene.add_splat` / `scene.add_camera` / `scene.add_point_cloud`; an in-memory trainer is built via `lf.prepare_training_from_scene()`. Zero files written.
  - **Dataset** — writes a COLMAP workspace (`images/`, `sparse/0/`, `gaussians.ply`, `points.ply`, `camera_params.json`) and auto-imports it. Best when you want the dataset persisted for later re-training.
  - **Both** — direct + dataset.
- **Training-init buttons** after a run — "Train from splats" (refine WorldMirror's output) or "Train from points" (classical SfM-style random init from the point cloud).

## Performance

- **bf16 by default** on Ampere+ GPUs (auto-detected). ~half the VRAM of fp32.
- **Auto-fit target size** — probes free VRAM before inference and drops the effective `target_size` as needed. Avoids OOM on large frame counts.
- **Calibrated VRAM profile** — learns your GPU's real bytes-per-pixel from actual runs (EWMA-smoothed), stored at `<plugin>/models/vram_profile.json`.
- **Perf flags** lifted from the Playground backend: `cudnn.benchmark`, TF32 on Ampere+, `expandable_segments` (Linux only).
- **Optional torch.compile** with a persistent Inductor + Triton cache at `<plugin>/cache/`. First forward at a new shape pays 30–60 s; subsequent forwards (and subsequent LFS launches) hit the cache.
- **Warmup on Load Model** — a 2-frame 280×280 dummy forward pays cuDNN selection + PTX→SASS JIT + (optional) `torch.compile` up front.
- **Auto-unload on training start** — frees the ~2.4 GB bf16 / 4.7 GB fp32 WorldMirror weights from VRAM when the LFS trainer starts.

## Install

**Fully self-contained** — the `hyworld2` Python package is vendored inside the plugin. No external Playground checkout needed. Model weights (~4.8 GB WorldMirror + 168 MB skyseg) download lazily into `<plugin>/models/` on first use.

```powershell
# Windows — no admin; uses a directory junction
.\install.ps1
```

```bash
# Linux / macOS
./install.sh
```

This creates one link at `~/.lichtfeld/plugins/hyworld2_plugin/` pointing at the plugin directory. Launch LichtFeld Studio; on first load LFS runs `uv sync` in the plugin dir to install the CUDA wheels (torch 2.11 cu130, gsplat 1.5.3, onnxruntime-gpu, triton-windows on Windows, etc.). After that, plugin loads in ~2 s.

Uninstall: `.\uninstall.ps1` / `./uninstall.sh` removes the junction, the `models/` cache, and the `cache/` (torch compile) cache.

## Requirements

- CUDA GPU — Ampere-class (RTX 30xx) or newer for bf16 default. Turing and older work with bf16 off.
- LFS's bundled Python 3.12. `uv sync` pulls everything else.
- ~10 GB free disk for the plugin venv + model cache.

## Architecture

```
Run click
  │
  ├─► prepare_input  (glob or extract frames)
  ├─► pipeline_loader.get_pipeline(bf16, compile, ...)
  │     └─ cached; reloads only on flag change
  ├─► auto_fit target_size ← vram_profile + free VRAM probe
  ├─► intercept save_results → capture predictions dict
  ├─► pipeline(...) forward pass                    ← INFERENCE
  ├─► vram_profile.record_run  (isolated activation bytes)
  │
  ├─ direct mode:
  │   └─► resave staged frames at inference resolution → outdir/_frames/
  │       scene.add_splat (log-scaled, logit opacity, voxel-pruned)
  │       scene.add_camera (world→cam, fx/fy for actual image dims)
  │       scene.add_point_cloud (back-projected from depth)
  │       lf.prepare_training_from_scene()          ← IN-MEMORY TRAINER
  │
  └─ dataset mode:
      └─► _populate_images_dir (resize originals to cameras.txt dims)
          _populate_points3d (hydrate from points.ply)
          lf.load_file(outdir, is_dataset=True, init_path=gaussians.ply)
```

## Known limits

- Single-GPU only. Multi-GPU FSDP is not exposed.
- Inference is not interruptible — Cancel takes effect at stage boundaries, not mid-forward.
- `torch.compile` is experimental; failures fall back to eager execution silently (check the `Build:` status line to see whether compile is actually active).

## Licensing

- **Plugin code** — MIT. See [LICENSE](LICENSE).
- **Vendored `hyworld2/`** — MIT, copied from [filliptm/HY-World-2.0-Playground](https://github.com/filliptm/HY-World-2.0-Playground). See [NOTICE](NOTICE).
- **WorldMirror model weights** — downloaded at runtime from `tencent/HY-World-2.0` on HuggingFace under the **Tencent HY-World 2.0 Community License**, which is geographically restricted. The plugin does not redistribute weights; users obtain them directly from HuggingFace under that license.

## Credits

- Tencent for the HunyuanWorld-Mirror model.
- [@filliptm](https://github.com/filliptm) for the Playground fork whose perf work (SDPA priority, warmup, vectorized voxel prune, fp16/bf16 head gating) this plugin inherits through the vendored package.
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) for the plugin host.
