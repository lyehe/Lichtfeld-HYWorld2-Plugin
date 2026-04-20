# HY-World-Mirror-2.0 Plugin for LichtFeld Studio

A feed-forward 3D reconstruction plugin for LichtFeld Studio. Runs Tencent's [HunyuanWorld-Mirror 2](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) on images, a video, or a COLMAP workspace and pushes Gaussian splats + camera poses + a point cloud straight into the scene — ready to view, train, or export.

## Features

- **Three input modes**: image folder, video file, or existing COLMAP workspace (poses auto-extracted as prior).
- **Three output modes**:
  - **Direct** — tensors go straight into the LFS scene; in-memory trainer built via `lf.prepare_training_from_scene()`. The only files written are the staged JPGs in `<output_dir>/_frames/` that the scene cameras' `image_path` points at.
  - **Dataset** — writes a full COLMAP workspace (`images/`, `sparse/0/`, `gaussians.ply`, `points.ply`, `camera_params.json`) and auto-imports via `lf.load_file(..., is_dataset=True, init_path=gaussians.ply)`.
  - **Both** — scene preview (direct) AND the trainable COLMAP workspace on disk.
- **Training-init buttons** — after a run, pick "Train from splats" (refine the WorldMirror output) or "Train from points" (classical random init from the point cloud).
- **bf16 by default** on Ampere+ GPUs. Calibrated VRAM auto-fit avoids OOM on large frame counts.
- **torch.compile** with persistent Inductor + Triton cache, gated behind a toggle.
- **Auto-unload on training start** — frees the ~2.4 GB model from VRAM when the LFS trainer begins.

## Installation

### Via LichtFeld UI (recommended)

1. Open LichtFeld Studio
2. Go to the **Plugins** panel
3. Paste the GitHub URL: `https://github.com/lyehe/Lichtfeld-HYWorld2-Plugin`
4. Click **Install**
5. Restart LichtFeld Studio

### Via Python

```python
import lichtfeld as lf
lf.plugins.install("lyehe/Lichtfeld-HYWorld2-Plugin")
```

### Manual (dev symlink)

```powershell
# Windows — no admin; uses a directory junction
git clone https://github.com/lyehe/Lichtfeld-HYWorld2-Plugin
cd Lichtfeld-HYWorld2-Plugin
.\install.ps1
```

```bash
# Linux / macOS
git clone https://github.com/lyehe/Lichtfeld-HYWorld2-Plugin
cd Lichtfeld-HYWorld2-Plugin
./install.sh
```

On first load, LFS runs `uv sync` in the plugin dir to install the CUDA wheels (torch 2.11 cu130, gsplat, onnxruntime-gpu, triton-windows on Windows, etc.). Subsequent launches are near-instant. Model weights (~4.8 GB WorldMirror + 168 MB skyseg) download lazily into `<plugin>/models/` on first Run.

## Usage

1. Open the **HY-World-Mirror-2** panel in LichtFeld Studio.
2. Pick **Input type** — Image folder, Video file, or COLMAP workspace.
3. Click **Browse** and select the path.
4. Leave **Output mode** on **Direct** for fastest results, or switch to **Dataset** / **Both** if you want a COLMAP workspace on disk.
5. Click **Load model** to pay the ~9 s model-to-GPU cost now (optional — Run does it lazily otherwise).
6. Click **Run Reconstruction**.
7. When the run finishes, either:
   - Explore the scene directly (splats, cameras, point cloud are already there), or
   - Click **Train from splats** / **Train from points** and hit Start Training in the built-in Training panel.

## Configuration

Settings (collapsible in the panel):

- **Target size** — longest-edge pixel count the model sees. Higher = more detail, quadratically more VRAM. Auto-fit treats this as an upper bound.
- **bfloat16 inference** — ~half VRAM, Ampere+ only. On by default when supported.
- **Auto-fit target size** — probes free VRAM before inference and drops the effective target if needed.
- **Unload model after each Run** — releases ~2.4 GB bf16 / 4.7 GB fp32. Costs ~9 s on next Run.
- **Reset VRAM calibration** — wipe the learned profile (useful after GPU swap).
- **torch.compile** (experimental) — 30-60 s first-forward compile cost; cached to disk across sessions.
- **FP32 heads** — slower, tighter numerical parity with an fp32 baseline.

## Output

**Direct mode** adds a scene group per run:

```
HY-World-Mirror-2 (run_<timestamp>)
├── splats        (pruned Gaussian splats — training model)
├── cameras       (one node per frame, with image_path)
└── points        (back-projected colored point cloud)
```

**Direct mode** also writes a single directory to your chosen output folder so the scene cameras have real image files to reference:

```
<output_dir>/
└── _frames/                      (input frames at inference resolution — scene cameras' image_path targets)
```

**Dataset mode** writes the full COLMAP workspace (no `_frames/`):

```
<output_dir>/
├── images/                       (frames resized to sparse/0/cameras.txt dims)
├── sparse/0/
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt              (populated from points.ply)
├── gaussians.ply                 (pretrained Gaussian splats, logit opacity)
├── points.ply                    (colored point cloud)
└── camera_params.json
```

**Both mode** writes the union of the two trees (`_frames/` for scene cameras + the COLMAP workspace for training).

## Requirements

- **GPU**: CUDA-capable. Ampere-class (RTX 30xx) or newer recommended for bf16. Turing and older work with bf16 off.
- **VRAM**: ~4 GB for small runs; 12-24 GB recommended for 32-frame / high-res runs.
- **Disk**: ~5 GB for the plugin venv + ~5 GB for model weights.

## Model weights

Downloaded lazily from HuggingFace on first Run:

- `tencent/HY-World-2.0` (subfolder: `HY-WorldMirror-2.0`) — ~4.8 GB, under the **Tencent HY-World 2.0 Community License** (geographically restricted).
- `JianyuanWang/skyseg` — 168 MB.

The plugin does not redistribute weights; they come from HuggingFace under their respective licenses.

## License

MIT — see [LICENSE](LICENSE) for the plugin code, [NOTICE](NOTICE) for third-party attributions.
