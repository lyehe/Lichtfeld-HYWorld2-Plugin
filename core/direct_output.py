"""Push hyworld2 predictions directly into an LFS scene — no file I/O.

The pipeline normally writes PLYs + a COLMAP sparse model + images, then
tells LFS to re-load them from disk. That's slow and wastes disk. LFS's
Scene Python API accepts tensors directly:

  Scene.add_splat(means, sh0, shN, scaling, rotation, opacity, ...)
  Scene.add_camera(R, T, fx, fy, w, h, ...)

Tensors move zero-copy via ``lf.Tensor.from_dlpack(torch_tensor)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import lichtfeld as lf


SH_C0 = 0.28209479177387814  # Same constant hyworld2 uses for f_dc <-> rgb.


def apply_predictions_to_scene(
    predictions: dict,
    img_paths: Iterable,
    imgs=None,
    filter_mask=None,
    gs_filter_mask=None,
    *,
    node_name: str = "HY-World-Mirror-2",
    attach_images: bool = True,
    add_point_cloud: bool = True,
    point_cloud_max_points: int = 500_000,
    gs_max_points: int = 5_000_000,
    voxel_prune_size: float = 0.002,
    log: Optional[callable] = None,
) -> Optional[dict]:
    """Materialise splats + cameras + optional point cloud as LFS scene nodes.

    Returns a dict with info needed by the panel for training-mode switches:
      {parent_id, splat_node_name, points_node_name, points_np, colors_np}
    or ``None`` if there's no scene.
    """
    _log = log or (lambda _msg: None)
    try:
        scene = lf.get_scene()
    except Exception as exc:
        _log(f"direct output: lf.get_scene() failed: {exc}")
        return None
    if scene is None:
        _log("direct output: no active scene; skipping.")
        return None

    # Dedupe by name so re-running cleanly replaces the previous output.
    try:
        scene.remove_node(node_name, keep_children=False)
    except Exception:
        pass

    parent_id = scene.add_group(node_name)

    splat_name = _add_splats(
        scene, predictions, parent_id, node_name, _log,
        filter_mask=filter_mask, gs_filter_mask=gs_filter_mask,
        gs_max_points=gs_max_points, voxel_prune_size=voxel_prune_size,
    )
    # Splat buffers are now held via DLPack by the LFS scene; the
    # predictions["splats"] torch refs are redundant. Drop them so the
    # caller's captured dict doesn't keep a second-owner on ~100-500 MB
    # of GPU memory through the camera / point-cloud work below.
    try:
        predictions.pop("splats", None)
    except Exception:
        pass

    _add_cameras(scene, predictions, list(img_paths), parent_id, node_name, attach_images, _log, imgs=imgs)
    pts_np, cols_np, points_name = (None, None, "")
    if add_point_cloud and imgs is not None:
        pts_np, cols_np, points_name = _add_point_cloud(
            scene, predictions, imgs, parent_id, node_name, point_cloud_max_points, _log
        )

    try:
        scene.notify_changed()
    except Exception as exc:
        _log(f"direct output: scene.notify_changed() failed: {exc}")

    # Spin up an LFS trainer directly from our in-scene cameras + point cloud
    # (no disk dataset needed). lf.prepare_training_from_scene() wraps the
    # C++ scene_manager::prepareTrainingFromScene path, which builds a
    # Trainer around scene_.getAllCameras() + the visible point cloud.
    trainer_ready = False
    if hasattr(lf, "prepare_training_from_scene"):
        try:
            lf.prepare_training_from_scene()
            trainer_ready = True
            _log("direct output: trainer initialized from scene (no disk dataset needed).")
        except Exception as exc:
            _log(f"direct output: prepare_training_from_scene() failed: {exc}")
    else:
        _log("direct output: lf.prepare_training_from_scene() unavailable; training panel will show 'no trainer loaded'.")

    return {
        "parent_id": parent_id,
        "splat_node_name": splat_name,
        "points_node_name": points_name,
        "points_np": pts_np,
        "colors_np": cols_np,
        "trainer_ready": trainer_ready,
    }


def add_splats_from_points(
    points_np,
    colors_np,
    *,
    node_name: str,
    parent_id: int = -1,
    init_opacity: float = 0.1,
    init_scale: Optional[float] = None,
    log: Optional[callable] = None,
) -> Optional[str]:
    """Create a splat node from a point cloud — seed for random-init training.

    Each point becomes one isotropic gaussian:
      means = point positions
      rotation = identity quaternion (wxyz = [1, 0, 0, 0])
      scaling = log(init_scale) per axis (auto-derived from bbox if None)
      sh0 = f_dc = (rgb_in_0_1 - 0.5) / SH_C0  (hyworld2 convention)
      opacity = logit(init_opacity)

    Returns the created splat node's name.
    """
    _log = log or (lambda _msg: None)
    import math
    import numpy as np
    import torch

    try:
        scene = lf.get_scene()
    except Exception as exc:
        _log(f"train-from-points: lf.get_scene() failed: {exc}")
        return None
    if scene is None:
        _log("train-from-points: no active scene.")
        return None

    n = int(points_np.shape[0])
    if n == 0:
        _log("train-from-points: empty point cloud; nothing to do.")
        return None

    # Auto-scale heuristic: bbox diagonal / n^(1/3) gives a rough per-point
    # extent. A fraction of that is a reasonable seed.
    if init_scale is None or init_scale <= 0:
        bbox_min = points_np.min(axis=0)
        bbox_max = points_np.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        density_step = diag / max(1.0, n ** (1.0 / 3.0))
        init_scale = max(1e-5, density_step * 0.5)
    log_scale = math.log(init_scale)

    # RGB (0-1) → f_dc (hyworld2's SH DC encoding).
    rgb_01 = colors_np.astype(np.float32)
    if rgb_01.max() > 1.5:
        rgb_01 = rgb_01 / 255.0
    f_dc = (rgb_01 - 0.5) / SH_C0

    # Assemble tensors on CPU; LFS will copy to GPU inside add_splat.
    means = torch.from_numpy(np.ascontiguousarray(points_np.astype(np.float32))).contiguous()
    sh0 = torch.from_numpy(np.ascontiguousarray(f_dc)).unsqueeze(1).contiguous()  # [N, 1, 3]
    shN = torch.zeros((n, 0, 3), dtype=torch.float32)
    scaling = torch.full((n, 3), log_scale, dtype=torch.float32)
    rotation = torch.zeros((n, 4), dtype=torch.float32)
    rotation[:, 0] = 1.0  # wxyz identity
    opacity_logit = math.log(init_opacity / (1.0 - init_opacity))
    opacity = torch.full((n, 1), opacity_logit, dtype=torch.float32)

    # Dedupe by name.
    try:
        scene.remove_node(node_name, keep_children=False)
    except Exception:
        pass

    try:
        scene.add_splat(
            name=node_name,
            means=_to_lf(means),
            sh0=_to_lf(sh0),
            shN=_to_lf(shN),
            scaling=_to_lf(scaling),
            rotation=_to_lf(rotation),
            opacity=_to_lf(opacity),
            sh_degree=0,
            scene_scale=1.0,
            parent=parent_id,
        )
    except Exception as exc:
        _log(f"train-from-points: add_splat failed: {exc}")
        return None
    _log(f"train-from-points: created {node_name} with {n} gaussians (init_scale={init_scale:.5f}).")
    try:
        scene.notify_changed()
    except Exception:
        pass
    return node_name


def set_training_node(node_name: str, log: Optional[callable] = None) -> bool:
    """Tell LFS to use ``node_name`` as the training model node."""
    _log = log or (lambda _msg: None)
    try:
        scene = lf.get_scene()
        if scene is None:
            _log("set_training_node: no active scene.")
            return False
        scene.set_training_model_node(node_name)
        _log(f"Training node set to: {node_name}")
        return True
    except Exception as exc:
        _log(f"set_training_node failed: {exc}")
        return False


# ----- internals --------------------------------------------------------

def _to_lf(tensor):
    """Convert a torch tensor to lf.Tensor (zero-copy when possible).

    Earlier we tried ``.clone()`` before DLPack export to give LFS its own
    buffer, but the clone goes out of scope immediately after this function
    returns — and DLPack's ownership transfer was not robust enough on the
    Windows/torch path to keep the buffer alive for LFS. The result was an
    NvCodec "Decoder creation failed: EXECUTION_FAILED" followed by CUDA
    illegal memory access when training later sampled the splats.
    Zero-copy works reliably as long as the source tensor stays alive via
    its enclosing captured/predictions dict for the duration of the scene.
    """
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        return lf.Tensor.from_dlpack(tensor)
    except Exception:
        # Fallback via numpy (forces CPU copy)
        import numpy as np
        arr = tensor.cpu().numpy().astype(np.float32, copy=False)
        return lf.Tensor.from_numpy(arr)


def _add_splats(
    scene, predictions, parent_id, node_name, log,
    *, filter_mask=None, gs_filter_mask=None,
    gs_max_points: int = 5_000_000, voxel_prune_size: float = 0.002,
):
    if "splats" not in predictions:
        log("direct output: no 'splats' in predictions; skipping splat node.")
        return ""
    import numpy as np
    import torch
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import _voxel_prune_gaussians

    sp = predictions["splats"]
    # Keep tensors on their source device (CUDA) through voxel prune. The
    # vendored _voxel_prune_gaussians is now device-aware, so we avoid a
    # 6-tensor d2h round-trip (~100-300 MB) and the CPU-side math cost.
    means = sp["means"][0].reshape(-1, 3).detach().float().contiguous()
    scales_linear = sp["scales"][0].reshape(-1, 3).detach().float().contiguous()
    quats = sp["quats"][0].reshape(-1, 4).detach().float().contiguous()
    colors = (sp["sh"][0] if "sh" in sp else sp["colors"][0]).reshape(-1, 3).detach().float().contiguous()
    opacities_prob = sp["opacities"][0].reshape(-1).detach().float().contiguous()
    weights = (sp["weights"][0] if "weights" in sp else torch.ones_like(opacities_prob)).reshape(-1).detach().float().contiguous()

    n_raw = means.shape[0]

    # (1) Apply the same mask save_results uses (sky / confidence / edge).
    keep = None
    if gs_filter_mask is not None:
        keep = torch.from_numpy(np.asarray(gs_filter_mask).reshape(-1)).bool()
    elif filter_mask is not None:
        keep = torch.from_numpy(np.asarray(filter_mask).reshape(-1)).bool()
    if keep is not None and keep.shape[0] == n_raw:
        means = means[keep]
        scales_linear = scales_linear[keep]
        quats = quats[keep]
        colors = colors[keep]
        opacities_prob = opacities_prob[keep]
        weights = weights[keep]

    # (2) Voxel-prune (weighted merge of co-located gaussians).
    n_pre_prune = means.shape[0]
    means, scales_linear, quats, colors, opacities_prob = _voxel_prune_gaussians(
        means, scales_linear, quats, colors, opacities_prob, weights,
        voxel_size=voxel_prune_size,
    )

    # (3) Cap total count with a deterministic random subsample.
    if gs_max_points > 0 and means.shape[0] > gs_max_points:
        idx = torch.from_numpy(
            np.random.default_rng(42).choice(means.shape[0], size=gs_max_points, replace=False)
        ).long()
        means = means[idx]
        scales_linear = scales_linear[idx]
        quats = quats[idx]
        colors = colors[idx]
        opacities_prob = opacities_prob[idx]

    # (4) Invert activations so LFS (which exp()s / sigmoid()s internally) sees
    # the same numeric scale/opacity gsplat does.
    scales = scales_linear.clamp(min=1e-8).log()                           # -> log-scale
    opacities = torch.logit(opacities_prob.clamp(1e-6, 1.0 - 1e-6))        # -> logit

    # LFS's add_splat expects:
    #   sh0       : [N, 1, 3]   (DC term)
    #   shN       : [N, K, 3]   (higher-order; empty for sh_degree=0)
    #   opacity   : [N, 1]
    sh0 = colors.unsqueeze(1).float()
    shN = torch.zeros((means.shape[0], 0, 3), dtype=torch.float32, device=colors.device)
    opacity = opacities.unsqueeze(1).float()

    log(f"direct output: splats {n_raw} raw -> {n_pre_prune} after mask -> {means.shape[0]} final.")

    splat_name = f"{node_name} / splats"
    splat_id = scene.add_splat(
        name=splat_name,
        means=_to_lf(means.float()),
        sh0=_to_lf(sh0),
        shN=_to_lf(shN),
        scaling=_to_lf(scales.float()),
        rotation=_to_lf(quats.float()),
        opacity=_to_lf(opacity),
        sh_degree=0,
        scene_scale=1.0,
        parent=parent_id,
    )
    log(f"direct output: added splat node id={splat_id} with {means.shape[0]} gaussians.")
    return splat_name


def _add_cameras(scene, predictions, img_paths, parent_id, node_name, attach_images, log, imgs=None):
    if "camera_poses" not in predictions or "camera_intrs" not in predictions:
        log("direct output: no camera_poses/intrs in predictions; skipping camera nodes.")
        return
    import numpy as np
    from PIL import Image

    # predictions["camera_poses"] is CAMERA-TO-WORLD (worldmirror.py inverts
    # w2c before storing). LFS's add_camera expects WORLD-TO-CAMERA, so we
    # invert each pose back to w2c.
    poses_c2w = predictions["camera_poses"][0].detach().cpu().float().numpy()   # [S, 4, 4]
    poses = np.linalg.inv(poses_c2w)                                            # -> world->cam
    intrs = predictions["camera_intrs"][0].detach().cpu().float().numpy()       # [S, 3, 3]
    S = poses.shape[0]

    # predictions["camera_intrs"] is scaled to the model's inference image
    # size (the H, W of `imgs`). If the staged frame on disk is a different
    # size (hyworld2 keeps a higher-res original), we need to scale fx / fy
    # proportionally so LFS's assumed principal point (w/2, h/2) lines up.
    if imgs is not None:
        try:
            inf_h = int(imgs.shape[-2])
            inf_w = int(imgs.shape[-1])
        except Exception:
            inf_h, inf_w = 0, 0
    else:
        inf_h, inf_w = 0, 0

    # Normalise image paths to absolute + forward-slash so LFS's loader
    # always finds them regardless of cwd. Verify each exists.
    resolved_paths: list[str] = []
    missing = 0
    for i in range(S):
        if not attach_images or i >= len(img_paths):
            resolved_paths.append("")
            continue
        p = Path(img_paths[i]).resolve()
        if not p.is_file():
            missing += 1
            resolved_paths.append("")
        else:
            resolved_paths.append(str(p).replace("\\", "/"))
    if missing:
        log(f"direct output: {missing} camera image(s) missing on disk; those cameras will have no image.")

    # Sizes from the actual image files (authoritative).
    sizes = []
    for p in resolved_paths:
        if not p:
            sizes.append((0, 0))
            continue
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (w, h)
        except Exception as exc:
            log(f"direct output: failed to read size from {p}: {exc}")
            sizes.append((0, 0))

    cam_group_id = scene.add_camera_group(f"{node_name} / cameras", parent_id, S)

    ok = 0
    for i in range(S):
        R = np.ascontiguousarray(poses[i, :3, :3], dtype=np.float32)
        T = np.ascontiguousarray(poses[i, :3, 3:4], dtype=np.float32)  # [3, 1]
        fx_inf = float(intrs[i, 0, 0])
        fy_inf = float(intrs[i, 1, 1])
        w, h = sizes[i]
        # Scale focal length to match the image dims we hand LFS.
        if inf_w > 0 and inf_h > 0 and w > 0 and h > 0:
            fx = fx_inf * (w / inf_w)
            fy = fy_inf * (h / inf_h)
        else:
            fx, fy = fx_inf, fy_inf
        image_path = resolved_paths[i]
        try:
            scene.add_camera(
                name=f"{node_name} / cam_{i + 1:04d}",
                parent=cam_group_id,
                R=lf.Tensor.from_numpy(R),
                T=lf.Tensor.from_numpy(T),
                focal_x=fx, focal_y=fy,
                width=int(w), height=int(h),
                image_path=image_path,
                uid=i,  # Stable per-run uid so the trainer can reference cams
            )
        except Exception as exc:
            log(f"direct output: add_camera #{i} failed: {exc}")
            break
        ok += 1
    linked = sum(1 for p in resolved_paths if p)
    log(f"direct output: added {ok}/{S} cameras; "
        f"{linked}/{S} linked to images ({sizes[0] if sizes else 'n/a'}).")


def _add_point_cloud(scene, predictions, imgs, parent_id, node_name, max_points, log):
    """Back-project depth maps into a coloured point cloud.

    Returns (pts_np [N,3] float32, cols_np [N,3] uint8, node_name) — these
    arrays are passed back up so the panel can later call
    ``add_splats_from_points`` for random-init training.
    """
    if "depth" not in predictions or "camera_params" not in predictions:
        log("direct output: no depth/camera_params in predictions; skipping point cloud.")
        return None, None, ""
    import numpy as np
    import torch
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import _compute_points_from_depth
    from hyworld2.worldrecon.hyworldmirror.models.utils.camera_utils import vector_to_camera_matrices

    B, S, C, H, W = imgs.shape
    e3x4, intr = vector_to_camera_matrices(predictions["camera_params"], image_hw=(H, W))
    pts_np, cols_np = _compute_points_from_depth(
        predictions["depth"], imgs, e3x4[0], intr[0], S, H, W, filter_mask=None
    )
    n_total = int(pts_np.shape[0])
    if n_total == 0:
        log("direct output: no valid depth points; skipping point cloud.")
        return None, None, ""

    # Subsample deterministically to cap scene memory.
    if n_total > max_points:
        idx = np.linspace(0, n_total - 1, max_points, dtype=np.int64)
        pts_np = pts_np[idx]
        cols_np = cols_np[idx]

    # LFS add_point_cloud wants [N,3] positions and [N,3] or [N,4] colors (0..1 float).
    pts = torch.from_numpy(np.ascontiguousarray(pts_np.astype(np.float32))).contiguous()
    cols_f = np.ascontiguousarray((cols_np.astype(np.float32) / 255.0))
    cols = torch.from_numpy(cols_f).contiguous()
    points_name = f"{node_name} / points"
    try:
        scene.add_point_cloud(
            name=points_name,
            points=_to_lf(pts),
            colors=_to_lf(cols),
            parent=parent_id,
        )
        log(f"direct output: added point cloud with {pts.shape[0]} points (of {n_total} total).")
    except Exception as exc:
        log(f"direct output: add_point_cloud failed: {exc}")
        points_name = ""
    return pts_np, cols_np, points_name
