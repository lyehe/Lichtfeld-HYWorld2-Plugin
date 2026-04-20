"""Lightweight COLMAP text-format reader.

Parses ``cameras.txt`` + ``images.txt`` from a COLMAP sparse model and emits
a JSON file matching ``hyworld2``'s prior camera schema:

    {"num_cameras": N,
     "extrinsics": [{"camera_id": "<image_stem>", "matrix": [[4x4 world->cam]]}],
     "intrinsics": [{"camera_id": "<image_stem>", "matrix": [[3x3 K]]}]}

We key by image-name stem (``Path(name).stem``) which ``hyworld2``'s
``load_prior_camera`` handles natively (see inference_utils.py line 243).

No ``pycolmap`` dependency — COLMAP text format is a stable tiny spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class _Camera:
    model: str
    width: int
    height: int
    params: List[float]

    def intrinsic_matrix(self) -> List[List[float]]:
        m = self.model.upper()
        p = self.params
        if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif m in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                   "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _parse_cameras_txt(path: Path) -> Dict[int, _Camera]:
    cams: Dict[int, _Camera] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        cams[cam_id] = _Camera(
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(x) for x in parts[4:]],
        )
    return cams


def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    s = 2.0 / n
    xx = qx * qx * s; yy = qy * qy * s; zz = qz * qz * s
    xy = qx * qy * s; xz = qx * qz * s; yz = qy * qz * s
    wx = qw * qx * s; wy = qw * qy * s; wz = qw * qz * s
    return [
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ]


def _parse_images_txt(path: Path) -> List[Tuple[int, List[List[float]], List[float], int, str]]:
    """Return list of (image_id, R_3x3, t_3, camera_id, name).

    COLMAP writes two lines per image: the header (pose + name) followed by
    a POINTS2D line which may be empty. We skip comment lines and then pair
    the remaining lines — even indices are headers, odd indices are POINTS2D
    (which we ignore).
    """
    raw_lines = [
        raw for raw in path.read_text(encoding="utf-8").splitlines()
        if not raw.lstrip().startswith("#")
    ]
    out = []
    for i in range(0, len(raw_lines), 2):
        line = raw_lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        R = _quat_wxyz_to_rotmat(qw, qx, qy, qz)
        out.append((image_id, R, [tx, ty, tz], cam_id, name))
    return out


def find_sparse_dir(root: Path) -> Path | None:
    """Locate a COLMAP sparse/0 directory under ``root``.

    Accepts ``root`` being the workspace root (``<root>/sparse/0/``) or the
    sparse dir itself.
    """
    root = Path(root)
    candidates = [root, root / "sparse" / "0", root / "sparse"]
    for c in candidates:
        if (c / "cameras.txt").is_file() and (c / "images.txt").is_file():
            return c
    # Try any sparse/<n>/ subdir.
    sparse = root / "sparse"
    if sparse.is_dir():
        for sub in sorted(sparse.iterdir()):
            if (sub / "cameras.txt").is_file() and (sub / "images.txt").is_file():
                return sub
    return None


def colmap_workspace_to_prior_json(workspace: Path, output_json: Path) -> int:
    """Convert a COLMAP sparse model to a hyworld2 prior JSON. Returns N."""
    sparse = find_sparse_dir(workspace)
    if sparse is None:
        raise FileNotFoundError(
            f"No COLMAP cameras.txt/images.txt found under {workspace}"
        )

    cameras = _parse_cameras_txt(sparse / "cameras.txt")
    images = _parse_images_txt(sparse / "images.txt")
    if not images:
        raise RuntimeError(f"No images found in {sparse / 'images.txt'}")

    extrinsics_list = []
    intrinsics_list = []
    for _image_id, R, t, cam_id, name in images:
        stem = Path(name).stem
        extr = [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0,     0.0,     0.0,     1.0],
        ]
        K = cameras[cam_id].intrinsic_matrix()
        extrinsics_list.append({"camera_id": stem, "matrix": extr})
        intrinsics_list.append({"camera_id": stem, "matrix": K})

    payload = {
        "num_cameras": len(images),
        "extrinsics": extrinsics_list,
        "intrinsics": intrinsics_list,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(images)
