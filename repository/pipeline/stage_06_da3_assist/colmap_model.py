from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config_access import cfg_get
from .io_utils import ensure_dir


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        model = self.model.upper()
        p = self.params
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            f = float(p[0])
            return f, f, float(p[1]), float(p[2])
        if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        # Conservative fallback for unknown camera models.
        f = float(p[0]) if p else max(self.width, self.height)
        return f, f, self.width / 2.0, self.height / 2.0


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)


@dataclass
class ColmapTextModel:
    cameras: dict[int, Camera]
    images: dict[int, ImagePose]
    points3d: dict[int, np.ndarray]


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError("qvec must have four values")
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def export_sparse_text_model(cfg: Any, sparse_model_dir: Path, text_dir: Path) -> Path:
    if (text_dir / "cameras.txt").exists() and (text_dir / "images.txt").exists() and (text_dir / "points3D.txt").exists():
        return text_dir
    colmap = str(cfg_get(cfg, "colmap.executable", "colmap"))
    if colmap == "auto":
        colmap = "colmap"
    if shutil.which(colmap) is None and not Path(colmap).exists():
        raise FileNotFoundError(f"COLMAP executable not found for Stage 6 model export: {colmap}")
    ensure_dir(text_dir)
    cmd = [
        colmap,
        "model_converter",
        "--input_path",
        str(sparse_model_dir),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "COLMAP model_converter failed while exporting Stage 6 sparse model.\n"
            f"Command: {' '.join(cmd)}\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
        )
    return text_dir


def read_colmap_text_model(text_dir: Path) -> ColmapTextModel:
    cameras = _read_cameras(text_dir / "cameras.txt")
    points = _read_points3d(text_dir / "points3D.txt")
    images = _read_images(text_dir / "images.txt")
    return ColmapTextModel(cameras=cameras, images=images, points3d=points)


def _non_comment_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.startswith("#")]


def _read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _non_comment_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=parts[1],
            width=int(float(parts[2])),
            height=int(float(parts[3])),
            params=[float(v) for v in parts[4:]],
        )
    return cameras


def _read_points3d(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for line in _non_comment_lines(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        point_id = int(parts[0])
        points[point_id] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return points


def _read_images(path: Path) -> dict[int, ImagePose]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip() and not line.startswith("#")]
    images: dict[int, ImagePose] = {}
    idx = 0
    while idx < len(lines):
        header = lines[idx].split()
        if len(header) < 10:
            idx += 1
            continue
        image_id = int(header[0])
        qvec = np.array([float(v) for v in header[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in header[5:8]], dtype=np.float64)
        camera_id = int(header[8])
        name = " ".join(header[9:])
        point_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        vals = point_line.split()
        xy: list[list[float]] = []
        pids: list[int] = []
        for j in range(0, len(vals) - 2, 3):
            xy.append([float(vals[j]), float(vals[j + 1])])
            pids.append(int(float(vals[j + 2])))
        images[image_id] = ImagePose(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=np.asarray(xy, dtype=np.float64) if xy else np.zeros((0, 2), dtype=np.float64),
            point3d_ids=np.asarray(pids, dtype=np.int64) if pids else np.zeros((0,), dtype=np.int64),
        )
        idx += 2
    return images


def camera_depth_for_point(image: ImagePose, point_world: np.ndarray) -> float:
    cam = image.rotation @ point_world.reshape(3) + image.tvec
    return float(cam[2])


def image_by_name(model: ColmapTextModel) -> dict[str, ImagePose]:
    """Return COLMAP image poses keyed by image filename.

    Shared helper used by Stage 6 depth alignment/fusion.
    """
    return {image.name: image for image in model.images.values()}
