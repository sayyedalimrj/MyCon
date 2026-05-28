from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .colmap_model import ColmapTextModel, image_by_name
from .config_access import bool_value, cfg_get
from .depth_alignment import AlignmentResult, load_depth_map
from .io_utils import ensure_dir, write_json_atomic



def _parse_bbox(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() in {"", "null", "None"}:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            parts = [float(v.strip()) for v in value.split(",") if v.strip()]
            value = parts
    if isinstance(value, dict):
        mn = value.get("min") or value.get("minimum")
        mx = value.get("max") or value.get("maximum")
        if mn is None or mx is None:
            return None
        min_xyz = np.asarray(mn, dtype=np.float64)
        max_xyz = np.asarray(mx, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            return None
        min_xyz = arr[:3]
        max_xyz = arr[3:]
    if min_xyz.size != 3 or max_xyz.size != 3:
        return None
    return np.minimum(min_xyz, max_xyz), np.maximum(min_xyz, max_xyz)


def _depth_edge_mask(depth: np.ndarray, threshold_m: float, relative_threshold: float) -> np.ndarray:
    valid_depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(valid_depth) & (valid_depth > 0)
    if not finite.any():
        return np.ones_like(valid_depth, dtype=bool)
    # Median filter stabilizes single-pixel depth spikes before Sobel.
    filtered = valid_depth.copy()
    filtered[~finite] = 0
    filtered = cv2.medianBlur(filtered, 3)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    local_thresh = np.maximum(float(threshold_m), float(relative_threshold) * np.maximum(filtered, 1e-6))
    return (grad_mag > local_thresh) | (~finite)


def _camera_backproject(
    depth: np.ndarray,
    image_rgb: np.ndarray | None,
    camera: Any,
    pose: Any,
    stride: int,
    min_depth: float,
    max_depth: float,
    *,
    edge_filter_enabled: bool,
    edge_threshold_m: float,
    edge_relative_threshold: float,
    bbox: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = camera.intrinsics
    h, w = depth.shape[:2]
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x].astype(np.float64)
    mask = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)

    if edge_filter_enabled:
        edge_mask = _depth_edge_mask(depth, threshold_m=edge_threshold_m, relative_threshold=edge_relative_threshold)
        mask &= ~edge_mask[grid_y, grid_x]

    if not mask.any():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    x = (grid_x[mask].astype(np.float64) - cx) * z[mask] / fx
    y = (grid_y[mask].astype(np.float64) - cy) * z[mask] / fy
    cam_pts = np.vstack([x, y, z[mask]]).T
    world_pts = (pose.rotation.T @ (cam_pts - pose.tvec.reshape(1, 3)).T).T

    if bbox is not None and world_pts.size:
        mn, mx = bbox
        inside = np.all((world_pts >= mn.reshape(1, 3)) & (world_pts <= mx.reshape(1, 3)), axis=1)
        if not inside.any():
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        world_pts = world_pts[inside]
        # Reduce masks to final pixel locations for color sampling.
        gx = grid_x[mask][inside]
        gy = grid_y[mask][inside]
    else:
        gx = grid_x[mask]
        gy = grid_y[mask]

    if image_rgb is not None and image_rgb.size:
        rgb = image_rgb[np.clip(gy, 0, image_rgb.shape[0] - 1), np.clip(gx, 0, image_rgb.shape[1] - 1), :]
        rgb = rgb[:, ::-1] if rgb.shape[1] == 3 else np.zeros((world_pts.shape[0], 3), dtype=np.uint8)
    else:
        rgb = np.full((world_pts.shape[0], 3), 180, dtype=np.uint8)
    return world_pts.astype(np.float64), rgb.astype(np.uint8)


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray, *, compressed: bool = False) -> None:
    ensure_dir(path.parent)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors.size:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=compressed, print_progress=False)
        if not ok:
            raise RuntimeError(f"Open3D failed to write {path}")
        return
    except Exception:
        # Fallback binary little-endian PLY writer; no ASCII fallback to avoid huge slow files.
        with path.open("wb") as handle:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {points.shape[0]}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            )
            handle.write(header.encode("ascii"))
            pts32 = points.astype(np.float32, copy=False)
            cols8 = colors.astype(np.uint8, copy=False)
            for p, c in zip(pts32, cols8):
                handle.write(struct.pack("<fffBBB", float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])))


def _append_with_cap(
    current_points: np.ndarray,
    current_colors: np.ndarray,
    new_points: np.ndarray,
    new_colors: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if new_points.shape[0] == 0:
        return current_points, current_colors
    if current_points.shape[0] == 0:
        pts = new_points
        cols = new_colors
    else:
        pts = np.vstack([current_points, new_points])
        cols = np.vstack([current_colors, new_colors])
    if pts.shape[0] > max_points:
        keep = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        cols = cols[keep]
    return pts, cols


def fuse_aligned_depths(
    cfg: Any,
    model: ColmapTextModel,
    alignment_results: list[AlignmentResult],
    image_dir: Path,
    output_ply: Path,
    fusion_plan_json: Path,
) -> dict[str, Any]:
    stride = int(cfg_get(cfg, "da3.fusion_stride", 8))
    min_depth = float(cfg_get(cfg, "da3.fusion_min_depth_m", 0.1))
    max_depth = float(cfg_get(cfg, "da3.fusion_max_depth_m", 80.0))
    max_points = int(cfg_get(cfg, "da3.fusion_max_points", 2_000_000))
    binary_ply = bool_value(cfg_get(cfg, "da3.fusion_binary_ply", True))
    compressed_ply = bool_value(cfg_get(cfg, "da3.fusion_compressed_ply", False))
    edge_filter_enabled = bool_value(cfg_get(cfg, "da3.fusion_edge_aware_filter", True))
    edge_threshold_m = float(cfg_get(cfg, "da3.fusion_edge_threshold_m", 0.5))
    edge_relative_threshold = float(cfg_get(cfg, "da3.fusion_edge_relative_threshold", 0.08))
    bbox = _parse_bbox(cfg_get(cfg, "da3.fusion_bounding_box", None))
    by_name = image_by_name(model)
    rng = np.random.default_rng(int(cfg_get(cfg, "project.random_seed", 42)))

    pts = np.zeros((0, 3), dtype=np.float64)
    cols = np.zeros((0, 3), dtype=np.uint8)
    per_image: list[dict[str, Any]] = []

    for r in alignment_results:
        if r.status not in {"ok", "warning"} or not r.aligned_depth_path.exists():
            continue
        pose = by_name.get(r.image_name)
        if pose is None:
            continue
        camera = model.cameras.get(pose.camera_id)
        if camera is None:
            continue
        depth = load_depth_map(r.aligned_depth_path)
        image_path = image_dir / r.image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        points, colors = _camera_backproject(
            depth,
            image,
            camera,
            pose,
            stride,
            min_depth,
            max_depth,
            edge_filter_enabled=edge_filter_enabled,
            edge_threshold_m=edge_threshold_m,
            edge_relative_threshold=edge_relative_threshold,
            bbox=bbox,
        )
        if points.shape[0] == 0:
            per_image.append({"image_name": r.image_name, "point_count": 0})
            continue
        pts, cols = _append_with_cap(pts, cols, points, colors, max_points=max_points, rng=rng)
        per_image.append({"image_name": r.image_name, "point_count": int(points.shape[0])})

    if binary_ply:
        write_binary_ply(output_ply, pts, cols, compressed=compressed_ply)
    else:
        # Kept only for debugging; binary is the project default.
        write_binary_ply(output_ply, pts, cols, compressed=False)

    plan = {
        "status": "ok",
        "output_ply": output_ply.as_posix(),
        "fused_point_count": int(pts.shape[0]),
        "source_image_count": len([p for p in per_image if p.get("point_count", 0) > 0]),
        "stride": stride,
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "max_points": max_points,
        "binary_ply": bool(binary_ply),
        "edge_aware_filter": bool(edge_filter_enabled),
        "edge_threshold_m": edge_threshold_m,
        "edge_relative_threshold": edge_relative_threshold,
        "fusion_bounding_box": None if bbox is None else {"min": bbox[0].tolist(), "max": bbox[1].tolist()},
        "per_image": per_image,
    }
    write_json_atomic(fusion_plan_json, plan)
    return plan
