from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .config_access import bool_value, cfg_get
from .io_utils import ensure_dir, write_json_atomic


@dataclass(frozen=True)
class PlaneRecord:
    plane_id: str
    label: str
    point_count: int
    point_ratio: float
    coefficients: list[float]
    normal: list[float]
    offset: float
    centroid: list[float]
    bounds_min: list[float]
    bounds_max: list[float]
    approx_area_m2: float
    cloud_path: str
    extraction_mode: str = "blind_ransac"


def _normalise_plane(model: list[float]) -> tuple[np.ndarray, float, list[float]]:
    coeff = np.array(model, dtype=np.float64)
    n = coeff[:3]
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0, [0.0, 0.0, 1.0, 0.0]
    coeff = coeff / norm
    n = coeff[:3]
    d = float(coeff[3])
    return n, d, coeff.astype(float).tolist()


def _up_vector(axis: str) -> np.ndarray:
    axis = axis.lower().strip()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def _classify_plane(normal: np.ndarray, centroid: np.ndarray, cfg: Any) -> str:
    up = _up_vector(str(cfg_get(cfg, "cleanup.classify_up_axis", "z")))
    dot = float(np.dot(normal, up))
    abs_dot = abs(dot)
    horizontal = float(cfg_get(cfg, "cleanup.floor_ceiling_normal_dot_min", 0.90))
    vertical = float(cfg_get(cfg, "cleanup.wall_vertical_abs_dot_max", 0.25))
    if abs_dot >= horizontal:
        return "floor_or_ceiling" if dot >= 0 else "ceiling_or_floor"
    if abs_dot <= vertical:
        return "wall"
    return "inclined_or_unknown"


def _bounds_and_area(points: np.ndarray) -> tuple[list[float], list[float], float]:
    if points.size == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    extents = np.maximum(mx - mn, 0.0)
    two = np.sort(extents)[-2:]
    return mn.astype(float).tolist(), mx.astype(float).tolist(), float(two[0] * two[1])


def _ensure_normals(pcd: o3d.geometry.PointCloud, cfg: Any) -> None:
    if len(pcd.points) < 3:
        return
    if len(pcd.normals) == len(pcd.points):
        return
    radius = float(cfg_get(cfg, "cleanup.normal_radius_m", 0.10))
    max_nn = int(cfg_get(cfg, "cleanup.normal_max_nn", 40))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def _candidate_indices_by_normals(remaining: o3d.geometry.PointCloud, cfg: Any, mode: str) -> np.ndarray:
    if len(remaining.points) == 0:
        return np.array([], dtype=np.int64)
    _ensure_normals(remaining, cfg)
    normals = np.asarray(remaining.normals)
    if normals.shape[0] != len(remaining.points):
        return np.arange(len(remaining.points), dtype=np.int64)
    up = _up_vector(str(cfg_get(cfg, "cleanup.classify_up_axis", "z")))
    abs_dot = np.abs(normals @ up)
    horizontal_min = float(cfg_get(cfg, "cleanup.normal_guided_horizontal_dot_min", 0.85))
    wall_max = float(cfg_get(cfg, "cleanup.normal_guided_wall_abs_dot_max", 0.35))
    if mode == "horizontal":
        return np.where(abs_dot >= horizontal_min)[0].astype(np.int64)
    if mode == "vertical":
        return np.where(abs_dot <= wall_max)[0].astype(np.int64)
    return np.arange(len(remaining.points), dtype=np.int64)


def _segment_candidate_plane(
    remaining: o3d.geometry.PointCloud,
    candidate_indices: np.ndarray,
    distance: float,
    ransac_n: int,
    iterations: int,
) -> tuple[list[float], list[int]] | None:
    if candidate_indices.size < ransac_n:
        return None
    candidate = remaining.select_by_index(candidate_indices.astype(int).tolist())
    if len(candidate.points) < ransac_n:
        return None
    try:
        model, local_inliers = candidate.segment_plane(
            distance_threshold=distance,
            ransac_n=ransac_n,
            num_iterations=iterations,
        )
    except Exception:
        return None
    if not local_inliers:
        return None
    global_inliers = candidate_indices[np.asarray(local_inliers, dtype=np.int64)]
    return list(model), global_inliers.astype(int).tolist()


def _append_plane_record(
    records: list[PlaneRecord],
    cfg: Any,
    plane_cloud: o3d.geometry.PointCloud,
    model: list[float],
    initial_count: int,
    plane_clouds_dir: Path,
    extraction_mode: str,
) -> None:
    plane_points = np.asarray(plane_cloud.points)
    n, d, coeffs = _normalise_plane(list(model))
    centroid = plane_points.mean(axis=0)
    mn, mx, area = _bounds_and_area(plane_points)
    label = _classify_plane(n, centroid, cfg)
    plane_id = f"plane_{len(records) + 1:03d}"
    cloud_path = plane_clouds_dir / f"{plane_id}_{label}.ply"
    o3d.io.write_point_cloud(str(cloud_path), plane_cloud, write_ascii=False)
    records.append(
        PlaneRecord(
            plane_id=plane_id,
            label=label,
            point_count=len(plane_cloud.points),
            point_ratio=float(len(plane_cloud.points) / initial_count) if initial_count else 0.0,
            coefficients=coeffs,
            normal=n.astype(float).tolist(),
            offset=d,
            centroid=centroid.astype(float).tolist(),
            bounds_min=mn,
            bounds_max=mx,
            approx_area_m2=area,
            cloud_path=cloud_path.as_posix(),
            extraction_mode=extraction_mode,
        )
    )


def _extract_guided_planes(
    cfg: Any,
    working: o3d.geometry.PointCloud,
    plane_clouds_dir: Path,
    max_planes: int,
    distance: float,
    ransac_n: int,
    iterations: int,
    min_points: int,
    min_ratio: float,
    min_remaining: int,
    min_remaining_ratio: float,
) -> list[PlaneRecord]:
    initial_count = len(working.points)
    records: list[PlaneRecord] = []
    remaining = working
    phase_order = ["horizontal", "vertical"]
    if bool_value(cfg_get(cfg, "cleanup.normal_guided_allow_residual_planes", True)):
        phase_order.append("residual")
    max_per_phase = {
        "horizontal": int(cfg_get(cfg, "cleanup.normal_guided_max_horizontal_planes", max_planes)),
        "vertical": int(cfg_get(cfg, "cleanup.normal_guided_max_vertical_planes", max_planes)),
        "residual": int(cfg_get(cfg, "cleanup.normal_guided_max_residual_planes", max_planes)),
    }

    for phase in phase_order:
        phase_count = 0
        while len(records) < max_planes and phase_count < max_per_phase[phase]:
            n_remaining = len(remaining.points)
            if n_remaining < max(ransac_n, min_remaining):
                return records
            if initial_count > 0 and n_remaining / initial_count < min_remaining_ratio:
                return records
            candidates = _candidate_indices_by_normals(remaining, cfg, phase if phase in {"horizontal", "vertical"} else "residual")
            if candidates.size < max(ransac_n, min_points):
                break
            result = _segment_candidate_plane(remaining, candidates, distance, ransac_n, iterations)
            if result is None:
                break
            model, inliers = result
            if len(inliers) < min_points and (initial_count == 0 or len(inliers) / initial_count < min_ratio):
                break
            plane_cloud = remaining.select_by_index(inliers)
            if len(plane_cloud.points) <= 0:
                break
            _append_plane_record(records, cfg, plane_cloud, model, initial_count, plane_clouds_dir, f"normal_guided_{phase}")
            remaining = remaining.select_by_index(inliers, invert=True)
            phase_count += 1
    return records


def _extract_blind_planes(
    cfg: Any,
    working: o3d.geometry.PointCloud,
    plane_clouds_dir: Path,
    max_planes: int,
    distance: float,
    ransac_n: int,
    iterations: int,
    min_points: int,
    min_ratio: float,
    min_remaining: int,
    min_remaining_ratio: float,
) -> list[PlaneRecord]:
    initial_count = len(working.points)
    records: list[PlaneRecord] = []
    remaining = working
    for _idx in range(max_planes):
        n_remaining = len(remaining.points)
        if n_remaining < max(ransac_n, min_remaining):
            break
        if initial_count > 0 and n_remaining / initial_count < min_remaining_ratio:
            break
        try:
            model, inliers = remaining.segment_plane(
                distance_threshold=distance,
                ransac_n=ransac_n,
                num_iterations=iterations,
            )
        except Exception:
            break
        if not inliers:
            break
        if len(inliers) < min_points and (initial_count == 0 or len(inliers) / initial_count < min_ratio):
            break
        plane_cloud = remaining.select_by_index(inliers)
        if len(plane_cloud.points) <= 0:
            break
        _append_plane_record(records, cfg, plane_cloud, list(model), initial_count, plane_clouds_dir, "blind_ransac")
        remaining = remaining.select_by_index(inliers, invert=True)
    return records


def extract_planes(cfg: Any, pcd: o3d.geometry.PointCloud, plane_clouds_dir: Path, planes_json: Path) -> list[PlaneRecord]:
    ensure_dir(plane_clouds_dir)
    if not bool_value(cfg_get(cfg, "cleanup.plane_extraction_enabled", True)):
        write_json_atomic(planes_json, {"planes": [], "status": "disabled"})
        return []

    working = pcd
    plane_voxel = float(cfg_get(cfg, "cleanup.plane_voxel_size_m", 0.035))
    if plane_voxel > 0 and len(working.points) > 0:
        working = working.voxel_down_sample(plane_voxel)
    _ensure_normals(working, cfg)

    initial_count = len(working.points)
    max_planes = int(cfg_get(cfg, "cleanup.max_planes", 12))
    distance = float(cfg_get(cfg, "cleanup.plane_distance_threshold_m", 0.035))
    ransac_n = int(cfg_get(cfg, "cleanup.plane_ransac_n", 3))
    iterations = int(cfg_get(cfg, "cleanup.plane_num_iterations", 1500))
    min_points = int(cfg_get(cfg, "cleanup.min_plane_points", 800))
    min_ratio = float(cfg_get(cfg, "cleanup.min_plane_ratio", 0.015))
    min_remaining = int(cfg_get(cfg, "cleanup.min_remaining_points", 2000))
    min_remaining_ratio = float(cfg_get(cfg, "cleanup.min_remaining_ratio", 0.15))

    if bool_value(cfg_get(cfg, "cleanup.normal_guided_plane_extraction", True)):
        records = _extract_guided_planes(
            cfg,
            working,
            plane_clouds_dir,
            max_planes,
            distance,
            ransac_n,
            iterations,
            min_points,
            min_ratio,
            min_remaining,
            min_remaining_ratio,
        )
    else:
        records = _extract_blind_planes(
            cfg,
            working,
            plane_clouds_dir,
            max_planes,
            distance,
            ransac_n,
            iterations,
            min_points,
            min_ratio,
            min_remaining,
            min_remaining_ratio,
        )

    payload = {
        "status": "ok",
        "input_point_count": initial_count,
        "plane_count": len(records),
        "normal_guided": bool_value(cfg_get(cfg, "cleanup.normal_guided_plane_extraction", True)),
        "planes": [asdict(r) for r in records],
    }
    write_json_atomic(planes_json, payload)
    return records
