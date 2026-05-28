"""Open3D geometry helpers used by Stage 8."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # noqa: BLE001
    o3d = None  # type: ignore[assignment]
    _OPEN3D_IMPORT_ERROR = exc
else:
    _OPEN3D_IMPORT_ERROR = None


@dataclass(frozen=True)
class BoundsSummary:
    min_bound: list[float]
    max_bound: list[float]
    center: list[float]
    extent: list[float]
    diagonal: float


def require_open3d() -> Any:
    if o3d is None:
        raise RuntimeError(f"Open3D is required for Stage 8 but could not be imported: {_OPEN3D_IMPORT_ERROR}")
    return o3d


def bounds_summary(geometry: Any) -> BoundsSummary:
    min_bound = np.asarray(geometry.get_min_bound(), dtype=float)
    max_bound = np.asarray(geometry.get_max_bound(), dtype=float)
    extent = max_bound - min_bound
    center = (min_bound + max_bound) * 0.5
    return BoundsSummary(
        min_bound=[float(x) for x in min_bound],
        max_bound=[float(x) for x in max_bound],
        center=[float(x) for x in center],
        extent=[float(x) for x in extent],
        diagonal=float(np.linalg.norm(extent)),
    )


def point_count(geometry: Any) -> int:
    if hasattr(geometry, "points"):
        return int(len(geometry.points))
    if hasattr(geometry, "vertices"):
        return int(len(geometry.vertices))
    return 0


def triangle_count(geometry: Any) -> int:
    if hasattr(geometry, "triangles"):
        return int(len(geometry.triangles))
    return 0


def load_point_cloud(path: Path) -> Any:
    o3d_mod = require_open3d()
    pcd = o3d_mod.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")
    return pcd


def write_point_cloud(path: Path, pcd: Any, *, binary: bool = True) -> None:
    o3d_mod = require_open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d_mod.io.write_point_cloud(str(path), pcd, write_ascii=not binary):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def write_mesh(path: Path, mesh: Any, *, binary: bool = True) -> None:
    o3d_mod = require_open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d_mod.io.write_triangle_mesh(str(path), mesh, write_ascii=not binary):
        raise RuntimeError(f"Failed to write mesh: {path}")


def mesh_to_point_cloud(mesh: Any, target_points: int) -> Any:
    o3d_mod = require_open3d()
    if len(mesh.vertices) == 0:
        raise RuntimeError("Cannot sample empty BIM mesh")
    if len(mesh.triangles) > 0 and target_points > 0:
        try:
            return mesh.sample_points_uniformly(number_of_points=int(target_points), use_triangle_normal=True)
        except Exception:  # noqa: BLE001
            pass
    pcd = o3d_mod.geometry.PointCloud()
    pcd.points = mesh.vertices
    if len(mesh.vertex_colors) == len(mesh.vertices):
        pcd.colors = mesh.vertex_colors
    return pcd


def make_box_point_cloud(bounds_min: np.ndarray, bounds_max: np.ndarray, points_per_edge: int = 20) -> Any:
    o3d_mod = require_open3d()
    mn = np.asarray(bounds_min, dtype=float)
    mx = np.asarray(bounds_max, dtype=float)
    xs = np.linspace(mn[0], mx[0], max(2, points_per_edge))
    ys = np.linspace(mn[1], mx[1], max(2, points_per_edge))
    zs = np.linspace(mn[2], mx[2], max(2, points_per_edge))
    pts: list[list[float]] = []
    for x in xs:
        for y in ys:
            pts.append([float(x), float(y), float(mn[2])])
            pts.append([float(x), float(y), float(mx[2])])
    for x in xs:
        for z in zs:
            pts.append([float(x), float(mn[1]), float(z)])
            pts.append([float(x), float(mx[1]), float(z)])
    for y in ys:
        for z in zs:
            pts.append([float(mn[0]), float(y), float(z)])
            pts.append([float(mx[0]), float(y), float(z)])
    pcd = o3d_mod.geometry.PointCloud()
    pcd.points = o3d_mod.utility.Vector3dVector(np.asarray(pts, dtype=float))
    return pcd


def identity_matrix() -> list[list[float]]:
    return np.eye(4, dtype=float).tolist()


def safe_voxel_size_from_bounds(source: Any, requested: float, fallback_fraction: float = 0.01) -> float:
    if requested > 0:
        return float(requested)
    b = bounds_summary(source)
    if not math.isfinite(b.diagonal) or b.diagonal <= 0:
        return 0.05
    return max(0.005, b.diagonal * fallback_fraction)


def seed_open3d_rng(seed: int) -> bool:
    """Best-effort seeding of Open3D's global RNG.

    Open3D 0.17+ exposes ``open3d.utility.random.seed(int)`` which seeds the
    same RNG used by ``registration_ransac_based_on_feature_matching`` and
    ``sample_points_uniformly``. This is the only available knob; the
    ``RANSACConvergenceCriteria`` struct does not expose a seed in current
    bindings (verified against Open3D 0.19.0).

    Returns True if seeding was applied, False if Open3D is unavailable or the
    ``utility.random.seed`` symbol is missing on the running build.
    """
    if o3d is None:
        return False
    try:
        random_mod = getattr(o3d.utility, "random", None)
        seed_fn = getattr(random_mod, "seed", None) if random_mod is not None else None
        if seed_fn is None:
            return False
        seed_fn(int(seed))
        return True
    except Exception:
        return False
