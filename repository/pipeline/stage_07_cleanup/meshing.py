from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d

from .config_access import bool_value, cfg_get, list_value
from .io_utils import ensure_dir


@dataclass(frozen=True)
class MeshResult:
    status: str
    method: str
    mesh_path: str | None
    vertex_count: int
    triangle_count: int
    warnings: list[str]


def _ensure_normals(pcd: o3d.geometry.PointCloud, cfg: Any) -> None:
    if len(pcd.points) < 3:
        return
    if len(pcd.normals) == len(pcd.points):
        return
    radius = float(cfg_get(cfg, "cleanup.normal_radius_m", 0.10))
    max_nn = int(cfg_get(cfg, "cleanup.normal_max_nn", 40))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def _trim_poisson(mesh: o3d.geometry.TriangleMesh, densities: object, trim_quantile: float) -> o3d.geometry.TriangleMesh:
    if trim_quantile <= 0:
        return mesh
    dens = np.asarray(densities)
    if dens.size == 0:
        return mesh
    threshold = np.quantile(dens, trim_quantile)
    vertices_to_remove = dens < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def _plane_coefficients(planes: Iterable[Any] | None) -> np.ndarray:
    coeffs: list[np.ndarray] = []
    if planes is None:
        return np.empty((0, 4), dtype=np.float64)
    for plane in planes:
        value = getattr(plane, "coefficients", None)
        if value is None and isinstance(plane, dict):
            value = plane.get("coefficients")
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (4,):
            continue
        n = arr[:3]
        norm = float(np.linalg.norm(n))
        if norm <= 1e-12:
            continue
        coeffs.append(arr / norm)
    if not coeffs:
        return np.empty((0, 4), dtype=np.float64)
    return np.vstack(coeffs)


def _trim_mesh_to_planes(
    cfg: Any,
    mesh: o3d.geometry.TriangleMesh,
    planes: Iterable[Any] | None,
    warnings: list[str],
) -> o3d.geometry.TriangleMesh:
    if not bool_value(cfg_get(cfg, "cleanup.mesh_plane_trim_enabled", False)):
        return mesh
    coeffs = _plane_coefficients(planes)
    min_planes = int(cfg_get(cfg, "cleanup.mesh_plane_trim_min_planes", 2))
    if coeffs.shape[0] < min_planes:
        warnings.append(f"mesh_plane_trim_skipped_insufficient_planes:{coeffs.shape[0]}<{min_planes}")
        return mesh
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return mesh
    threshold = float(cfg_get(cfg, "cleanup.mesh_plane_trim_distance_m", 0.12))
    distances = np.abs(vertices @ coeffs[:, :3].T + coeffs[:, 3][None, :])
    keep = np.min(distances, axis=1) <= threshold
    keep_ratio = float(np.mean(keep)) if keep.size else 0.0
    min_keep_ratio = float(cfg_get(cfg, "cleanup.mesh_plane_trim_min_keep_ratio", 0.25))
    if keep_ratio < min_keep_ratio:
        warnings.append(f"mesh_plane_trim_skipped_keep_ratio_too_low:{keep_ratio:.3f}<{min_keep_ratio:.3f}")
        return mesh
    mesh.remove_vertices_by_mask(~keep)
    warnings.append(f"mesh_plane_trim_keep_ratio:{keep_ratio:.3f}")
    return mesh


def _poisson_mesh(cfg: Any, pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    warnings: list[str] = []
    depth = int(cfg_get(cfg, "cleanup.poisson_depth", 9))
    scale = float(cfg_get(cfg, "cleanup.poisson_scale", 1.1))
    linear_fit = bool_value(cfg_get(cfg, "cleanup.poisson_linear_fit", False))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit,
    )
    trim = float(cfg_get(cfg, "cleanup.poisson_trim_quantile", 0.02))
    mesh = _trim_poisson(mesh, densities, trim)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh, warnings


def _ball_pivoting_mesh(cfg: Any, pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    radii = [float(v) for v in list_value(cfg_get(cfg, "cleanup.ball_pivoting_radii_m", [0.04, 0.08, 0.16]), [])]
    if not radii:
        radii = [0.04, 0.08, 0.16]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh, []


def create_mesh(cfg: Any, pcd: o3d.geometry.PointCloud, mesh_path: Path, planes: Iterable[Any] | None = None) -> MeshResult:
    if not bool_value(cfg_get(cfg, "cleanup.mesh_enabled", True)):
        return MeshResult("disabled", "disabled", None, 0, 0, [])
    method = str(cfg_get(cfg, "cleanup.mesh_method", "ball_pivoting")).strip().lower()
    if method in {"disabled", "none", "off"}:
        return MeshResult("disabled", method, None, 0, 0, [])
    warnings: list[str] = []
    min_vertices = int(cfg_get(cfg, "cleanup.mesh_min_vertices", 1000))
    if len(pcd.points) < min_vertices:
        return MeshResult("skipped_insufficient_points", method, None, 0, 0, [f"point_count_below_mesh_min_vertices:{len(pcd.points)}<{min_vertices}"])
    try:
        _ensure_normals(pcd, cfg)
        if method == "poisson":
            mesh, method_warnings = _poisson_mesh(cfg, pcd)
        else:
            method = "ball_pivoting"
            mesh, method_warnings = _ball_pivoting_mesh(cfg, pcd)
        warnings.extend(method_warnings)
        mesh = _trim_mesh_to_planes(cfg, mesh, planes, warnings)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        vertex_count = len(mesh.vertices)
        triangle_count = len(mesh.triangles)
        if vertex_count <= 0 or triangle_count <= 0:
            return MeshResult("failed_empty_mesh", method, None, vertex_count, triangle_count, warnings)
        ensure_dir(mesh_path.parent)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
        return MeshResult("ok", method, mesh_path.as_posix(), vertex_count, triangle_count, warnings)
    except Exception as exc:
        if bool_value(cfg_get(cfg, "cleanup.fail_if_mesh_fails", False)):
            raise
        return MeshResult("failed", method, None, 0, 0, [f"mesh_exception:{type(exc).__name__}:{exc}"])
