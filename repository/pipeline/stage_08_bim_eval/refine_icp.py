"""ICP refinement for Stage 8 scan-to-BIM registration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_float, cfg_get, cfg_int, cfg_list
from .geometry_utils import require_open3d, safe_voxel_size_from_bounds


@dataclass(frozen=True)
class IcpRefinementResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_set_size: int
    method: str
    warnings: list[str]


def _downsample_and_normals(pcd: Any, voxel: float, normal_radius: float, normal_max_nn: int) -> Any:
    o3d = require_open3d()
    down = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    if len(down.points) == 0:
        down = pcd
    # We estimate normals even for point-to-point so a later point-to-plane stage
    # can reuse the same downsampled geometry safely.
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(normal_radius, voxel * 2.0, 1e-3), max_nn=normal_max_nn))
    return down


def _estimation(method: str) -> Any:
    o3d = require_open3d()
    name = method.strip().lower()
    if name == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return o3d.pipelines.registration.TransformationEstimationPointToPoint(False)


def _run_icp_once(
    source_down: Any,
    target_down: Any,
    max_corr: float,
    initial_transform: np.ndarray,
    method: str,
    max_iter: int,
    rel_fitness: float,
    rel_rmse: float,
) -> Any:
    o3d = require_open3d()
    return o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr,
        np.asarray(initial_transform, dtype=float),
        _estimation(method),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=rel_fitness,
            relative_rmse=rel_rmse,
            max_iteration=max_iter,
        ),
    )


def _icp_stage_list(cfg: Any) -> list[str]:
    configured = cfg_list(cfg, "bim.icp_stages", [])
    stages = [str(x).strip().lower() for x in configured if str(x).strip()]
    if stages:
        return stages
    legacy = str(cfg_get(cfg, "bim.icp_estimation", "point_to_point_then_plane")).strip().lower()
    if legacy in {"point_to_point_then_plane", "staged", "multi_stage"}:
        return ["point_to_point", "point_to_plane"]
    if legacy == "point_to_plane":
        # Safer default: use point-to-point first to avoid singular point-to-plane
        # behavior when the coarse pose/scale is still approximate.
        return ["point_to_point", "point_to_plane"]
    return ["point_to_point"]


def refine_icp(cfg: Any, scan_pcd: Any, bim_pcd: Any, initial_transform: np.ndarray, logger: logging.Logger) -> IcpRefinementResult:
    enabled = cfg_bool(cfg, "bim.icp_enabled", True)
    if not enabled:
        return IcpRefinementResult(
            transformation=np.asarray(initial_transform, dtype=float),
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondence_set_size=0,
            method="disabled",
            warnings=["ICP refinement disabled by config."],
        )
    voxel = cfg_float(cfg, "bim.icp_voxel_size_m", 0.0)
    voxel = safe_voxel_size_from_bounds(bim_pcd, voxel, fallback_fraction=0.01)
    max_corr = cfg_float(cfg, "bim.icp_max_corr_distance_m", 0.08)
    normal_radius = cfg_float(cfg, "bim.icp_normal_radius_m", max(voxel * 3.0, 0.05))
    normal_max_nn = cfg_int(cfg, "bim.icp_normal_max_nn", 40)
    max_iter_default = cfg_int(cfg, "bim.icp_max_iteration", 80)
    p2p_iter = cfg_int(cfg, "bim.icp_point_to_point_max_iteration", max_iter_default)
    p2l_iter = cfg_int(cfg, "bim.icp_point_to_plane_max_iteration", min(max_iter_default, 30))
    rel_fitness = cfg_float(cfg, "bim.icp_relative_fitness", 1.0e-6)
    rel_rmse = cfg_float(cfg, "bim.icp_relative_rmse", 1.0e-6)
    source_down = _downsample_and_normals(scan_pcd, voxel, normal_radius, normal_max_nn)
    target_down = _downsample_and_normals(bim_pcd, voxel, normal_radius, normal_max_nn)
    stages = _icp_stage_list(cfg)
    current = np.asarray(initial_transform, dtype=float)
    warnings: list[str] = []
    last_result: Any = None
    completed: list[str] = []

    logger.info(
        "Running Stage 8 staged ICP: stages=%s voxel=%.6f max_corr=%.6f source_points=%d target_points=%d",
        stages,
        voxel,
        max_corr,
        len(source_down.points),
        len(target_down.points),
    )
    for stage in stages:
        if stage not in {"point_to_point", "point_to_plane"}:
            warnings.append(f"Unsupported ICP stage {stage!r}; skipping.")
            continue
        max_iter = p2l_iter if stage == "point_to_plane" else p2p_iter
        try:
            result = _run_icp_once(source_down, target_down, max_corr, current, stage, max_iter, rel_fitness, rel_rmse)
        except Exception as exc:  # noqa: BLE001
            if stage == "point_to_plane" and last_result is not None:
                warnings.append(f"Point-to-plane ICP failed after point-to-point; keeping previous transform. Error: {exc}")
                continue
            raise RuntimeError(f"Open3D ICP failed during {stage}: {exc}") from exc
        current = np.asarray(result.transformation, dtype=float)
        last_result = result
        completed.append(stage)

    if last_result is None:
        raise RuntimeError("No ICP stage completed successfully.")
    return IcpRefinementResult(
        transformation=np.asarray(last_result.transformation, dtype=float),
        fitness=float(last_result.fitness),
        inlier_rmse=float(last_result.inlier_rmse),
        correspondence_set_size=int(len(last_result.correspondence_set)),
        method="+".join(completed),
        warnings=warnings,
    )
