"""Coarse registration strategies for Stage 8.

The default initializer is deliberately conservative. It centers the selected
scan on the BIM reference without guessing monocular scale from global bounding
boxes. A scan can cover only a room, a facade, a slab zone, or an open exterior
area while the IFC may contain a whole building; bbox scale estimation would
therefore be a catastrophic false prior. Use a known scale/GCP/DA3 metric scale
when a scale correction is genuinely needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_float, cfg_get
from .geometry_utils import bounds_summary, require_open3d, safe_voxel_size_from_bounds
from .metric_initial_transform import load_metric_initial_transform


@dataclass(frozen=True)
class CoarseRegistrationResult:
    transformation: np.ndarray
    method: str
    scale_factor: float
    fitness: float | None
    inlier_rmse: float | None
    warnings: list[str]


def _clone_pcd(pcd: Any) -> Any:
    if hasattr(pcd, "clone"):
        return pcd.clone()
    return pcd.select_by_index(list(range(len(pcd.points))))


def _unsafe_bbox_scale(source: Any, target: Any, min_scale: float, max_scale: float) -> tuple[float, list[str]]:
    warnings: list[str] = [
        "UNSAFE: bbox-based initial scale was explicitly enabled. This is not recommended for partial scans, open-site scans, or IFC files larger than the captured area.",
    ]
    sb = bounds_summary(source)
    tb = bounds_summary(target)
    source_extent = np.asarray(sb.extent, dtype=float)
    target_extent = np.asarray(tb.extent, dtype=float)
    valid = (source_extent > 1e-9) & (target_extent > 1e-9)
    if not np.any(valid):
        warnings.append("Could not estimate bbox scale because one or both bounds are degenerate; using 1.0.")
        return 1.0, warnings
    ratios = target_extent[valid] / source_extent[valid]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        warnings.append("Could not estimate a finite bbox scale; using 1.0.")
        return 1.0, warnings
    clipped = max(float(min_scale), min(float(max_scale), scale))
    if abs(clipped - scale) > 1e-9:
        warnings.append(f"BBox scale {scale:.6f} clipped to configured range: {clipped:.6f}.")
    return clipped, warnings


def _center_scale_transform(
    source: Any,
    target: Any,
    scale_strategy: str,
    known_scale: float,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, float, list[str]]:
    sb = bounds_summary(source)
    tb = bounds_summary(target)
    source_center = np.asarray(sb.center, dtype=float)
    target_center = np.asarray(tb.center, dtype=float)
    warnings: list[str] = []

    strategy = (scale_strategy or "fixed_1").strip().lower()
    scale = 1.0
    method_warning = ""
    if strategy in {"fixed", "fixed_1", "none", "metric", "da3_metric", "gcp_locked"}:
        scale = 1.0
    elif strategy in {"known", "known_scale", "manual"}:
        scale = float(known_scale)
        if not np.isfinite(scale) or scale <= 0:
            warnings.append(f"Invalid known initial scale {known_scale!r}; using 1.0.")
            scale = 1.0
        scale = max(float(min_scale), min(float(max_scale), scale))
        method_warning = "Known/manual initial scale applied. Verify it came from GCPs, DA3 metric alignment, or a measured control distance."
    elif strategy in {"bbox_unsafe", "unsafe_bbox"}:
        scale, bbox_warnings = _unsafe_bbox_scale(source, target, min_scale, max_scale)
        warnings.extend(bbox_warnings)
    else:
        warnings.append(f"Unknown initial_scale_strategy={scale_strategy!r}; using fixed metric scale 1.0.")
        scale = 1.0

    if method_warning:
        warnings.append(method_warning)

    T = np.eye(4, dtype=float)
    T[:3, :3] *= scale
    T[:3, 3] = target_center - scale * source_center
    return T, scale, warnings


def _preprocess_for_fpfh(pcd: Any, voxel_size: float) -> tuple[Any, Any]:
    o3d = require_open3d()
    down = pcd.voxel_down_sample(voxel_size)
    if len(down.points) < 10:
        down = pcd
    radius_normal = max(voxel_size * 2.0, 1e-3)
    radius_feature = max(voxel_size * 5.0, 1e-3)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return down, feature


def _try_fpfh_ransac(source: Any, target: Any, initial: np.ndarray, voxel_size: float, distance_multiplier: float) -> tuple[np.ndarray, float, float] | None:
    o3d = require_open3d()
    source_init = _clone_pcd(source)
    source_init.transform(initial.copy())
    try:
        source_down, source_feat = _preprocess_for_fpfh(source_init, voxel_size)
        target_down, target_feat = _preprocess_for_fpfh(target, voxel_size)
    except Exception:
        return None
    if len(source_down.points) < 30 or len(target_down.points) < 30:
        return None
    distance = max(voxel_size * distance_multiplier, voxel_size)
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feat,
            target_feat,
            True,
            distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
        )
    except Exception:
        return None
    if result.fitness <= 0:
        return None
    return result.transformation @ initial, float(result.fitness), float(result.inlier_rmse)


def coarse_register(cfg: Any, scan_pcd: Any, bim_pcd: Any, logger: logging.Logger) -> CoarseRegistrationResult:

    metric_initial = load_metric_initial_transform(cfg, logger=logger)
    if metric_initial is not None:
        return CoarseRegistrationResult(
            transformation=metric_initial.matrix4x4,
            method=metric_initial.method,
            scale_factor=metric_initial.scale,
            fitness=0.0,
            inlier_rmse=0.0,
            warnings=metric_initial.warnings,
        )
    legacy_bbox_scale = cfg_bool(cfg, "bim.estimate_initial_scale_from_bbox", False)
    strategy = str(cfg_get(cfg, "bim.initial_scale_strategy", "fixed_1")).strip().lower()
    if legacy_bbox_scale and strategy in {"fixed", "fixed_1", "none", "metric", "da3_metric", "gcp_locked"}:
        # Backward-compatible safety: old configs may still contain the legacy flag.
        logger.warning(
            "bim.estimate_initial_scale_from_bbox=true is ignored unless bim.initial_scale_strategy=bbox_unsafe. "
            "This prevents catastrophic scale drift on partial/open scans."
        )
    known_scale = cfg_float(cfg, "bim.known_initial_scale", 1.0)
    min_scale = cfg_float(cfg, "bim.min_initial_scale", 0.01)
    max_scale = cfg_float(cfg, "bim.max_initial_scale", 100.0)
    initial, scale, warnings = _center_scale_transform(scan_pcd, bim_pcd, strategy, known_scale, min_scale, max_scale)
    method = "center_rigid" if abs(scale - 1.0) <= 1e-9 else f"center_{strategy}"
    fitness: float | None = None
    rmse: float | None = None

    if cfg_bool(cfg, "bim.coarse_fpfh_enabled", True):
        voxel = cfg_float(cfg, "bim.coarse_voxel_size_m", 0.0)
        voxel = safe_voxel_size_from_bounds(bim_pcd, voxel, fallback_fraction=0.02)
        multiplier = cfg_float(cfg, "bim.coarse_distance_multiplier", 2.5)
        logger.info("Trying Stage 8 FPFH coarse registration with voxel=%.6f", voxel)
        ransac = _try_fpfh_ransac(scan_pcd, bim_pcd, initial, voxel, multiplier)
        if ransac is not None:
            candidate_T, candidate_fitness, candidate_rmse = ransac
            min_fitness = cfg_float(cfg, "bim.coarse_min_fitness_accept", 0.05)
            if candidate_fitness >= min_fitness:
                initial = candidate_T
                method = f"{method}+fpfh_ransac"
                fitness = candidate_fitness
                rmse = candidate_rmse
            else:
                warnings.append(
                    f"FPFH coarse registration fitness {candidate_fitness:.6f} below acceptance threshold {min_fitness:.6f}; using center initialization."
                )
        else:
            warnings.append("FPFH coarse registration did not produce a usable transform; using center initialization.")

    return CoarseRegistrationResult(
        transformation=np.asarray(initial, dtype=float),
        method=method,
        scale_factor=float(scale),
        fitness=fitness,
        inlier_rmse=rmse,
        warnings=warnings,
    )
