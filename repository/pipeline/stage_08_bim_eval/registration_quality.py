"""Registration quality checks for Stage 8."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_float, cfg_int
from .geometry_utils import require_open3d


def nearest_neighbor_summary(source_pcd: Any, target_pcd: Any, *, sample_limit: int = 200_000) -> dict[str, Any]:
    o3d = require_open3d()
    source_points = np.asarray(source_pcd.points, dtype=np.float64)
    target_points = np.asarray(target_pcd.points, dtype=np.float64)
    if len(source_points) == 0 or len(target_points) == 0:
        return {"count": 0, "mean_m": None, "median_m": None, "p90_m": None, "p95_m": None, "max_m": None}
    if len(source_points) > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(source_points), size=sample_limit, replace=False)
        source_points = source_points[idx]
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    tree = o3d.geometry.KDTreeFlann(target)
    distances: list[float] = []
    for point in source_points:
        _k, _idx, d2 = tree.search_knn_vector_3d(point, 1)
        if d2:
            distances.append(float(np.sqrt(d2[0])))
    if not distances:
        return {"count": 0, "mean_m": None, "median_m": None, "p90_m": None, "p95_m": None, "max_m": None}
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.quantile(arr, 0.90)),
        "p95_m": float(np.quantile(arr, 0.95)),
        "max_m": float(np.max(arr)),
    }


def evaluate_registration_quality(cfg: Any, report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    icp = report.get("icp", {})
    scale_factor = float(report.get("coarse_registration", {}).get("scale_factor", 1.0) or 1.0)
    fitness = icp.get("fitness")
    rmse = icp.get("inlier_rmse")
    min_fitness = cfg_float(cfg, "bim.quality_min_icp_fitness", 0.05)
    max_rmse = cfg_float(cfg, "bim.quality_max_icp_rmse_m", 0.25)
    scale_warning = cfg_float(cfg, "bim.quality_scale_warning_ratio", 0.10)
    if fitness is None or float(fitness) < min_fitness:
        msg = f"ICP fitness {fitness} is below threshold {min_fitness}."
        if cfg_bool(cfg, "bim.fail_on_low_registration_quality", False):
            errors.append(msg)
        else:
            warnings.append(msg)
    if rmse is None or float(rmse) > max_rmse:
        msg = f"ICP RMSE {rmse} is above threshold {max_rmse} m."
        if cfg_bool(cfg, "bim.fail_on_low_registration_quality", False):
            errors.append(msg)
        else:
            warnings.append(msg)
    if abs(scale_factor - 1.0) > scale_warning:
        warnings.append(
            f"Initial scale factor {scale_factor:.6f} differs from 1.0 by more than {scale_warning:.3f}; verify metric scale before Stage 9."
        )
    status = "pass" if not errors else "fail"
    return {"status": status, "passed": not errors, "warnings": warnings, "errors": errors}
