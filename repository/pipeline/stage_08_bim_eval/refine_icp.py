"""ICP refinement for Stage 8 scan-to-BIM registration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_float, cfg_get, cfg_int, cfg_list
from .geometry_utils import require_open3d, safe_voxel_size_from_bounds
from .icp_robust_capability import RobustKernelDecision, build_robust_kernel, normalize_kernel_name


@dataclass(frozen=True)
class IcpRefinementResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_set_size: int
    method: str
    warnings: list[str]
    robust_loss: dict[str, Any] | None = None


def _downsample_and_normals(pcd: Any, voxel: float, normal_radius: float, normal_max_nn: int) -> Any:
    o3d = require_open3d()
    down = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    if len(down.points) == 0:
        down = pcd
    # We estimate normals even for point-to-point so a later point-to-plane stage
    # can reuse the same downsampled geometry safely.
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(normal_radius, voxel * 2.0, 1e-3), max_nn=normal_max_nn))
    return down


def _estimation(method: str, kernel: Any | None) -> Any:
    """Build an Open3D TransformationEstimation for the requested ICP variant.

    When ``kernel`` is non-None *and* the requested method is point-to-plane,
    the estimator is constructed with that robust kernel. For point-to-point
    Open3D's binding does not accept a robust kernel, so the kernel is silently
    ignored (the caller's :class:`RobustKernelDecision` already records the
    intent).
    """
    o3d = require_open3d()
    name = method.strip().lower()
    if name == "point_to_plane":
        if kernel is not None:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
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
    kernel: Any | None,
) -> Any:
    o3d = require_open3d()
    return o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr,
        np.asarray(initial_transform, dtype=float),
        _estimation(method, kernel),
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

    # O1: optional robust kernel for point-to-plane ICP. Default "none" preserves
    # historical behavior. Tukey is preferred when outlier fraction is bounded
    # (~<30%); Huber when it is not. Reference: Zhang & Singh, arXiv 2408.11809;
    # Zhang et al. arXiv 2007.07627. See docs/scientific_upgrades.md §2.
    requested_kernel = normalize_kernel_name(cfg_get(cfg, "bim.icp_robust_loss", "none"))
    kernel_k = cfg_float(cfg, "bim.icp_robust_loss_k_m", 0.05)
    kernel_obj, kernel_decision = build_robust_kernel(requested_kernel, kernel_k)

    source_down = _downsample_and_normals(scan_pcd, voxel, normal_radius, normal_max_nn)
    target_down = _downsample_and_normals(bim_pcd, voxel, normal_radius, normal_max_nn)
    stages = _icp_stage_list(cfg)
    current = np.asarray(initial_transform, dtype=float)
    warnings: list[str] = []
    last_result: Any = None
    completed: list[str] = []

    if kernel_decision.fallback_reason is not None:
        warnings.append(
            f"icp_robust_loss_fallback:requested={kernel_decision.requested},"
            f"applied={kernel_decision.applied},reason={kernel_decision.fallback_reason}"
        )
        logger.warning(
            "Stage 8 robust ICP loss requested=%s but unavailable (%s); falling back to non-robust point-to-plane.",
            kernel_decision.requested,
            kernel_decision.fallback_reason,
        )

    logger.info(
        "Running Stage 8 staged ICP: stages=%s voxel=%.6f max_corr=%.6f source_points=%d target_points=%d robust_loss=%s",
        stages,
        voxel,
        max_corr,
        len(source_down.points),
        len(target_down.points),
        kernel_decision.applied,
    )
    for stage in stages:
        if stage not in {"point_to_point", "point_to_plane"}:
            warnings.append(f"Unsupported ICP stage {stage!r}; skipping.")
            continue
        max_iter = p2l_iter if stage == "point_to_plane" else p2p_iter
        # The robust kernel only applies to point-to-plane; for point-to-point
        # Open3D's TransformationEstimationPointToPoint does not accept a kernel.
        kernel_for_stage = kernel_obj if stage == "point_to_plane" else None
        try:
            result = _run_icp_once(source_down, target_down, max_corr, current, stage, max_iter, rel_fitness, rel_rmse, kernel_for_stage)
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
        robust_loss={
            "requested": kernel_decision.requested,
            "applied": kernel_decision.applied,
            "k_m": kernel_decision.k_m,
            "binding_supports_kernel": kernel_decision.binding_supports_kernel,
            "fallback_reason": kernel_decision.fallback_reason,
        },
    )
