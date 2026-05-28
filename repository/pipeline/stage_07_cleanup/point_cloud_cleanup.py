from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .config_access import bool_value, cfg_get, float_list_or_none
from .io_utils import ensure_dir


class Stage7PointCloudError(RuntimeError):
    """Raised for point-cloud cleanup failures."""


@dataclass(frozen=True)
class CleanupResult:
    input_count: int
    finite_count: int
    downsampled_count: int
    cleaned_count: int
    removed_count: int
    removed_ratio: float
    input_bounds_min: list[float]
    input_bounds_max: list[float]
    cleaned_bounds_min: list[float]
    cleaned_bounds_max: list[float]
    downsampled_path: Path
    cleaned_path: Path
    warnings: list[str]


def _bounds(pcd: o3d.geometry.PointCloud) -> tuple[list[float], list[float]]:
    if len(pcd.points) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    pts = np.asarray(pcd.points)
    return pts.min(axis=0).astype(float).tolist(), pts.max(axis=0).astype(float).tolist()


def _has_colors(pcd: o3d.geometry.PointCloud) -> bool:
    return len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0


def _subset_random(pcd: o3d.geometry.PointCloud, max_points: int, seed: int) -> o3d.geometry.PointCloud:
    """Last-resort cap after spatial voxelization has already made density uniform."""

    n = len(pcd.points)
    if max_points <= 0 or n <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return pcd.select_by_index(idx.tolist())


def _remove_non_finite_compat(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    try:
        pcd.remove_non_finite_points()
        return pcd
    except TypeError:
        pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        return pcd


def _crop_if_configured(pcd: o3d.geometry.PointCloud, cfg: Any) -> o3d.geometry.PointCloud:
    bbox = float_list_or_none(cfg_get(cfg, "cleanup.crop_bounding_box", None))
    if bbox is None:
        return pcd
    if len(bbox) != 6:
        raise Stage7PointCloudError("cleanup.crop_bounding_box must be [min_x,min_y,min_z,max_x,max_y,max_z].")
    margin = float(cfg_get(cfg, "cleanup.crop_margin_m", 0.0))
    min_bound = np.array(bbox[:3], dtype=np.float64) - margin
    max_bound = np.array(bbox[3:], dtype=np.float64) + margin
    if np.any(max_bound <= min_bound):
        raise Stage7PointCloudError("cleanup.crop_bounding_box has invalid min/max bounds.")
    return pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))


def _rgb_to_hsv_vectorized(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float colors in [0, 1] to HSV with H in degrees and S/V in [0, 1]."""

    rgb = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nonzero = diff > 1e-12
    mask = nonzero & (mx == r)
    hue[mask] = (60.0 * ((g[mask] - b[mask]) / diff[mask]) + 360.0) % 360.0
    mask = nonzero & (mx == g)
    hue[mask] = 60.0 * ((b[mask] - r[mask]) / diff[mask] + 2.0)
    mask = nonzero & (mx == b)
    hue[mask] = 60.0 * ((r[mask] - g[mask]) / diff[mask] + 4.0)
    sat = np.zeros_like(mx)
    sat[mx > 1e-12] = diff[mx > 1e-12] / mx[mx > 1e-12]
    return np.column_stack([hue, sat, mx])


def _range_mask(values: np.ndarray, lo: float, hi: float, *, circular_hue: bool = False) -> np.ndarray:
    if not circular_hue or lo <= hi:
        return (values >= lo) & (values <= hi)
    return (values >= lo) | (values <= hi)


def _semantic_filter_should_run(cfg: Any, semantic_context: dict[str, Any] | None) -> bool:
    if not bool_value(cfg_get(cfg, "cleanup.semantic_color_filter_enabled", False)):
        return False
    if not bool_value(cfg_get(cfg, "cleanup.semantics_enabled", True)):
        return False
    mode = str(cfg_get(cfg, "cleanup.semantic_color_filter_activation", "if_yolo_transients")).lower().strip()
    if mode in {"always", "true", "on"}:
        return True
    if mode in {"never", "false", "off"}:
        return False
    if not semantic_context:
        return False
    yolo = semantic_context.get("yolo", {}) if isinstance(semantic_context, dict) else {}
    try:
        return int(yolo.get("transient_frame_count", 0)) >= int(cfg_get(cfg, "cleanup.semantic_color_filter_min_transient_frames", 1))
    except Exception:
        return False


def _remove_semantic_high_visibility_colors(
    pcd: o3d.geometry.PointCloud,
    cfg: Any,
    semantic_context: dict[str, Any] | None,
    warnings: list[str],
    logger: logging.Logger,
) -> o3d.geometry.PointCloud:
    """Conservative HSV filter for transient high-visibility colors when YOLO context indicates dynamic objects.

    This is intentionally optional and capped. It is not a semantic segmentation replacement; it removes only
    strongly saturated safety-orange/yellow/green points that commonly correspond to workers/equipment in site videos.
    """

    if not _semantic_filter_should_run(cfg, semantic_context):
        return pcd
    if not _has_colors(pcd):
        warnings.append("semantic_color_filter_skipped_no_colors")
        return pcd

    colors = np.asarray(pcd.colors)
    hsv = _rgb_to_hsv_vectorized(colors)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    sat_min = float(cfg_get(cfg, "cleanup.semantic_color_filter_saturation_min", 0.45))
    value_min = float(cfg_get(cfg, "cleanup.semantic_color_filter_value_min", 0.35))
    ranges = cfg_get(
        cfg,
        "cleanup.semantic_color_filter_hsv_ranges",
        [
            [18.0, 48.0],   # orange/yellow high-vis PPE and machinery
            [50.0, 85.0],   # yellow-green high-vis vests
        ],
    )
    remove = np.zeros(len(colors), dtype=bool)
    for item in ranges or []:
        try:
            lo, hi = float(item[0]), float(item[1])
        except Exception:
            continue
        remove |= _range_mask(h, lo % 360.0, hi % 360.0, circular_hue=True)
    remove &= s >= sat_min
    remove &= v >= value_min

    max_ratio = float(cfg_get(cfg, "cleanup.semantic_color_filter_max_removed_ratio", 0.12))
    removed_ratio = float(np.mean(remove)) if remove.size else 0.0
    if removed_ratio <= 0:
        warnings.append("semantic_color_filter_removed_ratio:0.000")
        return pcd
    if removed_ratio > max_ratio:
        warnings.append(f"semantic_color_filter_skipped_ratio_above_cap:{removed_ratio:.3f}>{max_ratio:.3f}")
        return pcd
    keep_indices = np.where(~remove)[0]
    if keep_indices.size < 3:
        warnings.append("semantic_color_filter_skipped_would_remove_all_points")
        return pcd
    logger.info("Semantic HSV filtering removed %.2f%% of colored points", removed_ratio * 100.0)
    warnings.append(f"semantic_color_filter_removed_ratio:{removed_ratio:.3f}")
    return pcd.select_by_index(keep_indices.tolist())


def _dynamic_voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    cfg: Any,
    max_processing: int,
    warnings: list[str],
    logger: logging.Logger,
) -> tuple[o3d.geometry.PointCloud, float]:
    base_voxel = float(cfg_get(cfg, "cleanup.voxel_size_m", 0.025))
    if base_voxel <= 0:
        return pcd, base_voxel
    voxel = base_voxel
    max_iterations = int(cfg_get(cfg, "cleanup.dynamic_voxel_max_iterations", 6))
    enabled = bool_value(cfg_get(cfg, "cleanup.dynamic_voxel_enabled", True))
    exponent = float(cfg_get(cfg, "cleanup.dynamic_voxel_growth_exponent", 0.5))
    margin = float(cfg_get(cfg, "cleanup.dynamic_voxel_growth_margin", 1.05))

    down = pcd.voxel_down_sample(voxel_size=voxel)
    if not enabled or max_processing <= 0:
        return down, voxel

    for _ in range(max_iterations):
        count = len(down.points)
        if count <= max_processing or count <= 0:
            return down, voxel
        factor = max(margin, (count / max_processing) ** max(0.05, exponent) * margin)
        voxel *= factor
        logger.warning("Voxel downsampling kept %d points above max_processing=%d; increasing voxel to %.5fm", count, max_processing, voxel)
        warnings.append(f"dynamic_voxel_increased_to:{voxel:.5f}")
        down = pcd.voxel_down_sample(voxel_size=voxel)
    return down, voxel


def _estimate_normals(pcd: o3d.geometry.PointCloud, cfg: Any) -> None:
    if not bool_value(cfg_get(cfg, "cleanup.estimate_normals", True)) or len(pcd.points) < 3:
        return
    radius = float(cfg_get(cfg, "cleanup.normal_radius_m", 0.10))
    max_nn = int(cfg_get(cfg, "cleanup.normal_max_nn", 40))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    strategy = str(cfg_get(cfg, "cleanup.normal_orientation_strategy", "none")).lower().strip()
    # Consistent tangent-plane orientation is useful for watertight object scans, but can be unstable in open indoor
    # construction scenes. Keep it opt-in rather than default.
    if strategy == "consistent_tangent_plane":
        k = int(cfg_get(cfg, "cleanup.orient_normals_consistent_tangent_plane_k", 0))
        if k > 0 and len(pcd.points) > k:
            try:
                pcd.orient_normals_consistent_tangent_plane(k)
            except Exception:
                pass
    elif strategy == "towards_centroid":
        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if pts.size and normals.size:
            centroid = pts.mean(axis=0)
            vectors_to_center = centroid[None, :] - pts
            flip = np.sum(normals * vectors_to_center, axis=1) < 0
            normals[flip] *= -1.0
            pcd.normals = o3d.utility.Vector3dVector(normals)


def clean_point_cloud(
    cfg: Any,
    input_path: Path,
    downsampled_path: Path,
    cleaned_path: Path,
    logger: logging.Logger,
    semantic_context: dict[str, Any] | None = None,
) -> tuple[o3d.geometry.PointCloud, CleanupResult]:
    warnings: list[str] = []
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        raise Stage7PointCloudError(f"Input point cloud is empty or unreadable: {input_path}")

    input_count = len(pcd.points)
    input_min, input_max = _bounds(pcd)
    max_input = int(cfg_get(cfg, "cleanup.max_input_points", 5_000_000))
    if input_count > max_input:
        warnings.append(f"input_point_count_exceeds_max_input_points:{input_count}>{max_input}")

    if bool_value(cfg_get(cfg, "cleanup.remove_non_finite", True)):
        pcd = _remove_non_finite_compat(pcd)
    finite_count = len(pcd.points)

    if bool_value(cfg_get(cfg, "cleanup.remove_duplicated_points", True)):
        try:
            pcd.remove_duplicated_points()
        except Exception:
            warnings.append("remove_duplicated_points_failed")

    pcd = _crop_if_configured(pcd, cfg)
    pcd = _remove_semantic_high_visibility_colors(pcd, cfg, semantic_context, warnings, logger)

    max_processing = int(cfg_get(cfg, "cleanup.max_processing_points", 1_500_000))
    seed = int(cfg_get(cfg, "cleanup.random_seed", 42))

    down, used_voxel = _dynamic_voxel_downsample(pcd, cfg, max_processing, warnings, logger)
    downsampled_count = len(down.points)
    if max_processing > 0 and downsampled_count > max_processing:
        warnings.append(f"post_voxel_random_subsample_applied:{downsampled_count}>{max_processing}")
        down = _subset_random(down, max_processing, seed)
        downsampled_count = len(down.points)
    warnings.append(f"voxel_size_used_m:{used_voxel:.5f}")

    ensure_dir(downsampled_path.parent)
    o3d.io.write_point_cloud(str(downsampled_path), down, write_ascii=not bool_value(cfg_get(cfg, "cleanup.write_binary_ply", True)))

    cleaned = down
    if bool_value(cfg_get(cfg, "cleanup.statistical_enabled", True)) and len(cleaned.points) > 10:
        nb = int(cfg_get(cfg, "cleanup.statistical_nb_neighbors", 24))
        std = float(cfg_get(cfg, "cleanup.statistical_std_ratio", 2.0))
        filtered, _ind = cleaned.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        if len(filtered.points) > 0:
            cleaned = filtered
        else:
            warnings.append("statistical_outlier_filter_would_remove_all_points")

    if bool_value(cfg_get(cfg, "cleanup.radius_enabled", True)) and len(cleaned.points) > 10:
        nbp = int(cfg_get(cfg, "cleanup.radius_nb_points", 8))
        radius = float(cfg_get(cfg, "cleanup.radius_m", 0.10))
        filtered, _ind = cleaned.remove_radius_outlier(nb_points=nbp, radius=radius)
        if len(filtered.points) > 0:
            cleaned = filtered
        else:
            warnings.append("radius_outlier_filter_would_remove_all_points")

    _estimate_normals(cleaned, cfg)
    cleaned_count = len(cleaned.points)
    if cleaned_count <= 0:
        raise Stage7PointCloudError("Cleanup removed all points.")

    ensure_dir(cleaned_path.parent)
    o3d.io.write_point_cloud(str(cleaned_path), cleaned, write_ascii=not bool_value(cfg_get(cfg, "cleanup.write_binary_ply", True)))

    cleaned_min, cleaned_max = _bounds(cleaned)
    removed_count = max(0, input_count - cleaned_count)
    removed_ratio = removed_count / input_count if input_count else 0.0
    logger.info("Cleaned point cloud: input=%d finite=%d downsampled=%d cleaned=%d", input_count, finite_count, downsampled_count, cleaned_count)

    return cleaned, CleanupResult(
        input_count=input_count,
        finite_count=finite_count,
        downsampled_count=downsampled_count,
        cleaned_count=cleaned_count,
        removed_count=removed_count,
        removed_ratio=removed_ratio,
        input_bounds_min=input_min,
        input_bounds_max=input_max,
        cleaned_bounds_min=cleaned_min,
        cleaned_bounds_max=cleaned_max,
        downsampled_path=downsampled_path,
        cleaned_path=cleaned_path,
        warnings=warnings,
    )
