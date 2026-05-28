from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisualAnchorQualityThresholds:
    min_observations: int = 2
    min_ray_angle_deg: float = 1.0
    warn_ray_angle_deg: float = 3.0
    reprojection_warn_px: float = 4.0
    reprojection_fail_px: float = 10.0
    max_depth_m: float = 500.0


@dataclass(frozen=True)
class VisualAnchorQualityResult:
    anchor_id: str
    status: str
    accepted: bool
    observation_count: int
    min_ray_angle_deg: float | None
    mean_reprojection_error_px: float | None
    max_reprojection_error_px: float | None
    failures: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_xyz(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    if isinstance(value, dict):
        for keys in (
            ("x", "y", "z"),
            ("x_m", "y_m", "z_m"),
            ("scan_x_m", "scan_y_m", "scan_z_m"),
        ):
            if all(k in value for k in keys):
                try:
                    arr = np.asarray([value[k] for k in keys], dtype=float)
                    return arr if arr.shape == (3,) and np.all(np.isfinite(arr)) else None
                except Exception:
                    return None

    for attr in ("xyz", "point_xyz", "scan_xyz", "triangulated_xyz", "point"):
        if hasattr(value, attr):
            try:
                arr = np.asarray(getattr(value, attr), dtype=float).reshape(-1)
                return arr if arr.size == 3 and np.all(np.isfinite(arr)) else None
            except Exception:
                return None

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return arr if arr.size == 3 and np.all(np.isfinite(arr)) else None
    except Exception:
        return None


def _get_anchor_id(value: Any, fallback: str = "unknown") -> str:
    if isinstance(value, dict):
        for key in ("anchor_id", "id", "name"):
            if value.get(key):
                return str(value[key])

    for attr in ("anchor_id", "id", "name"):
        if hasattr(value, attr):
            raw = getattr(value, attr)
            if raw:
                return str(raw)

    return fallback


def _get_observation_count(value: Any, fallback: int = 0) -> int:
    if isinstance(value, dict):
        for key in ("observation_count", "observations", "view_count", "num_observations"):
            if key in value:
                raw = value[key]
                if isinstance(raw, list):
                    return len(raw)
                try:
                    return int(raw)
                except Exception:
                    pass

    for attr in ("observation_count", "observations", "view_count", "num_observations"):
        if hasattr(value, attr):
            raw = getattr(value, attr)
            if isinstance(raw, list):
                return len(raw)
            try:
                return int(raw)
            except Exception:
                pass

    return fallback


def angle_between_rays_deg(ray_a: np.ndarray, ray_b: np.ndarray) -> float:
    a = np.asarray(ray_a, dtype=float).reshape(3)
    b = np.asarray(ray_b, dtype=float).reshape(3)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def camera_center_from_world_to_camera(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return camera center C for COLMAP-style x_cam = R * x_world + t."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    return -r.T @ t


def min_ray_angle_from_camera_centers(point_xyz: np.ndarray, camera_centers: list[np.ndarray]) -> float | None:
    if len(camera_centers) < 2:
        return None

    point = np.asarray(point_xyz, dtype=float).reshape(3)
    rays = [point - np.asarray(center, dtype=float).reshape(3) for center in camera_centers]

    angles: list[float] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            angles.append(angle_between_rays_deg(rays[i], rays[j]))

    return float(min(angles)) if angles else None


def reprojection_errors_px(
    point_xyz: np.ndarray,
    observed_pixels: list[tuple[float, float]],
    projected_pixels: list[tuple[float, float]],
) -> list[float]:
    del point_xyz

    errors: list[float] = []
    for observed, projected in zip(observed_pixels, projected_pixels):
        ou, ov = observed
        pu, pv = projected
        errors.append(float(math.hypot(float(ou) - float(pu), float(ov) - float(pv))))

    return errors


def evaluate_visual_anchor_quality(
    anchor: Any,
    *,
    observation_count: int | None = None,
    min_ray_angle_deg: float | None = None,
    reprojection_errors: list[float] | None = None,
    thresholds: VisualAnchorQualityThresholds | None = None,
) -> VisualAnchorQualityResult:
    thresholds = thresholds or VisualAnchorQualityThresholds()

    anchor_id = _get_anchor_id(anchor)
    point = _as_xyz(anchor)
    count = observation_count if observation_count is not None else _get_observation_count(anchor)

    failures: list[str] = []
    warnings: list[str] = []

    if point is None:
        failures.append("invalid_or_missing_3d_point")
    else:
        depth_norm = float(np.linalg.norm(point))
        if not np.isfinite(depth_norm):
            failures.append("non_finite_3d_point")
        elif depth_norm > thresholds.max_depth_m:
            warnings.append(f"large_coordinate_norm:{depth_norm:.3f}>{thresholds.max_depth_m:.3f}")

    if count < thresholds.min_observations:
        failures.append(f"insufficient_observations:{count}<{thresholds.min_observations}")

    if min_ray_angle_deg is None:
        warnings.append("ray_angle_not_available")
    elif min_ray_angle_deg < thresholds.min_ray_angle_deg:
        failures.append(
            f"ray_angle_too_small:{min_ray_angle_deg:.3f}<{thresholds.min_ray_angle_deg:.3f}"
        )
    elif min_ray_angle_deg < thresholds.warn_ray_angle_deg:
        warnings.append(
            f"ray_angle_low:{min_ray_angle_deg:.3f}<{thresholds.warn_ray_angle_deg:.3f}"
        )

    mean_reprojection = None
    max_reprojection = None

    if reprojection_errors:
        arr = np.asarray(reprojection_errors, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size:
            mean_reprojection = float(np.mean(arr))
            max_reprojection = float(np.max(arr))

            if max_reprojection > thresholds.reprojection_fail_px:
                failures.append(
                    f"reprojection_error_too_high:{max_reprojection:.3f}>{thresholds.reprojection_fail_px:.3f}"
                )
            elif max_reprojection > thresholds.reprojection_warn_px:
                warnings.append(
                    f"reprojection_error_high:{max_reprojection:.3f}>{thresholds.reprojection_warn_px:.3f}"
                )
        else:
            warnings.append("reprojection_errors_not_finite")
    else:
        warnings.append("reprojection_error_not_available")

    status = "accepted" if not failures else "rejected"

    return VisualAnchorQualityResult(
        anchor_id=anchor_id,
        status=status,
        accepted=not failures,
        observation_count=count,
        min_ray_angle_deg=min_ray_angle_deg,
        mean_reprojection_error_px=mean_reprojection,
        max_reprojection_error_px=max_reprojection,
        failures=failures,
        warnings=warnings,
    )


def summarize_visual_anchor_quality(results: list[VisualAnchorQualityResult]) -> dict[str, Any]:
    total = len(results)
    accepted = sum(1 for item in results if item.accepted)
    rejected = total - accepted

    return {
        "stage": "stage_08_visual_anchor_quality",
        "status": "ok" if rejected == 0 else "has_rejections",
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "accepted_anchor_ids": [item.anchor_id for item in results if item.accepted],
        "rejected_anchor_ids": [item.anchor_id for item in results if not item.accepted],
        "results": [item.to_dict() for item in results],
    }
