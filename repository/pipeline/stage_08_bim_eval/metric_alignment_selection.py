from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: list[list[float]]
    translation: list[float]
    matrix4x4: list[list[float]]
    residuals_m: list[float]
    residual_mean_m: float
    residual_median_m: float
    residual_max_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_attr_or_key(obj: Any, names: list[str]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _xyz(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _anchor_id(anchor: Any, index: int) -> str:
    raw = _get_attr_or_key(anchor, ["anchor_id", "id", "name"])
    return str(raw) if raw not in (None, "") else f"anchor_{index:03d}"


def extract_anchor_correspondences(anchors: list[Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids: list[str] = []
    scan_points: list[np.ndarray] = []
    bim_points: list[np.ndarray] = []

    for idx, anchor in enumerate(anchors):
        scan = _xyz(
            _get_attr_or_key(
                anchor,
                [
                    "scan_xyz",
                    "scan_point",
                    "source_xyz",
                    "source_point",
                    "asbuilt_xyz",
                    "picked_scan_xyz",
                ],
            )
        )
        bim = _xyz(
            _get_attr_or_key(
                anchor,
                [
                    "bim_xyz",
                    "bim_point",
                    "target_xyz",
                    "target_point",
                    "design_xyz",
                ],
            )
        )

        if scan is None or bim is None:
            continue

        ids.append(_anchor_id(anchor, idx))
        scan_points.append(scan)
        bim_points.append(bim)

    if not scan_points:
        return [], np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    return ids, np.vstack(scan_points), np.vstack(bim_points)


def _geometry_rank(points: np.ndarray) -> int:
    if len(points) < 2:
        return 0
    centered = points - points.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered, tol=1e-9))


def estimate_umeyama_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate target ~= scale * R @ source + t."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be Nx3 arrays with matching shape")
    if len(source) < 3:
        raise ValueError("at least 3 correspondences are required")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)

    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 1e-12:
        raise ValueError("source anchors are degenerate")

    cov = (tgt_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(cov)

    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0

    rotation = u @ np.diag(d) @ vt
    scale = float(np.sum(singular_values * d) / src_var)
    translation = tgt_mean - scale * (rotation @ src_mean)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated scale is invalid")

    return scale, rotation, translation


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (scale * (rotation @ points.T)).T + translation


def _matrix4x4(scale: float, rotation: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = scale * rotation
    mat[:3, 3] = translation
    return mat.tolist()


def _transform_summary(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> SimilarityTransform:
    predicted = apply_sim3(source, scale, rotation, translation)
    residuals = np.linalg.norm(predicted - target, axis=1)

    return SimilarityTransform(
        scale=float(scale),
        rotation=rotation.tolist(),
        translation=translation.tolist(),
        matrix4x4=_matrix4x4(scale, rotation, translation),
        residuals_m=[float(x) for x in residuals],
        residual_mean_m=float(np.mean(residuals)) if len(residuals) else math.inf,
        residual_median_m=float(np.median(residuals)) if len(residuals) else math.inf,
        residual_max_m=float(np.max(residuals)) if len(residuals) else math.inf,
    )


def estimate_full_sim3(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    scale, rotation, translation = estimate_umeyama_sim3(source, target)
    return _transform_summary(source, target, scale, rotation, translation)


def leave_one_out_residuals(source: np.ndarray, target: np.ndarray) -> list[float]:
    if len(source) < 4:
        return []

    residuals: list[float] = []
    for holdout in range(len(source)):
        train = [idx for idx in range(len(source)) if idx != holdout]
        try:
            scale, rotation, translation = estimate_umeyama_sim3(source[train], target[train])
            pred = apply_sim3(source[[holdout]], scale, rotation, translation)[0]
            residuals.append(float(np.linalg.norm(pred - target[holdout])))
        except Exception:
            residuals.append(math.inf)
    return residuals


def ransac_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float,
    iterations: int = 200,
    random_seed: int = 42,
) -> dict[str, Any]:
    n = len(source)

    if n < 4:
        return {
            "status": "skipped_insufficient_anchors",
            "inlier_indices": [],
            "outlier_indices": list(range(n)),
            "reason": f"{n}<4",
        }

    if _geometry_rank(source) < 2 or _geometry_rank(target) < 2:
        return {
            "status": "skipped_degenerate_anchor_geometry",
            "inlier_indices": [],
            "outlier_indices": list(range(n)),
            "source_rank": _geometry_rank(source),
            "target_rank": _geometry_rank(target),
        }

    rng = random.Random(random_seed)

    if n <= 8:
        samples = list(itertools.combinations(range(n), 3))
    else:
        samples = [tuple(rng.sample(range(n), 3)) for _ in range(iterations)]

    best: dict[str, Any] | None = None

    for sample in samples:
        try:
            scale, rotation, translation = estimate_umeyama_sim3(source[list(sample)], target[list(sample)])
            pred = apply_sim3(source, scale, rotation, translation)
            residuals = np.linalg.norm(pred - target, axis=1)
            inliers = [int(i) for i, r in enumerate(residuals) if float(r) <= threshold_m]
        except Exception:
            continue

        if len(inliers) < 3:
            continue

        score = (
            len(inliers),
            -float(np.median(residuals[inliers])) if inliers else -math.inf,
            -float(np.max(residuals[inliers])) if inliers else -math.inf,
        )

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "sample_indices": list(sample),
                "inlier_indices": inliers,
                "all_residuals_m": [float(x) for x in residuals],
            }

    if best is None:
        return {
            "status": "failed_no_consensus",
            "inlier_indices": [],
            "outlier_indices": list(range(n)),
            "threshold_m": threshold_m,
        }

    inliers = best["inlier_indices"]
    try:
        refined = estimate_full_sim3(source[inliers], target[inliers])
    except Exception as exc:
        return {
            "status": "failed_refine",
            "inlier_indices": inliers,
            "outlier_indices": [i for i in range(n) if i not in inliers],
            "error": str(exc),
            "threshold_m": threshold_m,
        }

    return {
        "status": "ok",
        "threshold_m": float(threshold_m),
        "sample_indices": best["sample_indices"],
        "inlier_indices": inliers,
        "outlier_indices": [int(i) for i in range(n) if i not in inliers],
        "inlier_count": len(inliers),
        "outlier_count": n - len(inliers),
        "transform": refined.to_dict(),
    }


def _threshold_from_report(report: dict[str, Any]) -> float:
    thresholds = (
        report.get("quality_gate", {}).get("thresholds", {})
        if isinstance(report.get("quality_gate"), dict)
        else {}
    )
    raw = thresholds.get("residual_fail_m") or thresholds.get("anchor_residual_fail_m") or 0.15
    try:
        return float(raw)
    except Exception:
        return 0.15


def attach_metric_alignment_selection(report: dict[str, Any], anchors: list[Any]) -> dict[str, Any]:
    ids, source, target = extract_anchor_correspondences(list(anchors or []))

    threshold_m = _threshold_from_report(report)

    selection: dict[str, Any] = {
        "stage": "stage_08_metric_alignment_selection",
        "status": "not_run",
        "input_anchor_count": len(ids),
        "anchor_ids": ids,
        "threshold_m": threshold_m,
        "source_rank": _geometry_rank(source),
        "target_rank": _geometry_rank(target),
    }

    if len(ids) < 3:
        selection.update(
            {
                "status": "skipped_insufficient_anchors",
                "selected_solution": "none",
                "reason": f"{len(ids)}<3",
            }
        )
        report["metric_alignment_selection"] = selection
        return report

    try:
        full = estimate_full_sim3(source, target)
        selection["full_sim3"] = full.to_dict()
        selection["leave_one_out_residuals_m"] = leave_one_out_residuals(source, target)
    except Exception as exc:
        selection["full_sim3_error"] = str(exc)

    ransac = ransac_sim3(source, target, threshold_m=threshold_m)
    selection["ransac_sim3"] = ransac

    if ransac.get("status") == "ok":
        recommended = ransac["transform"]
        selection["status"] = "ok"
        selection["selected_solution"] = "ransac_sim3"
        selection["recommended_transform"] = recommended
        selection["recommended_inlier_anchor_ids"] = [ids[i] for i in ransac.get("inlier_indices", [])]
        selection["recommended_outlier_anchor_ids"] = [ids[i] for i in ransac.get("outlier_indices", [])]

        # Stable aliases for downstream Stage 8 initial transform loaders.
        report["selected_metric_solution"] = "ransac_sim3"
        report["recommended_transform_scan_to_bim"] = recommended
        report["transform_scan_to_bim_ransac"] = recommended
        report["transform_scan_to_bim"] = recommended
        report["matrix4x4"] = recommended.get("matrix4x4")

    elif "full_sim3" in selection:
        selection["status"] = "ok_full_sim3_only"
        selection["selected_solution"] = "full_sim3"
        selection["recommended_transform"] = selection["full_sim3"]

        report.setdefault("selected_metric_solution", "full_sim3")
        report.setdefault("recommended_transform_scan_to_bim", selection["full_sim3"])
        report.setdefault("matrix4x4", selection["full_sim3"].get("matrix4x4"))

    else:
        selection["status"] = ransac.get("status", "failed")
        selection["selected_solution"] = "none"

    report["metric_alignment_selection"] = selection
    return report
