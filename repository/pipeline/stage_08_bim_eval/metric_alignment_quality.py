from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimilarityFit:
    status: str
    confidence: str
    scale: float
    rotation: list[list[float]]
    translation: list[float]
    matrix4x4: list[list[float]]
    residuals_m: list[float]
    rmse_m: float
    max_residual_m: float
    inlier_indices: list[int]
    failures: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={arr.shape}")
    return arr


def evaluate_anchor_geometry(points: Any, *, min_anchors: int = 3, min_rank: int = 2) -> dict[str, Any]:
    pts = _as_points(points)
    failures: list[str] = []
    warnings: list[str] = []

    if len(pts) < min_anchors:
        failures.append(f"insufficient_registration_anchors:{len(pts)}<{min_anchors}")

    centered = pts - pts.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-9))

    if len(pts) >= min_anchors and rank < min_rank:
        failures.append(f"degenerate_anchor_geometry_rank:{rank}<{min_rank}")

    min_pairwise_distance = None
    if len(pts) >= 2:
        distances = []
        for i, j in itertools.combinations(range(len(pts)), 2):
            distances.append(float(np.linalg.norm(pts[i] - pts[j])))
        min_pairwise_distance = min(distances)
        if min_pairwise_distance < 1e-6:
            failures.append("duplicate_or_near_duplicate_anchors")

    return {
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "anchor_count": int(len(pts)),
        "rank": rank,
        "min_pairwise_distance_m": min_pairwise_distance,
        "failures": failures,
        "warnings": warnings,
    }


def estimate_similarity_umeyama(scan_points: Any, bim_points: Any) -> SimilarityFit:
    """Estimate similarity transform mapping scan_points -> bim_points.

    Uses Umeyama/Procrustes with uniform scale only. This is the defensible
    metric alignment default; anisotropic affine is intentionally not used.
    """
    src = _as_points(scan_points)
    dst = _as_points(bim_points)

    if src.shape != dst.shape:
        raise ValueError(f"Point shape mismatch: scan={src.shape} bim={dst.shape}")

    geometry = evaluate_anchor_geometry(src)
    if not geometry["passed"]:
        return SimilarityFit(
            status="failed_anchor_geometry",
            confidence="low",
            scale=1.0,
            rotation=np.eye(3).tolist(),
            translation=[0.0, 0.0, 0.0],
            matrix4x4=np.eye(4).tolist(),
            residuals_m=[],
            rmse_m=float("inf"),
            max_residual_m=float("inf"),
            inlier_indices=[],
            failures=list(geometry["failures"]),
            warnings=list(geometry["warnings"]),
        )

    n = len(src)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    var_src = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if var_src <= 1e-12:
        return SimilarityFit(
            status="failed_zero_source_variance",
            confidence="low",
            scale=1.0,
            rotation=np.eye(3).tolist(),
            translation=[0.0, 0.0, 0.0],
            matrix4x4=np.eye(4).tolist(),
            residuals_m=[],
            rmse_m=float("inf"),
            max_residual_m=float("inf"),
            inlier_indices=[],
            failures=["zero_source_variance"],
            warnings=[],
        )

    cov = (dst_centered.T @ src_centered) / n
    u, s, vt = np.linalg.svd(cov)

    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0

    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / var_src)
    translation = dst_mean - scale * (rotation @ src_mean)

    transformed = (scale * (rotation @ src.T)).T + translation
    residuals = np.linalg.norm(transformed - dst, axis=1)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    max_residual = float(np.max(residuals))

    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    confidence = "high" if rmse <= 0.03 and max_residual <= 0.08 else "medium" if rmse <= 0.08 else "low"

    return SimilarityFit(
        status="ok",
        confidence=confidence,
        scale=scale,
        rotation=rotation.tolist(),
        translation=translation.tolist(),
        matrix4x4=matrix.tolist(),
        residuals_m=[float(x) for x in residuals],
        rmse_m=rmse,
        max_residual_m=max_residual,
        inlier_indices=list(range(n)),
        failures=[],
        warnings=[],
    )


def leave_one_out_residuals(scan_points: Any, bim_points: Any) -> dict[str, Any]:
    src = _as_points(scan_points)
    dst = _as_points(bim_points)

    if len(src) < 4:
        return {
            "status": "skipped_insufficient_anchors_for_leave_one_out",
            "residuals_m": [],
            "max_residual_m": None,
            "rmse_m": None,
            "failures": [],
            "warnings": ["leave_one_out_requires_at_least_4_anchors"],
        }

    residuals: list[float] = []
    failures: list[str] = []

    for held_out in range(len(src)):
        keep = [idx for idx in range(len(src)) if idx != held_out]
        fit = estimate_similarity_umeyama(src[keep], dst[keep])
        if fit.status != "ok":
            failures.extend(f"held_out_{held_out}:{failure}" for failure in fit.failures)
            continue

        matrix = np.asarray(fit.matrix4x4, dtype=float)
        point_h = np.array([src[held_out, 0], src[held_out, 1], src[held_out, 2], 1.0])
        predicted = (matrix @ point_h)[:3]
        residuals.append(float(np.linalg.norm(predicted - dst[held_out])))

    if failures:
        return {
            "status": "failed",
            "residuals_m": residuals,
            "max_residual_m": max(residuals) if residuals else None,
            "rmse_m": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
            "failures": failures,
            "warnings": [],
        }

    return {
        "status": "ok",
        "residuals_m": residuals,
        "max_residual_m": max(residuals) if residuals else None,
        "rmse_m": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
        "failures": [],
        "warnings": [],
    }


def robust_similarity_ransac(
    scan_points: Any,
    bim_points: Any,
    *,
    residual_threshold_m: float = 0.08,
    min_inliers: int = 3,
) -> SimilarityFit:
    src = _as_points(scan_points)
    dst = _as_points(bim_points)

    if src.shape != dst.shape:
        raise ValueError(f"Point shape mismatch: scan={src.shape} bim={dst.shape}")

    if len(src) < min_inliers:
        return SimilarityFit(
            status="failed_insufficient_anchors",
            confidence="low",
            scale=1.0,
            rotation=np.eye(3).tolist(),
            translation=[0.0, 0.0, 0.0],
            matrix4x4=np.eye(4).tolist(),
            residuals_m=[],
            rmse_m=float("inf"),
            max_residual_m=float("inf"),
            inlier_indices=[],
            failures=[f"insufficient_registration_anchors:{len(src)}<{min_inliers}"],
            warnings=[],
        )

    best_fit: SimilarityFit | None = None
    best_inliers: list[int] = []

    for combo in itertools.combinations(range(len(src)), 3):
        fit = estimate_similarity_umeyama(src[list(combo)], dst[list(combo)])
        if fit.status != "ok":
            continue

        matrix = np.asarray(fit.matrix4x4, dtype=float)
        src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)
        predicted = (matrix @ src_h.T).T[:, :3]
        residuals = np.linalg.norm(predicted - dst, axis=1)
        inliers = [int(idx) for idx, value in enumerate(residuals) if value <= residual_threshold_m]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_fit = fit
        elif len(inliers) == len(best_inliers) and best_fit is not None:
            if float(np.mean(residuals[inliers])) < best_fit.rmse_m if inliers else False:
                best_inliers = inliers
                best_fit = fit

    if best_fit is None or len(best_inliers) < min_inliers:
        return SimilarityFit(
            status="failed_no_consensus",
            confidence="low",
            scale=1.0,
            rotation=np.eye(3).tolist(),
            translation=[0.0, 0.0, 0.0],
            matrix4x4=np.eye(4).tolist(),
            residuals_m=[],
            rmse_m=float("inf"),
            max_residual_m=float("inf"),
            inlier_indices=[],
            failures=[f"insufficient_ransac_inliers:{len(best_inliers)}<{min_inliers}"],
            warnings=[],
        )

    refit = estimate_similarity_umeyama(src[best_inliers], dst[best_inliers])
    if refit.status != "ok":
        return refit

    matrix = np.asarray(refit.matrix4x4, dtype=float)
    src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    predicted = (matrix @ src_h.T).T[:, :3]
    residuals = np.linalg.norm(predicted - dst, axis=1)

    warnings: list[str] = []
    if len(best_inliers) < len(src):
        warnings.append(f"ransac_rejected_outliers:{len(src) - len(best_inliers)}")

    return SimilarityFit(
        status="ok",
        confidence=refit.confidence,
        scale=refit.scale,
        rotation=refit.rotation,
        translation=refit.translation,
        matrix4x4=refit.matrix4x4,
        residuals_m=[float(x) for x in residuals],
        rmse_m=float(np.sqrt(np.mean(np.square(residuals[best_inliers])))),
        max_residual_m=float(np.max(residuals[best_inliers])),
        inlier_indices=best_inliers,
        failures=[],
        warnings=warnings,
    )


def enrich_metric_alignment_report(report: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(report)
    failures: list[str] = []
    warnings: list[str] = []

    usable = report.get("usable_registration_anchor_count")
    try:
        usable_count = int(usable)
    except Exception:
        usable_count = None

    if usable_count is None:
        warnings.append("usable_registration_anchor_count_missing")
    elif usable_count < 3:
        failures.append(f"insufficient_registration_anchors:{usable_count}<3")

    quality_gate = report.get("quality_gate")
    if not isinstance(quality_gate, dict):
        quality_gate = {"passed": not failures, "failures": [], "warnings": [], "thresholds": {}}

    q_failures = list(quality_gate.get("failures", []) or [])
    q_warnings = list(quality_gate.get("warnings", []) or [])

    for failure in failures:
        if failure not in q_failures:
            q_failures.append(failure)

    for warning in warnings:
        if warning not in q_warnings:
            q_warnings.append(warning)

    quality_gate["failures"] = q_failures
    quality_gate["warnings"] = q_warnings
    quality_gate["passed"] = not q_failures

    enriched["quality_gate"] = quality_gate
    enriched["metric_alignment_quality"] = {
        "status": "pass" if quality_gate["passed"] else "fail",
        "failures": failures,
        "warnings": warnings,
    }

    if not quality_gate["passed"]:
        enriched["confidence"] = "low"

    return enriched


def enrich_metric_alignment_report_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    enriched = enrich_metric_alignment_report(data if isinstance(data, dict) else {})
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return enriched
