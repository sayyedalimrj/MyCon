"""Research-only, explicit-meter scan-to-BIM geometry metrics.

Use Stage 8 aligned scan and BIM reference in the SAME metric frame.
No Stage 8 output paths or existing reports are changed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _points(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite Nx3 array")
    return a


def _sample(a: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n < 1:
        raise ValueError("max_points must be >=1")
    if len(a) <= n:
        return a
    rng = np.random.default_rng(seed)
    return a[np.sort(rng.choice(len(a), size=n, replace=False))]


def _nn(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for research geometry metrics") from exc
    dist, _ = cKDTree(target).query(src, k=1, workers=1)
    return np.asarray(dist, dtype=np.float64)


def compare_clouds(
    scan_aligned_m: Any,
    bim_reference_m: Any,
    *, threshold_m: float = 0.05,
    max_points: int = 100000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute point-sampled, NOT surface/visibility-normalized metrics.

    Accuracy is scan->BIM; completeness is BIM->scan. The latter is not a
    construction-completion rate, particularly for partially captured sites.
    Chamfer L1 = mean(d_scan_bim) + mean(d_bim_scan);
    Chamfer L2 squared = mean(d_scan_bim^2) + mean(d_bim_scan^2).
    """
    if not math.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("threshold_m must be positive and finite")
    scan = _points(scan_aligned_m, "scan_aligned_m")
    bim = _points(bim_reference_m, "bim_reference_m")
    scan_s = _sample(scan, max_points, random_seed)
    bim_s = _sample(bim, max_points, random_seed + 1)
    s_to_b = _nn(scan_s, bim_s)
    b_to_s = _nn(bim_s, scan_s)
    precision = float(np.mean(s_to_b <= threshold_m))
    recall = float(np.mean(b_to_s <= threshold_m))
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "unit": "meter",
        "threshold_m": float(threshold_m),
        "sampled_scan_points": int(len(scan_s)),
        "sampled_bim_points": int(len(bim_s)),
        "total_scan_points": int(len(scan)),
        "total_bim_points": int(len(bim)),
        "random_seed": int(random_seed),
        "accuracy_mean_m": float(np.mean(s_to_b)),
        "accuracy_median_m": float(np.median(s_to_b)),
        "accuracy_p95_m": float(np.quantile(s_to_b, 0.95)),
        "completeness_mean_m": float(np.mean(b_to_s)),
        "completeness_median_m": float(np.median(b_to_s)),
        "completeness_p95_m": float(np.quantile(b_to_s, 0.95)),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f_score_at_threshold": float(f_score),
        "chamfer_l1_sum_m": float(np.mean(s_to_b) + np.mean(b_to_s)),
        "chamfer_l2_squared_sum_m2": float(np.mean(s_to_b ** 2) + np.mean(b_to_s ** 2)),
        "interpretation": "sampled point-to-point diagnostic; not independent site ground truth",
    }


def alignment_residuals(scan_anchor_m: Any, bim_anchor_m: Any) -> dict[str, Any]:
    """Matched, already-aligned anchors; not an ICP fit/self-fit score."""
    scan = _points(scan_anchor_m, "scan_anchor_m")
    bim = _points(bim_anchor_m, "bim_anchor_m")
    if scan.shape != bim.shape or len(scan) < 3:
        raise ValueError("At least 3 paired independent anchors are required")
    diff = np.linalg.norm(scan - bim, axis=1)
    return {
        "count": int(len(diff)),
        "registration_error_rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "registration_error_median_m": float(np.median(diff)),
        "registration_error_max_m": float(np.max(diff)),
        "provenance": "independent_corresponding_anchors_after_alignment",
    }


def gravity_error_deg(scan_up: Any, bim_up: Any) -> float:
    a = np.asarray(scan_up, dtype=float)
    b = np.asarray(bim_up, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Gravity/up vectors must each be finite XYZ triples")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("Gravity vectors must be nonzero")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_file_comparison(
    scan_ply: Path, bim_ply: Path, output_json: Path, *,
    threshold_m: float = 0.05, max_points: int = 100000,
    seed: int = 42, anchors_json: Path | None = None,
    gravity_scan: tuple[float, float, float] | None = None,
    gravity_bim: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Strictly create-only report; never touches Stage 8 legacy artifacts."""
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite geometry metrics: {output_json}")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Install open3d to read PLY point clouds") from exc
    scan = o3d.io.read_point_cloud(str(scan_ply.resolve()))
    bim = o3d.io.read_point_cloud(str(bim_ply.resolve()))
    result = compare_clouds(scan.points, bim.points, threshold_m=threshold_m,
                            max_points=max_points, random_seed=seed)
    result.update({
        "schema": "mycon.geometry.metrics.v1",
        "scan_aligned_ply": str(scan_ply.resolve()),
        "bim_reference_ply": str(bim_ply.resolve()),
        "input_sha256": {"scan": _hash(scan_ply), "bim": _hash(bim_ply)},
        "metric_frame": "explicitly_asserted_by_operator_both_inputs_in_bim_world_m",
        "ground_truth_verified": False,
        "independent_holdout": False,
        "registration_error": None,
        "gravity_alignment_error_deg": None,
        "created_at_unix": time.time(),
    })
    if anchors_json is not None:
        anchors = json.loads(anchors_json.read_text(encoding="utf-8"))
        result["registration_error"] = alignment_residuals(anchors["scan_aligned_m"], anchors["bim_m"])
        result["input_sha256"]["anchors"] = _hash(anchors_json)
    if (gravity_scan is None) != (gravity_bim is None):
        raise ValueError("Both gravity vectors must be supplied together")
    if gravity_scan is not None:
        result["gravity_alignment_error_deg"] = gravity_error_deg(gravity_scan, gravity_bim)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Non-destructive aligned scan-to-BIM geometry benchmark")
    p.add_argument("--scan-aligned", type=Path, required=True)
    p.add_argument("--bim-reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="New report path under runs/geom-*/geometry/")
    p.add_argument("--threshold-m", type=float, default=0.05)
    p.add_argument("--max-points", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchors-json", type=Path)
    p.add_argument("--gravity-scan", type=float, nargs=3)
    p.add_argument("--gravity-bim", type=float, nargs=3)
    p.add_argument("--assert-bim-world-meters", action="store_true",
                   help="Required: operator verified both inputs share a metric BIM frame")
    a = p.parse_args(argv)
    if not a.assert_bim_world_meters:
        p.error("Refusing unverified units/frames; specify --assert-bim-world-meters")
    if "geom-" not in a.output.parts:
        p.error("Output MUST live in a geom-* research run (runs/geom-.../geometry/)")
    try:
        result = run_file_comparison(
            a.scan_aligned, a.bim_reference, a.output,
            threshold_m=a.threshold_m, max_points=a.max_points, seed=a.seed,
            anchors_json=a.anchors_json,
            gravity_scan=tuple(a.gravity_scan) if a.gravity_scan else None,
            gravity_bim=tuple(a.gravity_bim) if a.gravity_bim else None,
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(json.dumps({"status": "blocked_or_failed", "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
