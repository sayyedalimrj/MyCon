from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .colmap_model import ColmapTextModel, image_by_name, camera_depth_for_point
from .config_access import cfg_get
from .depth_provider import DepthMapRecord
from .io_utils import ensure_dir


@dataclass(frozen=True)
class AlignmentResult:
    image_name: str
    depth_path: Path
    aligned_depth_path: Path
    status: str
    scale: float
    shift: float
    anchor_count: int
    rmse_m: float
    reason: str
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    method: str = "scale_only_ransac"


def load_depth_map(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float32)
    if suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data.files else data.files[0]
        return np.asarray(data[key], dtype=np.float32)
    if suffix == ".pfm":
        return _read_pfm(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read depth map: {path}")
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.max(initial=0) > 255:
        scale = 1000.0
    else:
        scale = 255.0
    return arr / scale


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii", errors="replace").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        dims = handle.readline().decode("ascii", errors="replace").strip()
        width, height = [int(v) for v in dims.split()]
        scale = float(handle.readline().decode("ascii", errors="replace").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
        channels = 3 if header == "PF" else 1
        data = data.reshape((height, width, channels)) if channels == 3 else data.reshape((height, width))
        return np.flipud(data)


def save_aligned_depth(path: Path, depth: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, depth.astype(np.float32))


def _weighted_scale(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(x, y) / denom)


def _bucket_depth_pairs(x: np.ndarray, y: np.ndarray, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce uneven anchor distributions by averaging depth-sorted buckets.

    Sparse COLMAP anchors are often concentrated on textured patches. Bucketing by
    target depth prevents one wall/edge cluster from dominating the scale estimate.
    """
    if bucket_count <= 1 or x.size < bucket_count * 3:
        return x, y
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]
    buckets = np.array_split(np.arange(x_sorted.size), min(bucket_count, x_sorted.size))
    bx: list[float] = []
    by: list[float] = []
    for b in buckets:
        if b.size == 0:
            continue
        xv = x_sorted[b]
        yv = y_sorted[b]
        valid = np.isfinite(xv) & np.isfinite(yv) & (xv > 0) & (yv > 0)
        if valid.any():
            bx.append(float(np.median(xv[valid])))
            by.append(float(np.median(yv[valid])))
    if len(bx) < 3:
        return x, y
    return np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)


def fit_scale_depth_ransac(
    raw_values: np.ndarray,
    target_depths: np.ndarray,
    *,
    iterations: int = 200,
    inlier_abs_threshold_m: float = 0.20,
    inlier_rel_threshold: float = 0.05,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
    bucket_count: int = 16,
    random_seed: int = 42,
) -> tuple[float, float, float, int, float]:
    """Robust scale-only alignment: Z_colmap = scale * Z_da3.

    A constant depth shift is intentionally not fitted. In a pinhole camera model,
    adding a global shift to Z creates pixel-dependent X/Y distortions during
    backprojection. DA3 Metric depths should therefore be corrected by a scale
    factor only, with shift locked to zero.
    """
    valid = np.isfinite(raw_values) & np.isfinite(target_depths) & (raw_values > 0) & (target_depths > 0)
    x = raw_values[valid].astype(np.float64)
    y = target_depths[valid].astype(np.float64)
    if x.size < 3:
        return 1.0, 0.0, math.inf, int(x.size), 0.0

    # Candidate generation can use bucketed anchors, but scoring always uses all anchors.
    cand_x, cand_y = _bucket_depth_pairs(x, y, bucket_count=bucket_count)
    rng = np.random.default_rng(random_seed)
    best_scale = float(np.median(y / np.maximum(x, 1e-12)))
    best_inliers = np.zeros_like(x, dtype=bool)
    best_score = (-1, math.inf)

    candidate_scales: list[float] = [best_scale, _weighted_scale(x, y), _weighted_scale(cand_x, cand_y)]
    sample_size = 2 if cand_x.size >= 2 else 1
    ransac_iters = max(0, int(iterations))
    for _ in range(ransac_iters):
        if cand_x.size < sample_size:
            break
        idx = rng.choice(cand_x.size, size=sample_size, replace=False)
        ratios = cand_y[idx] / np.maximum(cand_x[idx], 1e-12)
        scale = float(np.median(ratios))
        candidate_scales.append(scale)

    for scale in candidate_scales:
        if not np.isfinite(scale) or scale < min_scale or scale > max_scale:
            continue
        residuals = np.abs(y - scale * x)
        thresholds = np.maximum(float(inlier_abs_threshold_m), float(inlier_rel_threshold) * np.maximum(y, 1e-9))
        inliers = residuals <= thresholds
        count = int(inliers.sum())
        if count >= 2:
            refined = _weighted_scale(x[inliers], y[inliers])
            if np.isfinite(refined) and min_scale <= refined <= max_scale:
                scale = refined
                residuals = np.abs(y - scale * x)
                inliers = residuals <= thresholds
                count = int(inliers.sum())
        rmse = float(np.sqrt(np.mean((y[inliers] - scale * x[inliers]) ** 2))) if count else math.inf
        score = (count, -rmse if np.isfinite(rmse) else -1e18)
        if score > best_score:
            best_score = score
            best_scale = float(scale)
            best_inliers = inliers

    inlier_count = int(best_inliers.sum())
    if inlier_count >= 2:
        best_scale = _weighted_scale(x[best_inliers], y[best_inliers])
        residuals = y[best_inliers] - best_scale * x[best_inliers]
        rmse = float(np.sqrt(np.mean(residuals * residuals)))
    else:
        residuals = y - best_scale * x
        rmse = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else math.inf
        inlier_count = 0
    inlier_ratio = float(inlier_count / x.size) if x.size else 0.0
    return float(best_scale), 0.0, rmse, int(x.size), inlier_ratio


def fit_affine_depth(raw_values: np.ndarray, target_depths: np.ndarray, *, trim_ratio: float = 0.15) -> tuple[float, float, float, int]:
    """Backward-compatible wrapper; Stage 6 now enforces scale-only alignment."""
    scale, shift, rmse, used, _ = fit_scale_depth_ransac(raw_values, target_depths)
    return scale, shift, rmse, used


def _sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    xs = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return depth[ys, xs].astype(np.float64)



def align_depth_maps(
    cfg: Any,
    model: ColmapTextModel,
    records: list[DepthMapRecord],
    aligned_dir: Path,
) -> list[AlignmentResult]:
    min_anchors = int(cfg_get(cfg, "da3.min_alignment_anchors", 20))
    max_rmse = float(cfg_get(cfg, "da3.max_alignment_rmse_m", 2.0))
    min_inlier_ratio = float(cfg_get(cfg, "da3.min_alignment_inlier_ratio", 0.25))
    ransac_iterations = int(cfg_get(cfg, "da3.alignment_ransac_iterations", 200))
    ransac_abs = float(cfg_get(cfg, "da3.alignment_ransac_inlier_abs_m", 0.20))
    ransac_rel = float(cfg_get(cfg, "da3.alignment_ransac_inlier_rel", 0.05))
    min_scale = float(cfg_get(cfg, "da3.alignment_min_scale", 0.05))
    max_scale = float(cfg_get(cfg, "da3.alignment_max_scale", 20.0))
    bucket_count = int(cfg_get(cfg, "da3.alignment_depth_bucket_count", 16))
    seed = int(cfg_get(cfg, "project.random_seed", 42))
    by_name = image_by_name(model)
    results: list[AlignmentResult] = []
    ensure_dir(aligned_dir)

    for idx, record in enumerate(records):
        image = by_name.get(record.image_name)
        out_path = aligned_dir / f"{Path(record.image_name).stem}.npy"
        if image is None:
            results.append(AlignmentResult(record.image_name, record.depth_path, out_path, "failed", 1.0, 0.0, 0, math.inf, "image_not_registered"))
            continue
        raw_depth = load_depth_map(record.depth_path)
        anchors_xy: list[list[float]] = []
        anchors_z: list[float] = []
        for xy, pid in zip(image.xys, image.point3d_ids):
            if int(pid) < 0:
                continue
            point = model.points3d.get(int(pid))
            if point is None:
                continue
            z = camera_depth_for_point(image, point)
            if z <= 0 or not np.isfinite(z):
                continue
            anchors_xy.append([float(xy[0]), float(xy[1])])
            anchors_z.append(float(z))
        if len(anchors_z) < min_anchors:
            results.append(AlignmentResult(record.image_name, record.depth_path, out_path, "failed", 1.0, 0.0, len(anchors_z), math.inf, "insufficient_sparse_depth_anchors"))
            continue
        raw_values = _sample_depth(raw_depth, np.asarray(anchors_xy, dtype=np.float64))
        target_values = np.asarray(anchors_z, dtype=np.float64)
        scale, shift, rmse, n_total, inlier_ratio = fit_scale_depth_ransac(
            raw_values,
            target_values,
            iterations=ransac_iterations,
            inlier_abs_threshold_m=ransac_abs,
            inlier_rel_threshold=ransac_rel,
            min_scale=min_scale,
            max_scale=max_scale,
            bucket_count=bucket_count,
            random_seed=seed + idx,
        )
        inlier_count = int(round(inlier_ratio * n_total))
        aligned = scale * raw_depth.astype(np.float32)
        aligned[~np.isfinite(aligned)] = 0
        aligned = np.maximum(aligned, 0).astype(np.float32)
        save_aligned_depth(out_path, aligned)
        status = "ok" if inlier_count >= min_anchors and rmse <= max_rmse and inlier_ratio >= min_inlier_ratio else "warning"
        reason = "scale_only_ransac_aligned" if status == "ok" else f"alignment_warning:rmse={rmse:.4f},inliers={inlier_count},ratio={inlier_ratio:.3f}"
        results.append(AlignmentResult(record.image_name, record.depth_path, out_path, status, scale, shift, n_total, rmse, reason, inlier_count, inlier_ratio))
    return results


def write_alignment_manifest(path: Path, results: list[AlignmentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image_name",
            "depth_path",
            "aligned_depth_path",
            "status",
            "scale",
            "shift",
            "anchor_count",
            "inlier_count",
            "inlier_ratio",
            "rmse_m",
            "method",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image_name": r.image_name,
                    "depth_path": r.depth_path.as_posix(),
                    "aligned_depth_path": r.aligned_depth_path.as_posix(),
                    "status": r.status,
                    "scale": f"{r.scale:.8f}",
                    "shift": f"{r.shift:.8f}",
                    "anchor_count": r.anchor_count,
                    "inlier_count": r.inlier_count,
                    "inlier_ratio": f"{r.inlier_ratio:.8f}",
                    "rmse_m": f"{r.rmse_m:.8f}" if np.isfinite(r.rmse_m) else "inf",
                    "method": r.method,
                    "reason": r.reason,
                }
            )
