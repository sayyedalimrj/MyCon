"""Auditable old-vs-new geometry comparison; no invented quality superiority.

Source cloud measures retention/proximity only (it may itself contain noise).
Independent GT is optional and never silently inferred from the source.
Compatible PLY or NPY files can be sampled deterministically on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from benchmarks.mycon_2026.metrics import read_ply_xyz


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cloud(path: Path, max_points: int, seed: int):
    if not path.is_file() or path.suffix.lower() not in {".ply", ".npy"}:
        raise ValueError("Expected existing .ply or .npy point-cloud input")
    if max_points < 3:
        raise ValueError("max_points must be >= 3")
    if path.suffix.lower() == ".ply":
        data, count = read_ply_xyz(path, max_points=max_points, seed=seed)
        a = np.asarray(data, dtype=np.float64)
    else:
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] < 3 or not np.issubdtype(raw.dtype, np.number):
            raise ValueError("Invalid numeric Nx3 NPY")
        count = int(len(raw))
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(count, min(count, max_points), replace=False))
        a = np.asarray(raw[selected, :3], dtype=np.float64)
    if len(a) < 3 or not np.isfinite(a).all():
        raise ValueError("Invalid, nonfinite or undersized point cloud")
    return a, count


def directed(source: np.ndarray, target: np.ndarray, threshold: float):
    distances = cKDTree(target).query(source, k=1, workers=1)[0]
    return {"within_threshold_fraction": float(np.mean(distances <= threshold)),
            "mean_distance": float(np.mean(distances)),
            "p95_distance": float(np.quantile(distances, .95))}


def scored(pred: np.ndarray, reference: np.ndarray, threshold: float):
    a = directed(pred, reference, threshold)
    b = directed(reference, pred, threshold)
    precision, completeness = a["within_threshold_fraction"], b["within_threshold_fraction"]
    return {
        "accuracy_within_threshold": precision,
        "completeness_within_threshold": completeness,
        "fscore_within_threshold": float(2 * precision * completeness / (precision + completeness))
            if precision + completeness else 0.,
        "chamfer_l1_source_units": float((a["mean_distance"] + b["mean_distance"]) / 2),
        "pred_to_reference_p95_source_units": a["p95_distance"],
        "reference_to_pred_p95_source_units": b["p95_distance"],
    }


def read_manifest(path: Path):
    m = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(m, dict) or not m.get("coordinate_frame") or not m.get("units"):
        raise ValueError("Manifest must declare coordinate_frame and units")
    return m


def run_compare(source: Path, before: Path, after: Path, *,
                frame: str, units: str, threshold: float, max_points: int = 20000,
                seed: int = 42, gt: Path | None = None, gt_kind: str = "synthetic",
                gt_provenance: str | None = None, before_manifest: Path | None = None,
                after_manifest: Path | None = None):
    if (not frame or not units or not math.isfinite(threshold) or threshold <= 0
            or not isinstance(seed, int)):
        raise ValueError("Explicit source frame/units and a positive threshold are required")
    if (before_manifest is None) != (after_manifest is None):
        raise ValueError("Both manifests are required to independently check coordinate consistency")
    if before_manifest is not None:
        b, a = read_manifest(before_manifest), read_manifest(after_manifest)
        for m in (b, a):
            if (m["coordinate_frame"], m["units"]) != (frame, units):
                raise ValueError("Source frame/units mismatch: do not compare unregistered clouds")
    if gt_kind not in {"synthetic", "independent_survey", "independent_lidar"}:
        raise ValueError("Unrecognized GT origin")
    if gt is None and gt_provenance:
        raise ValueError("GT provenance cannot be supplied without actual GT")
    datasets = {"source": source, "before": before, "after": after}
    if gt is not None:
        datasets["ground_truth"] = gt
    loaded = {k: cloud(v, max_points, seed) for k, v in datasets.items()}
    src, old, new = (loaded[k][0] for k in ("source", "before", "after"))
    report = {"schema_version": "mycon.geometry_before_after_audit.v1",
              "coordinate_frame": frame, "units": units, "threshold_source_units": threshold,
              "seed": seed, "max_sample_points": max_points,
              "inputs": {k: {"path": str(path), "sha256": digest(path),
                             "total_points": loaded[k][1],
                             "sampled_points": len(loaded[k][0])} for k, path in datasets.items()},
              "comparability": "operator_declared_same_frame" if before_manifest is None
                               else "matching_declared_output_frames",
              "source_is_not_ground_truth": True,
              "before": {"source_retention_proxy": directed(src, old, threshold),
                         "proximity_to_source_proxy": directed(old, src, threshold)},
              "after": {"source_retention_proxy": directed(src, new, threshold),
                        "proximity_to_source_proxy": directed(new, src, threshold)},
              "quality_conclusion": "not_determined_without_independent_ground_truth",
              "scientific_field_validation": False,
              "warnings": ["Source retention/proximity do not establish geometric accuracy; source may contain outliers.",
                           "Sample-based nearest-neighbor metrics are affected by density, not just geometry.",
                           "Feature correctness/corner false positives require independent structural GT."]}
    if gt is not None:
        ref = loaded["ground_truth"][0]
        report["before"]["against_supplied_ground_truth"] = scored(old, ref, threshold)
        report["after"]["against_supplied_ground_truth"] = scored(new, ref, threshold)
        report["gt_origin_declared"] = gt_kind
        report["gt_provenance_ref"] = gt_provenance
        report["quality_conclusion"] = "comparative_metrics_reported_no_unconditional_superiority"
        report["scientific_field_validation"] = False
        report["warnings"].append(
            "GT origin/provenance is only operator-declared; independent review and registration QA are not performed by this script.")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare identical-frame old/new geometry without invented accuracy")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--units", default="unknown_reconstruction_units")
    p.add_argument("--threshold", required=True, type=float)
    p.add_argument("--max-points", default=20000, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--ground-truth", type=Path)
    p.add_argument("--gt-kind", default="synthetic",
                   choices=["synthetic", "independent_survey", "independent_lidar"])
    p.add_argument("--gt-provenance")
    p.add_argument("--before-manifest", type=Path)
    p.add_argument("--after-manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        report = run_compare(a.source, a.before, a.after, frame=a.frame, units=a.units,
                             threshold=a.threshold, max_points=a.max_points, seed=a.seed,
                             gt=a.ground_truth, gt_kind=a.gt_kind, gt_provenance=a.gt_provenance,
                             before_manifest=a.before_manifest, after_manifest=a.after_manifest)
        if a.out.exists():
            raise FileExistsError("Refusing to overwrite existing comparison report")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"BEFORE_AFTER_AUDIT_READY: {a.out}; {report['quality_conclusion']}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(f"BEFORE_AFTER_AUDIT_BLOCKED: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
