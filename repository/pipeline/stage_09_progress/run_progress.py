from __future__ import annotations

import argparse
import html
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.common.config import load_config
from pipeline.common.logging_utils import setup_logging

from .config_access import cfg_float, project_name, run_id
from .input_selection import missing_required_inputs, stage9_paths
from .io_utils import read_csv, read_json, read_jsonl, write_csv_atomic, write_json_atomic
from .decision_enrichment import enrich_progress_decisions_from_config

LOGGER_NAME = "pipeline.stage_09_progress"


def _o3d():
    import open3d as o3d
    return o3d


def _load_cloud(path: Path):
    o3d = _o3d()
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"Point cloud has zero points: {path}")
    return cloud


def _points(cloud: Any) -> np.ndarray:
    return np.asarray(cloud.points, dtype=np.float64)


def _registration_confidence(report: dict[str, Any], cfg: Any | None = None) -> dict[str, Any]:
    icp = report.get("icp", {}) or {}
    qg = report.get("quality_gate", {}) or {}
    ifc = report.get("ifc", {}) or {}

    fitness = float(icp.get("fitness", 0.0) or 0.0)
    rmse = float(icp.get("inlier_rmse", math.inf) or math.inf)

    high_threshold = cfg_float(cfg, "progress.registration_high_fitness_threshold", 0.50) if cfg is not None else 0.50
    medium_threshold = cfg_float(cfg, "progress.registration_medium_fitness_threshold", 0.05) if cfg is not None else 0.05
    high_score = cfg_float(cfg, "progress.registration_high_confidence_score", 1.0) if cfg is not None else 1.0
    medium_score = cfg_float(cfg, "progress.registration_medium_confidence_score", 0.65) if cfg is not None else 0.65
    low_score = cfg_float(cfg, "progress.registration_low_confidence_score", 0.25) if cfg is not None else 0.25

    warnings = list(qg.get("warnings", []) or [])
    if high_threshold < medium_threshold:
        warnings.append(f"registration_thresholds_reordered:high={high_threshold:.6f}<medium={medium_threshold:.6f}")
        high_threshold, medium_threshold = medium_threshold, high_threshold

    if fitness >= high_threshold:
        label, score = "high", high_score
    elif fitness >= medium_threshold:
        label, score = "medium", medium_score
    else:
        label, score = "low", low_score

    synthetic_used = bool(ifc.get("synthetic_bim_fallback_used")) or str(ifc.get("source", "")).strip().lower() == "synthetic_test_fallback"
    if synthetic_used:
        label = "low"
        score = min(score, low_score)
        warnings.append("synthetic_bim_fallback_used_metrics_are_not_real_progress_evidence")

    if label == "low":
        warnings.append("registration_confidence_low_metrics_are_for_pipeline_validation")

    return {
        "fitness": fitness,
        "rmse_m": rmse,
        "confidence_label": label,
        "confidence_score": score,
        "quality_gate": qg,
        "warnings": sorted(set(warnings)),
        "synthetic_bim_fallback_used": synthetic_used,
        "thresholds": {
            "registration_high_fitness_threshold": high_threshold,
            "registration_medium_fitness_threshold": medium_threshold,
            "registration_high_confidence_score": high_score,
            "registration_medium_confidence_score": medium_score,
            "registration_low_confidence_score": low_score,
        },
    }


def _build_kdtree(cloud: Any):
    o3d = _o3d()
    return o3d.geometry.KDTreeFlann(cloud)


def _nearest_distances_with_indices(
    query_points: np.ndarray,
    target_cloud: Any,
    limit: int = 200000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled query indices and nearest-neighbor distances.

    Uses scipy.spatial.cKDTree for batched queries when SciPy is available;
    falls back to Open3D KDTreeFlann one-by-one queries otherwise.
    """
    if len(query_points) == 0 or len(target_cloud.points) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    all_indices = np.arange(len(query_points), dtype=np.int64)
    if len(query_points) > limit:
        rng = np.random.default_rng(9)
        sampled_indices = np.sort(rng.choice(all_indices, size=limit, replace=False))
    else:
        sampled_indices = all_indices

    sampled_points = np.asarray(query_points[sampled_indices], dtype=np.float64)
    target_points = np.asarray(target_cloud.points, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree

        distances, _nearest = cKDTree(target_points).query(sampled_points, k=1, workers=-1)
        finite = np.isfinite(distances)
        return sampled_indices[finite].astype(np.int64), distances[finite].astype(np.float64)
    except Exception:
        tree = _build_kdtree(target_cloud)
        kept_indices: list[int] = []
        distances: list[float] = []
        for original_index, p in zip(sampled_indices, sampled_points):
            _k, _idx, d2 = tree.search_knn_vector_3d(p.astype(float), 1)
            if d2:
                kept_indices.append(int(original_index))
                distances.append(math.sqrt(float(d2[0])))
        return np.asarray(kept_indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def _nearest_distances(query_points: np.ndarray, target_cloud: Any, limit: int = 200000) -> np.ndarray:
    _indices, distances = _nearest_distances_with_indices(query_points, target_cloud, limit=limit)
    return distances


def _select_points_in_bbox(points: np.ndarray, mn: list[Any], mx: list[Any], margin: float) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    mnv = np.asarray([float(x) for x in mn], dtype=np.float64) - margin
    mxv = np.asarray([float(x) for x in mx], dtype=np.float64) + margin
    mask = np.all((points >= mnv) & (points <= mxv), axis=1)
    return points[mask]


def _activity_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mapping = {}
    for row in rows:
        gid = row.get("global_id") or row.get("GlobalId") or row.get("element_global_id")
        if gid:
            mapping[gid] = row
    return mapping


def _schedule_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("activity_id", ""): row for row in rows if row.get("activity_id")}


def _status_from_metrics(
    coverage: float,
    confidence_label: str,
    threshold: float,
    *,
    partial_threshold: float = 0.20,
) -> str:
    if confidence_label == "low":
        return "uncertain_low_registration"
    if coverage >= threshold:
        return "likely_completed"
    if coverage >= partial_threshold:
        return "partially_observed"
    return "not_evidenced"


def _fieldnames() -> list[str]:
    return [
        "global_id",
        "name",
        "ifc_class",
        "activity_id",
        "activity_name",
        "coverage",
        "in_tolerance_ratio",
        "mean_deviation_m",
        "median_deviation_m",
        "p95_deviation_m",
        "point_count_evaluated",
        "status",
        "confidence",
        "registration_confidence",
        "notes",
    ]


def _write_deviation_map(path: Path, scan_cloud: Any, bim_cloud: Any, threshold: float) -> None:
    o3d = _o3d()
    scan_points = _points(scan_cloud)
    sampled_indices, sampled_distances = _nearest_distances_with_indices(scan_points, bim_cloud, limit=300000)
    if len(sampled_distances) == 0:
        o3d.io.write_point_cloud(str(path), scan_cloud, write_ascii=False)
        return

    distances = np.full(len(scan_points), float(np.nan), dtype=np.float64)
    distances[sampled_indices] = sampled_distances

    denom = max(threshold * 3.0, float(np.nanpercentile(distances, 95)) if np.isfinite(distances).any() else threshold)
    values = np.nan_to_num(np.clip(distances / max(denom, 1e-6), 0.0, 1.0))
    colors = np.column_stack([values, 1.0 - values, np.zeros_like(values)])
    out = scan_cloud.select_by_index(list(range(len(scan_cloud.points))))
    out.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), out, write_ascii=False)


def _html_value(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _html_value(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _dashboard_html(summary: dict[str, Any], element_rows: list[dict[str, Any]], activity_rows: list[dict[str, Any]]) -> str:
    rows_html = "\n".join(
        "<tr>"
        f"<td>{_html_value(r.get('name', ''))}</td>"
        f"<td>{_html_value(r.get('ifc_class', ''))}</td>"
        f"<td>{_html_value(r.get('coverage', ''))}</td>"
        f"<td>{_html_value(r.get('status', ''))}</td>"
        f"<td>{_html_value(r.get('confidence', ''))}</td>"
        "</tr>"
        for r in element_rows
    )
    act_html = "\n".join(
        "<tr>"
        f"<td>{_html_value(r.get('activity_id', ''))}</td>"
        f"<td>{_html_value(r.get('activity_name', ''))}</td>"
        f"<td>{_html_value(r.get('observed_percent', ''))}</td>"
        f"<td>{_html_value(r.get('confidence', ''))}</td>"
        f"<td>{_html_value(r.get('status', ''))}</td>"
        "</tr>"
        for r in activity_rows
    )

    project = summary.get("project", {}) or {}
    registration = summary.get("registration_quality", {}) or {}

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 9 Progress Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
.badge {{ display: inline-block; padding: 4px 8px; border: 1px solid #999; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
td, th {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #f2f2f2; }}
.warning {{ color: #8a4b00; font-weight: bold; }}
</style>
</head>
<body>
<h1>Stage 9 Progress Dashboard</h1>
<p><span class="badge">Project: {_html_value(project.get('name', ''))}</span>
<span class="badge">Run: {_html_value(project.get('run_id', ''))}</span></p>
<p><strong>Status:</strong> {_html_value(summary.get('status', ''))}</p>
<p><strong>Registration confidence:</strong> {_html_value(registration.get('confidence_label', ''))}
(fitness={float(registration.get('fitness', 0.0) or 0.0):.6f}, rmse={float(registration.get('rmse_m', 0.0) or 0.0):.6f})</p>
<p class="warning">Mode note: {_html_value(summary.get('mode_note', ''))}</p>
<h2>Element Metrics</h2>
<table>
<tr><th>Name</th><th>IFC Class</th><th>Coverage</th><th>Status</th><th>Confidence</th></tr>
{rows_html}
</table>
<h2>Activity Progress</h2>
<table>
<tr><th>Activity</th><th>Name</th><th>Observed %</th><th>Confidence</th><th>Status</th></tr>
{act_html}
</table>
</body>
</html>
"""


def _run_progress_unenriched(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    t0 = time.perf_counter()
    paths = stage9_paths(cfg)
    missing = missing_required_inputs(paths)
    if missing:
        raise RuntimeError("Missing Stage 9 inputs: " + "; ".join(missing))

    scan = _load_cloud(paths["scan_aligned"])
    bim = _load_cloud(paths["bim_reference"])
    scan_points = _points(scan)
    bim_points = _points(bim)

    elements = read_jsonl(paths["bim_elements"])
    reg_report = read_json(paths["registration_report"])
    schedule_rows = read_csv(paths["schedule_csv"])
    map_rows = read_csv(paths["element_activity_map"])
    gid_to_activity = _activity_map(map_rows)
    activity_info = _schedule_map(schedule_rows)

    threshold = cfg_float(cfg, "progress.deviation_threshold_m", 0.05)
    coverage_threshold = cfg_float(cfg, "progress.coverage_threshold", 0.65)
    partial_observed_threshold = cfg_float(cfg, "progress.partial_observed_threshold", 0.20)
    bbox_margin = cfg_float(cfg, "progress.element_bbox_margin_m", max(0.05, threshold * 2.0))
    reg_quality = _registration_confidence(reg_report, cfg)

    element_rows: list[dict[str, Any]] = []
    all_deviation_values: list[float] = []
    undercovered: list[dict[str, Any]] = []

    for element in elements:
        gid = str(element.get("global_id", ""))
        activity_row = gid_to_activity.get(gid, {})
        activity_id = activity_row.get("activity_id", "UNMAPPED")
        act = activity_info.get(activity_id, {})
        mn = element.get("bounds_min", [0, 0, 0])
        mx = element.get("bounds_max", [0, 0, 0])
        target_pts = _select_points_in_bbox(bim_points, mn, mx, bbox_margin)
        if len(target_pts) < 10:
            target_pts = bim_points

        distances = _nearest_distances(target_pts, scan, limit=50000)
        if len(distances) == 0:
            coverage = 0.0
            in_tol = 0.0
            mean_dev = median_dev = p95_dev = float("nan")
        else:
            all_deviation_values.extend(float(x) for x in distances if math.isfinite(float(x)))
            in_tol = float(np.mean(distances <= threshold))
            coverage = in_tol
            mean_dev = float(np.mean(distances))
            median_dev = float(np.median(distances))
            p95_dev = float(np.percentile(distances, 95))

        confidence = float(max(0.0, min(1.0, coverage * reg_quality["confidence_score"])))
        status = _status_from_metrics(
            coverage,
            reg_quality["confidence_label"],
            coverage_threshold,
            partial_threshold=partial_observed_threshold,
        )
        if coverage < coverage_threshold:
            undercovered.append({"global_id": gid, "name": element.get("name", ""), "coverage": coverage, "status": status})

        row = {
            "global_id": gid,
            "name": element.get("name", ""),
            "ifc_class": element.get("ifc_class", ""),
            "activity_id": activity_id,
            "activity_name": act.get("activity_name", ""),
            "coverage": f"{coverage:.6f}",
            "in_tolerance_ratio": f"{in_tol:.6f}",
            "mean_deviation_m": f"{mean_dev:.6f}" if math.isfinite(mean_dev) else "",
            "median_deviation_m": f"{median_dev:.6f}" if math.isfinite(median_dev) else "",
            "p95_deviation_m": f"{p95_dev:.6f}" if math.isfinite(p95_dev) else "",
            "point_count_evaluated": int(len(distances)),
            "status": status,
            "confidence": f"{confidence:.6f}",
            "registration_confidence": reg_quality["confidence_label"],
            "notes": "synthetic_or_low_confidence" if reg_quality["confidence_label"] == "low" else "",
        }
        element_rows.append(row)

    by_activity: dict[str, list[dict[str, Any]]] = {}
    for row in element_rows:
        by_activity.setdefault(row["activity_id"], []).append(row)

    activity_rows: list[dict[str, Any]] = []
    for activity_id, rows in sorted(by_activity.items()):
        act = activity_info.get(activity_id, {})
        coverages = [float(r["coverage"]) for r in rows]
        confidences = [float(r["confidence"]) for r in rows]
        observed = 100.0 * (sum(coverages) / max(1, len(coverages)))
        confidence = sum(confidences) / max(1, len(confidences))
        status = "uncertain_low_registration" if reg_quality["confidence_label"] == "low" else ("on_track_or_complete" if observed >= coverage_threshold * 100 else "behind_or_partial")
        activity_rows.append(
            {
                "activity_id": activity_id,
                "activity_name": act.get("activity_name", ""),
                "element_count": len(rows),
                "observed_percent": f"{observed:.2f}",
                "planned_percent": "100.00",
                "confidence": f"{confidence:.6f}",
                "status": status,
            }
        )

    deviations = np.asarray(all_deviation_values, dtype=np.float64)
    deviation_summary = {
        "threshold_m": threshold,
        "evaluated_distance_count": int(len(deviations)),
        "mean_deviation_m": float(np.mean(deviations)) if len(deviations) else None,
        "median_deviation_m": float(np.median(deviations)) if len(deviations) else None,
        "p95_deviation_m": float(np.percentile(deviations, 95)) if len(deviations) else None,
        "max_deviation_m": float(np.max(deviations)) if len(deviations) else None,
    }

    coverage_summary = {
        "coverage_threshold": coverage_threshold,
        "partial_observed_threshold": partial_observed_threshold,
        "element_count": len(element_rows),
        "undercovered_count": len(undercovered),
        "undercovered_regions": undercovered,
    }

    mode_note = "Pipeline validation mode. Metrics are deterministic, but interpretation depends on registration quality and whether the BIM is synthetic or real."
    summary = {
        "stage": "stage_09_progress",
        "status": "complete",
        "project": {"name": project_name(cfg), "run_id": run_id(cfg)},
        "inputs": {k: v.as_posix() for k, v in paths.items() if k in {"scan_aligned", "bim_reference", "bim_elements", "registration_report", "schedule_csv", "element_activity_map"}},
        "outputs": {k: v.as_posix() for k, v in paths.items() if k not in {"scan_aligned", "bim_reference", "bim_elements", "registration_report", "schedule_csv", "element_activity_map", "metrics_dir"}},
        "registration_quality": reg_quality,
        "deviation_summary": deviation_summary,
        "coverage_summary": coverage_summary,
        "element_count": len(element_rows),
        "activity_count": len(activity_rows),
        "mode_note": mode_note,
        "elapsed_sec": time.perf_counter() - t0,
    }

    write_csv_atomic(paths["element_metrics_csv"], element_rows, _fieldnames())
    write_csv_atomic(paths["activity_progress_csv"], activity_rows)
    write_json_atomic(paths["deviation_summary_json"], deviation_summary)
    write_json_atomic(paths["coverage_summary_json"], coverage_summary)
    write_json_atomic(paths["registration_quality_json"], reg_quality)
    write_json_atomic(paths["progress_summary_json"], summary)
    paths["dashboard_html"].parent.mkdir(parents=True, exist_ok=True)
    paths["dashboard_html"].write_text(_dashboard_html(summary, element_rows, activity_rows), encoding="utf-8")
    _write_deviation_map(paths["deviation_map_ply"], scan, bim, threshold)

    print(
        "STAGE_09_PROGRESS_OK "
        f"elements={len(element_rows)} activities={len(activity_rows)} "
        f"registration_confidence={reg_quality['confidence_label']} "
        f"dashboard={paths['dashboard_html']}"
    )
    return summary


def run_progress(*args, **kwargs):
    """Run Stage 9 and then enrich progress outputs with conservative decision fields."""
    result = _run_progress_unenriched(*args, **kwargs)

    cfg_or_path = kwargs.get("cfg") or kwargs.get("config") or kwargs.get("config_path")
    if cfg_or_path is None and args:
        cfg_or_path = args[0]

    try:
        if cfg_or_path is not None:
            enrich_progress_decisions_from_config(cfg_or_path)
    except Exception as exc:  # pragma: no cover - enrichment must not hide base Stage 9 output
        import logging
        logging.getLogger(__name__).warning("Stage 9 decision enrichment skipped: %s", exc)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 9: deviation, coverage and progress metrics")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(name=LOGGER_NAME, level=args.log_level)
    cfg = load_config(args.config)
    try:
        run_progress(cfg, force=args.force, log_level=args.log_level)
    except Exception as exc:
        print(f"STAGE_09_PROGRESS_FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
