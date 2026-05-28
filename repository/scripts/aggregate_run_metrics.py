#!/usr/bin/env python3
"""Aggregate per-stage JSON reports into a single flat run-metrics artifact.

The pipeline writes a JSON report per stage under ``runs/<run_id>/reports/``.
Each report uses a partly-shared envelope (``stage``, ``status``,
``elapsed_sec``, sometimes ``quality_gate``) plus stage-specific sub-trees.
Without a single per-run summary, comparing two runs (e.g. baseline vs robust
ICP) means manually opening 10+ JSONs.

This script produces:

- ``runs/<run_id>/reports/run_metrics.json`` — a list of flat records, one per
  stage, with the keys most useful for cross-run comparison surfaced.
- ``runs/<run_id>/reports/run_metrics.csv`` — the same records as a CSV with a
  superset header, suitable for ``pandas.read_csv``.

It is read-only with respect to all existing artifacts and never invokes any
stage. If a stage's report is missing the script records that as a row with
``status="missing_report"`` rather than failing — this lets it run on a
partial pipeline run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Maps stage display name -> conventional report basename.
# Discovery is filesystem-based, so missing reports do not crash the run.
_STAGE_REPORTS: dict[str, str] = {
    "stage_01_ingest": "stage_01_ingest_report.json",
    "stage_02_keyframes": "keyframe_summary.json",
    "stage_03_colmap": "sparse_stats.json",
    "stage_04_refinement": "refinement_stats.json",
    "stage_04_5_cams_gs": "cams_gs_prepare_summary.json",
    "stage_05_dense": "dense_summary.json",
    "stage_06_da3_assist": "da3_summary.json",
    "stage_07_cleanup": "cleanup_summary.json",
    "stage_07_5_vlm_qa": "vlm_qa_summary.json",
    "stage_07_6_viewer_export": "viewer_export_summary.json",
    "stage_07_7_cams_gs_evidence": "cams_gs_evidence_summary.json",
    "stage_08_metric_alignment": "metric_alignment_report.json",
    "stage_08_bim_registration": "registration_report.json",
    "stage_09_progress": "progress_summary.json",
}


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_path(d: Any, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _flat_record_for(stage: str, report: dict[str, Any] | None) -> dict[str, Any]:
    """Collapse a stage report into a flat record suitable for cross-run comparison.

    The selected keys are the smallest defensible set that covers the
    questions a researcher asks about a run: did each stage run, how long, and
    what is its primary quality signal.
    """
    if report is None:
        return {"stage": stage, "status": "missing_report"}

    record: dict[str, Any] = {
        "stage": stage,
        "status": report.get("status"),
        "elapsed_sec": report.get("elapsed_sec") or report.get("duration_sec"),
    }

    # Stage-specific signals are surfaced where they exist; absent fields stay
    # absent so consumers can use ``DictReader.get`` safely.
    fitness = _get_path(report, "icp", "fitness")
    if fitness is not None:
        record["icp_fitness"] = fitness
    rmse = _get_path(report, "icp", "inlier_rmse")
    if rmse is not None:
        record["icp_inlier_rmse_m"] = rmse
    method = _get_path(report, "icp", "method")
    if method is not None:
        record["icp_method"] = method
    robust = _get_path(report, "icp", "robust_loss")
    if isinstance(robust, dict):
        record["icp_robust_loss_requested"] = robust.get("requested")
        record["icp_robust_loss_applied"] = robust.get("applied")
        record["icp_robust_loss_k_m"] = robust.get("k_m")

    # Stage 5: dense MVS.
    dense_vertices = _get_path(report, "dense_stats", "fused_vertex_count")
    if dense_vertices is not None:
        record["dense_fused_vertex_count"] = dense_vertices

    # Stage 4: BA registered images / point counts.
    after = report.get("after_stats")
    if isinstance(after, dict):
        if "registered_image_count" in after:
            record["sparse_registered_images"] = after["registered_image_count"]
        if "sparse_point_count" in after:
            record["sparse_point_count"] = after["sparse_point_count"]

    # Stage 9: bidirectional + registration confidence.
    bidir = report.get("bidirectional_summary")
    if isinstance(bidir, dict):
        record["bidirectional_accuracy"] = bidir.get("accuracy")
        record["bidirectional_completeness"] = bidir.get("completeness")
        record["bidirectional_f_score"] = bidir.get("f_score")
        record["bidirectional_tau_m"] = bidir.get("tau_m")
    reg = report.get("registration_quality")
    if isinstance(reg, dict):
        record["registration_confidence_label"] = reg.get("confidence_label")
        record["registration_confidence_score"] = reg.get("confidence_score")
        record["registration_fitness"] = reg.get("fitness")

    qg = report.get("quality_gate")
    if isinstance(qg, dict):
        record["quality_gate_passed"] = qg.get("passed")
        if "failures" in qg:
            record["quality_gate_failures"] = ";".join(str(x) for x in qg.get("failures") or [])

    return record


def aggregate(reports_dir: Path) -> list[dict[str, Any]]:
    """Build the flat record list for every stage that has a report on disk."""
    return [
        _flat_record_for(stage, _safe_load_json(reports_dir / fname))
        for stage, fname in _STAGE_REPORTS.items()
    ]


def _all_keys(records: Iterable[dict[str, Any]]) -> list[str]:
    """Stable ordered union of keys across all records.

    First-seen order is preserved so the CSV header is stable across runs that
    produce the same set of stages.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in records:
        for k in r.keys():
            if k not in seen_set:
                seen.append(k)
                seen_set.add(k)
    return seen


def write_outputs(records: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    fieldnames = _all_keys(records) or ["stage", "status"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate per-stage reports into a single run summary.")
    p.add_argument("--reports-dir", required=True, type=Path, help="runs/<run_id>/reports directory")
    p.add_argument("--out-json", default=None, type=Path, help="Output JSON path (default: <reports_dir>/run_metrics.json)")
    p.add_argument("--out-csv", default=None, type=Path, help="Output CSV path (default: <reports_dir>/run_metrics.csv)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    reports_dir: Path = args.reports_dir
    if not reports_dir.exists():
        print(f"AGGREGATE_RUN_METRICS_FAILED: reports_dir does not exist: {reports_dir}", file=sys.stderr)
        return 1
    out_json: Path = args.out_json or (reports_dir / "run_metrics.json")
    out_csv: Path = args.out_csv or (reports_dir / "run_metrics.csv")

    records = aggregate(reports_dir)
    write_outputs(records, out_json, out_csv)

    n_with_status = sum(1 for r in records if r.get("status") not in {None, "missing_report"})
    print(
        f"AGGREGATE_RUN_METRICS_OK stages_total={len(records)} "
        f"stages_with_report={n_with_status} "
        f"json={out_json} csv={out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
