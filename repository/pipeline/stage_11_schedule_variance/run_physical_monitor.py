"""Non-destructive Stage 09→11 physical-monitor delivery of five auditable files.

CLI reads existing Stage9 CSV, A's optional *verified* geometry manifest,
approved IFC-to-activity mapping/BOQ, dated baseline, and optional independent
held-out truth. Raw images, BIM, videos, checkpoints and Stage8 never modified.

Example (from repository/):
python -m pipeline.stage_11_schedule_variance.run_physical_monitor \
  --element-metrics-csv configs/research_progress/round2_contract_fixture/elements_stage9.csv \
  --schedule-csv configs/research_progress/round2_contract_fixture/schedule.csv \
  --mapping-csv configs/research_progress/round2_contract_fixture/mapping.csv \
  --data-date-utc 2026-01-16T00:00:00Z --output-root /tmp/mycon-b \
  --research-run-id progress-monitor-synthetic-01
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.common.bim_schedule_mapping import load_mapping_csv
from pipeline.common.schedule_io import load_schedule_csv, parse_iso_datetime
from pipeline.stage_09_progress.enrich_scientific_evidence import enrich_scientific_evidence_rows
from pipeline.stage_09_progress.element_status_export import (
    evaluate_element_truth, evaluate_expert_progress, status_for_element,
)
from .scientific_progress import build_scientific_progress_report

SCHEMA = "mycon.physical_monitor.v1"
FILENAMES = (
    "element_status.csv", "element_metrics.csv", "activity_progress.csv",
    "schedule_variance.json", "progress_dashboard.json",
)


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or any(col not in reader.fieldnames for col in required):
            raise ValueError("missing_required_csv_header:" + str(path))
        data = list(reader)
    if any(None in row for row in data):
        raise ValueError("malformed_csv_columns:" + str(path))
    return data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = ()) -> None:
    columns = list(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def build_monitor_outputs(
    *, element_rows: list[dict[str, Any]], schedule: Any, mapping: Any,
    baselines: list[dict[str, Any]], date: dt.datetime, run_id: str,
    geometry_summary: dict[str, Any], ground_truth: list[dict[str, Any]] | None = None,
    expert_truth: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           dict[str, Any], dict[str, Any]]:
    """Pure assembly: output payloads only, no filesystem changes."""
    if not element_rows or not schedule.activities or not mapping.entries:
        raise ValueError("empty_elements_schedule_or_mapping")
    ids = [str(r.get("global_id") or "").strip() for r in element_rows]
    if any(not x for x in ids):
        raise ValueError("missing_global_id")
    schedule_ids = {a.activity_id for a in schedule.activities}
    map_ids = {e.activity_id for e in mapping.entries}
    if map_ids - schedule_ids:
        raise ValueError("mapping_activity_not_in_schedule:" + ",".join(sorted(map_ids - schedule_ids)))
    counts = Counter(ids)
    assigned = {e.ifc_global_id for e in mapping.entries}
    id_to_aids: dict[str, list[str]] = {}
    id_to_wbs: dict[str, list[str]] = {}
    wbs_by_aid = {a.activity_id: a.wbs_code for a in schedule.activities}
    for entry in mapping.entries:
        id_to_aids.setdefault(entry.ifc_global_id, []).append(entry.activity_id)
        id_to_wbs.setdefault(entry.ifc_global_id, []).append(wbs_by_aid[entry.activity_id])
    status_rows = []
    enriched_rows = []
    for original in element_rows:
        row = dict(original)
        gid = row["global_id"].strip()
        stat = status_for_element(row, data_date=date,
                                  mapped=gid in assigned, unique=counts[gid] == 1)
        stat["activity_ids"] = ";".join(sorted(set(id_to_aids.get(gid, ()))))
        stat["wbs_codes"] = ";".join(sorted(set(id_to_wbs.get(gid, ()))))
        stat["run_id"] = run_id
        stat["data_date_utc"] = date.isoformat(timespec="seconds")
        status_rows.append(stat)
        row["construction_status"] = stat["construction_status"]
        row["construction_status_reason_codes"] = stat["reason_codes"]
        row["construction_status_schema"] = stat["schema_version"]
        row["activity_ids"] = stat["activity_ids"]
        row["wbs_codes"] = stat["wbs_codes"]
        row["run_id"] = run_id
        row["data_date_utc"] = stat["data_date_utc"]
        enriched_rows.append(row)

    scientific = build_scientific_progress_report(
        schedule=schedule, mapping=mapping, element_rows=enriched_rows,
        schedule_rows=baselines, data_date=date, run_id=run_id,
    )
    activity_rows = []
    variance_rows = []
    for a in scientific["activities"]:
        record = {key: a.get(key) for key in (
            "activity_id", "activity_name", "wbs_code", "physical_progress_estimate",
            "physical_progress_lower_bound_pct", "physical_progress_upper_bound_pct",
            "planned_percent_complete", "schedule_variance_percent",
            "schedule_variance_status", "observability_ratio", "evidence_confidence",
            "verified_quantity_weight_ratio", "n_mapped_elements",
            "n_verified_quantity_elements", "mapping_weight_basis",
            "schedule_decision_quality_gate",
        )}
        record.update({
            "schema_version": "mycon.activity_progress.v1",
            "run_id": run_id,
            "data_date_utc": date.isoformat(timespec="seconds"),
            "progress_interval_type": "deterministic_partial_identification_not_CI",
            "schedule_variance_bounds_pct": json.dumps(a["schedule_variance_bounds_pct"]),
            "reason_codes": ";".join(a["warnings"]),
        })
        activity_rows.append(record)
        variance_rows.append({
            key: a.get(key) for key in (
                "activity_id", "wbs_code", "physical_progress_estimate",
                "planned_percent_complete", "schedule_variance_percent",
                "schedule_variance_bounds_pct", "schedule_variance_status",
                "schedule_decision_quality_gate", "baseline_audit", "warnings",
            )
        })
    variance = {
        "schema_version": "mycon.physical_monitor.schedule_variance.v1",
        "scientific_source_schema": scientific["schema_version"],
        "run_id": run_id, "data_date_utc": scientific["data_date_utc"],
        "overall": scientific["overall"], "activities": variance_rows,
        "uncertainty_method": scientific["uncertainty_method"],
    }
    scores = ({
        "evaluation_status": "not_executed_missing_independent_ground_truth",
        "precision_recall_f1": None, "fraction_mae": None
    } if ground_truth is None else evaluate_element_truth(status_rows, ground_truth, data_date=date))
    expert = ({
        "evaluation_status": "not_executed_missing_independent_expert_report",
        "mae_percentage_points": None
    } if expert_truth is None else evaluate_expert_progress(
        scientific["activities"], expert_truth, data_date=date))
    dashboard = {
        "schema_version": "mycon.physical_monitor.dashboard.v1",
        "scientific_source_schema": scientific["schema_version"],
        "run_id": run_id, "data_date_utc": scientific["data_date_utc"],
        "n_elements": len(status_rows),
        "element_status_counts": dict(Counter(r["construction_status"] for r in status_rows)),
        "n_occluded": sum(r["occlusion_state"] == "occluded" for r in status_rows),
        "n_out_of_fov": sum(r["occlusion_state"] == "out_of_fov" for r in status_rows),
        "unknown_reason_counts": dict(Counter(
            code for row in status_rows for code in row["reason_codes"].split(";") if code)),
        "geometry_validation": geometry_summary,
        "quality_gates": scientific["quality_gates"],
        "overall": scientific["overall"],
        "wbs": scientific["wbs"],
        "ground_truth_metrics": scores,
        "expert_report_comparison": expert,
        "scientific_progress_v2": scientific,
        "safety": {
            "not_evidenced_is_not_not_built": True,
            "metric_geometry_requires_independent_scale_registration_and_ifc_mapping": True,
            "physical_progress_derived_from_approved_quantity_only": True,
            "unknowns_are_not_zero": True,
            "partial_identification_bounds_are_not_confidence_intervals": True,
            "synthetic_fixture_does_not_establish_real_world_validity": True,
        },
    }
    return status_rows, enriched_rows, activity_rows, variance, dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("element-metrics-csv", "schedule-csv", "mapping-csv"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--expert-progress-csv", type=Path)
    parser.add_argument("--data-date-utc", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--research-run-id", required=True)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--expected-ifc-sha256")
    parser.add_argument("--expected-scan-sha256")
    parser.add_argument("--expected-registration-sha256")
    parser.add_argument("--expected-project-id")
    args = parser.parse_args(argv)
    try:
        run_id = args.research_run_id
        if not run_id.startswith("progress-") or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            raise ValueError("unsafe_or_invalid_progress_run_id")
        date = parse_iso_datetime(args.data_date_utc)
        schedule = load_schedule_csv(args.schedule_csv)
        if schedule.provenance.n_skipped:
            raise ValueError("schedule_has_skipped_invalid_rows")
        mapping = load_mapping_csv(args.mapping_csv)
        raw = _read_csv(args.element_metrics_csv, ("global_id",))
        baseline_path = args.baseline_csv or args.schedule_csv
        baselines = _read_csv(baseline_path, ("activity_id",))
        ground = _read_csv(args.ground_truth_csv, ("global_id", "split", "site_id", "capture_session_id", "reviewer", "verified_at_utc", "document_sha256")) if args.ground_truth_csv else None
        expert = _read_csv(args.expert_progress_csv, ("activity_id", "split", "site_id", "capture_session_id", "reviewer", "verified_at_utc", "document_sha256")) if args.expert_progress_csv else None
        stage9, geometry = enrich_scientific_evidence_rows(
            raw, manifest_path=args.geometry_manifest, project_root=args.project_root,
            expected_ifc_sha256=args.expected_ifc_sha256,
            expected_scan_sha256=args.expected_scan_sha256,
            expected_registration_sha256=args.expected_registration_sha256,
            expected_project_id=args.expected_project_id, data_date=date,
        )
        result = build_monitor_outputs(
            element_rows=stage9, schedule=schedule, mapping=mapping,
            baselines=baselines, date=date, run_id=run_id,
            geometry_summary=geometry, ground_truth=ground, expert_truth=expert,
        )
        out = (args.output_root / run_id).resolve()
        if not out.is_relative_to(args.output_root.resolve()):
            raise ValueError("unsafe_output_path")
        out.mkdir(parents=True, exist_ok=False)
        statuses, elements, activities, variance, dashboard = result
        _csv(out / FILENAMES[0], statuses)
        _csv(out / FILENAMES[1], elements)
        _csv(out / FILENAMES[2], activities)
        input_files = {
            "element_metrics": args.element_metrics_csv, "schedule": args.schedule_csv,
            "mapping": args.mapping_csv, "baseline": baseline_path,
        }
        if args.geometry_manifest:
            input_files["geometry_manifest"] = args.geometry_manifest
        if args.ground_truth_csv:
            input_files["ground_truth"] = args.ground_truth_csv
        if args.expert_progress_csv:
            input_files["expert_progress"] = args.expert_progress_csv
        if any(not path.resolve().is_relative_to(args.project_root.resolve())
               for path in input_files.values()):
            raise ValueError("input_outside_project_root")
        inputs = {
            name: {"relative_path": str(path.resolve().relative_to(args.project_root.resolve())),
                   "sha256": _sha(path)}
            for name, path in (
                ("element_metrics", args.element_metrics_csv),
                ("schedule", args.schedule_csv), ("mapping", args.mapping_csv),
                ("baseline", baseline_path),
                *((("geometry_manifest", args.geometry_manifest),) if args.geometry_manifest else ()),
                *((("ground_truth", args.ground_truth_csv),) if args.ground_truth_csv else ()),
                *((("expert_progress", args.expert_progress_csv),) if args.expert_progress_csv else ()),
            )
        }
        provenance = {
            "schema_version": SCHEMA, "run_id": run_id,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "input_artifacts": inputs,
            "output_csv_sha256": {name: _sha(out / name) for name in FILENAMES[:3]},
            "coordinate_frame": "bim_world" if geometry["geometry_adapter_valid"] else "unknown",
            "coordinate_units": "m" if geometry["geometry_adapter_valid"] else "unknown",
            "run_directory": str(out),
        }
        variance["provenance"] = provenance
        dashboard["provenance"] = provenance
        _json(out / FILENAMES[3], variance)
        provenance["outputs"] = {
            name: {"relative_path": name, "sha256": _sha(out / name)}
            for name in FILENAMES[:4]
        }
        provenance["outputs"]["progress_dashboard.json"] = {
            "relative_path": "progress_dashboard.json", "sha256": None,
            "reason": "self_digest_cannot_be_embedded_in_the_same_file"
        }
        _json(out / FILENAMES[4], dashboard)
        print("PHYSICAL_MONITOR_OK run_id=" + run_id + " output=" + str(out)
              + " decision_gate=" + str(dashboard["quality_gates"]["schedule_decision"]))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print("PHYSICAL_MONITOR_FAILED " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
