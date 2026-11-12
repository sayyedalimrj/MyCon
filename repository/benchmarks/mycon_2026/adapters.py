"""Fail-closed, read-only consumers of optional A-v1 and B-v2 research outputs.

Never derive construction completion from Stage9 geometric coverage.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import read_csv
from .protocol import resolve_input, sha256

APPROVED_QUANTITY = {"approved_measurement", "signed_site_log", "survey_qa"}
APPROVED_VISIBILITY = {"calibrated_camera_frustum", "raycast_zbuffer", "manual_occlusion_audit"}
APPROVED_CALIBRATION = {"heldout_calibration", "audited_quality_model"}
APPROVED_BASELINE = "approved_baseline"


def bounded(value, upper, *, field, nullable=True):
    if value is None or str(value).strip() == "":
        if nullable:
            return None
        raise ValueError(field + " is required")
    v = float(value)
    if not math.isfinite(v) or not 0 <= v <= upper:
        raise ValueError(field + " must be finite in [0," + str(upper) + "]")
    return v


def load_stage9_v2(path: Path):
    """Accept additive Stage9 CSV; legacy geometric coverage is never a fraction.

    A numerical progress estimate is consumable only with verifiable quantity
    provenance and matching approved executed/planned quantities.
    """
    out, seen = [], set()
    for row in read_csv(path):
        gid = str(row.get("global_id") or "").strip()
        if not gid or gid in seen:
            raise ValueError("Missing/duplicate Stage9 global_id")
        seen.add(gid)
        physical = bounded(row.get("physical_progress_estimate"), 1, field="Stage9 physical_progress_estimate")
        observed = bounded(row.get("observability_ratio"), 1, field="Stage9 observability_ratio")
        conf = bounded(row.get("evidence_confidence"), 1, field="Stage9 evidence_confidence")
        if physical is not None:
            planned = bounded_nonnegative(row.get("planned_quantity"), "planned_quantity")
            executed = bounded_nonnegative(row.get("executed_quantity"), "executed_quantity")
            verified = str(row.get("quantity_verified") or "").lower() in {"true", "1", "yes"}
            source = str(row.get("quantity_source") or "").strip().lower()
            if (not verified or source not in APPROVED_QUANTITY or planned is None
                    or planned <= 0 or executed is None or executed > planned
                    or not str(row.get("quantity_unit") or "").strip()
                    or str(row.get("physical_progress_status") or "").strip() != "verified_quantity"
                    or not math.isclose(physical, executed / planned, abs_tol=1e-7)):
                raise ValueError("Unverified/mismatched physical progress for " + gid)
        # Unknown provenance is not interpreted as confirmed visibility/confidence.
        if observed is not None and str(row.get("observability_source") or "").strip().lower() not in APPROVED_VISIBILITY:
            raise ValueError("Invalid observability provenance")
        if conf is not None and str(row.get("evidence_confidence_source") or "").strip().lower() not in APPROVED_CALIBRATION:
            raise ValueError("Invalid calibration provenance")
        out.append(dict(row, physical_progress_estimate=physical,
                        observability_ratio=observed, evidence_confidence=conf))
    return out


def bounded_nonnegative(value, field):
    if value is None or str(value).strip() == "":
        return None
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(field + " must be nonnegative finite quantity")
    return v


def load_scientific_progress_v2(path: Path):
    """Require exact v2 units. Activities and project use percent, not fractions."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "mycon.scientific_progress.v2":
        raise ValueError("Expected standalone mycon.scientific_progress.v2")
    units = obj.get("units") or {}
    if units.get("element_progress") != "fraction_0_1" or units.get("activity_and_project_percent") != "percent_0_100":
        raise ValueError("Scientific progress v2 units mismatch")
    if not isinstance(obj.get("activities"), list) or not isinstance(obj.get("overall"), dict):
        raise ValueError("Invalid v2 activity/overall shape")
    seen = set()
    for activity in obj["activities"]:
        aid = str(activity.get("activity_id") or "").strip()
        if not aid or aid in seen:
            raise ValueError("Missing/duplicate activity_id")
        seen.add(aid)
        _check_progress_bounds(activity)
        if (activity.get("physical_progress_estimate") is not None and
                str(activity.get("mapping_weight_basis") or "").lower() not in
                {"approved_boq", "approved_cost", "approved_quantity"}):
            raise ValueError("Unapproved activity mapping weight basis")
        bounded(activity.get("planned_percent_complete"), 100, field="approved plan")
        v = activity.get("schedule_variance_percent")
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 100):
            raise ValueError("Invalid schedule variance percentage points")
    _check_progress_bounds(obj["overall"])
    if (obj["overall"].get("physical_progress_estimate") is not None and
            obj["overall"].get("activity_weights_approved") is not True):
        raise ValueError("Unapproved project activity weights")
    return obj


def _check_progress_bounds(row):
    estimate = bounded(row.get("physical_progress_estimate"), 100, field="physical progress percent")
    lo = bounded(row.get("physical_progress_lower_bound_pct"), 100, field="progress lower bound")
    hi = bounded(row.get("physical_progress_upper_bound_pct"), 100, field="progress upper bound")
    if (lo is None) != (hi is None) or (lo is not None and (lo > hi or (estimate is not None and not lo - 1e-8 <= estimate <= hi + 1e-8))):
        raise ValueError("Invalid deterministic identification bounds")
    if row.get("progress_interval_type") and row["progress_interval_type"] != "partial_identification_not_confidence_interval":
        raise ValueError("Identification bounds misidentified as confidence intervals")


def activity_progress_metrics(report: dict, ground_truth: list[dict]):
    """Independent approved measured quantities + approved schedule baseline.

    GT rows: activity_id,physical_progress_pct,planned_percent_pct,baseline_id.
    A missing prediction is NOT a 0-percent prediction. Percent values stay
    percent values; schedule variance is measured in percentage points.
    """
    index = {a["activity_id"]: a for a in report["activities"]}
    if len(index) != len(report["activities"]):
        raise ValueError("Duplicate v2 activities")
    errors, variance_errors, missing, no_truth, all_ids = [], [], 0, 0, set()
    for gt in ground_truth:
        aid = str(gt.get("activity_id") or "").strip()
        if not aid or aid in all_ids:
            raise ValueError("Duplicate/empty GT activity_id")
        all_ids.add(aid)
        if not str(gt.get("baseline_id") or "").strip():
            raise ValueError("Independent schedule baseline_id required")
        actual = bounded(gt.get("physical_progress_pct"), 100, field="GT activity percent")
        planned = bounded(gt.get("planned_percent_pct"), 100, field="GT approved planned percent")
        if actual is None:
            no_truth += 1
            continue
        pred = index.get(aid)
        estimate = bounded(pred.get("physical_progress_estimate"), 100, field="predicted activity percent") if pred else None
        if estimate is None:
            missing += 1
            continue
        errors.append(abs(estimate - actual))
        if planned is not None and pred.get("schedule_variance_percent") is not None:
            variance_errors.append(abs(float(pred["schedule_variance_percent"]) - (actual - planned)))
    return {
        "n_activities_gt": len(all_ids), "n_gt_numeric": len(all_ids) - no_truth,
        "n_numeric_pairs": len(errors), "n_numeric_missing_prediction": missing,
        "physical_progress_mae_pp": sum(errors) / len(errors) if errors else None,
        "n_schedule_variance_pairs": len(variance_errors),
        "schedule_variance_mae_pp": sum(variance_errors) / len(variance_errors) if variance_errors else None,
    }


def project_progress_metrics(report: dict, truth: list[dict]):
    if len(truth) != 1 or not str(truth[0].get("baseline_id") or "").strip():
        raise ValueError("One approved project baseline and measurement required")
    gt = bounded(truth[0].get("physical_progress_pct"), 100, field="GT project percent", nullable=False)
    pred = bounded(report["overall"].get("physical_progress_estimate"), 100, field="project percent")
    plan = bounded(truth[0].get("planned_percent_pct"), 100, field="GT planned percent")
    var = report["overall"].get("schedule_variance_percent")
    return {"n_project_gt": 1, "n_project_numeric_pairs": int(pred is not None),
            "project_progress_absolute_error_pp": abs(pred - gt) if pred is not None else None,
            "project_schedule_variance_absolute_error_pp":
                abs(float(var) - (gt - plan)) if var is not None and plan is not None else None}


def validate_geometry_evidence(path: Path, *, root: Path):
    """Read-only manifest check. Never promote missing/invalid A evidence to truth."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    mandatory = ("run_id", "project_id", "created_at_utc", "ifc_sha256",
                 "scan_sha256", "registration_report_sha256", "registration_quality_gate",
                 "scan_to_bim_4x4", "metric_scale_verified", "provenance")
    if obj.get("schema_version") != "mycon.bim_geometry_evidence.v1":
        raise ValueError("Unknown A geometry evidence schema")
    if obj.get("coordinate_frame") != "bim_world" or obj.get("units") != "m":
        raise ValueError("A geometry frame must be bim_world in metres")
    if any(x not in obj for x in mandatory) or not isinstance(obj["metric_scale_verified"], bool):
        raise ValueError("Incomplete A geometry provenance")
    mat = obj["scan_to_bim_4x4"]
    if (not isinstance(mat, list) or len(mat) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in mat)
            or any(not math.isfinite(float(v)) for row in mat for v in row)
            or any(abs(float(v) - target) > 1e-6 for v, target in zip(mat[3], (0, 0, 0, 1)))):
        raise ValueError("Invalid scan-to-BIM transformation")
    assets = {}
    for key in ("surface_samples", "camera_poses", "element_visibility"):
        info = obj.get(key)
        if not isinstance(info, dict) or not info.get("path") or not info.get("sha256"):
            raise ValueError("A geometry asset missing path/hash: " + key)
        file = resolve_input(root, info["path"])
        if not file.is_file() or sha256(file) != info["sha256"]:
            raise ValueError("A geometry asset missing or SHA-256 mismatch: " + key)
        assets[key] = {"path": str(file), "sha256": info["sha256"]}
    gate = obj.get("registration_quality_gate")
    gate_status = gate.get("status") if isinstance(gate, dict) else gate
    accepted = obj["metric_scale_verified"] and str(gate_status).lower() in ("pass", "passed", "accepted")
    # A's declared metric scale is NOT independent evidence. Require a separate,
    # hashed control survey before even marking this manifest a candidate.
    control = obj.get("independent_control_evidence")
    verified_control = False
    if isinstance(control, dict) and control.get("path") and control.get("sha256"):
        survey = resolve_input(root, control["path"])
        verified_control = (survey.is_file() and sha256(survey) == control["sha256"]
                            and str(control.get("source") or "").lower() in
                            {"total_station_survey", "independent_lidar", "approved_survey"})
    return {"schema_version": obj["schema_version"], "asset_hashes_verified": True,
            "n_assets": len(assets), "independent_control_verified": verified_control,
            "valid_for_visibility_candidate": bool(accepted and verified_control),
            "independent_survey_gt_still_required": True, "assets": assets,
            "reason": None if accepted and verified_control else "missing_independent_survey_or_registration_approval"}
