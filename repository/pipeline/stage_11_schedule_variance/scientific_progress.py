"""Research-only, evidence-safe v2 physical progress and schedule variance.

Legacy Stage 11 acceptance proxies are intentionally untouched, for the
frozen dashboard v1 contract. THIS v2 output is authoritative for research.
It never uses Wilson intervals for weighted/partial element outcomes.

Unknown fractions lead to *partial-identification bounds*, not a 95% CI.
These bounds assume only that unknown executed fractions are within [0,1].
They are not sampling confidence intervals and have no coverage guarantee.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from pipeline.common.bim_schedule_mapping import BimScheduleMapping
from pipeline.common.schedule_io import Schedule
from pipeline.stage_09_progress.evidence_metrics import (
    element_scientific_fields,
    finite_number,
    unit_ratio,
)

SCHEMA = "mycon.scientific_progress.v2"
APPROVED_WEIGHTS = {"approved_boq", "approved_cost", "approved_quantity"}
APPROVED_BASELINE = "approved_baseline"


def _valid_weight(value: Any) -> float | None:
    parsed = finite_number(value)
    return parsed if parsed is not None and parsed > 0 else None


def _approved_baseline(row: Mapping[str, Any]) -> float | None:
    if str(row.get("baseline_source") or "").strip().lower() != APPROVED_BASELINE:
        return None
    if not str(row.get("baseline_id") or "").strip():
        return None
    percent = finite_number(row.get("approved_planned_percent"))
    return percent if percent is not None and 0 <= percent <= 100 else None


def _classify(planned: float | None, actual: float | None, band: float) -> str:
    if planned is None:
        return "unknown_schedule_baseline"
    if actual is None:
        return "unknown_evidence"
    if actual - planned > band:
        return "ahead"
    if planned - actual > band:
        return "behind"
    return "on_schedule"


def build_scientific_progress_report(
    *,
    schedule: Schedule,
    mapping: BimScheduleMapping,
    element_rows: Iterable[Mapping[str, Any]],
    data_date: dt.datetime,
    schedule_rows: Iterable[Mapping[str, Any]] = (),
    on_schedule_band_pct: float = 5.0,
    run_id: str = "progress-unspecified",
) -> dict[str, Any]:
    """Build research v2 without turning missing observations into zero.

    Element `executed_quantity/planned_quantity` requires signed source,
    `quantity_verified=true`, matching `quantity_unit` and a positive plan.
    Mapping weights need explicit approved `weight_basis`. A single missing
    quantity prevents a point estimate for the *whole* activity. Observability
    and confidence remain independent of quantities and each other.
    """
    if not run_id.startswith("progress-"):
        raise ValueError("research run ID must start with progress-")
    if not (math.isfinite(on_schedule_band_pct) and on_schedule_band_pct >= 0):
        raise ValueError("invalid tolerance")
    if data_date.tzinfo is None:
        data_date = data_date.replace(tzinfo=dt.timezone.utc)
    data_date = data_date.astimezone(dt.timezone.utc)

    input_rows = list(element_rows)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in input_rows:
        gid = str(row.get("global_id") or row.get("GlobalId") or "").strip()
        if gid:
            grouped.setdefault(gid, []).append(row)
    schedule_lookup = {
        str(row.get("activity_id") or "").strip(): row for row in schedule_rows
        if row.get("activity_id")
    }
    activities: list[dict[str, Any]] = []
    for activity in schedule.activities:
        aid = activity.activity_id
        entries = mapping.elements_for_activity(aid)
        warnings: list[str] = []
        ids = [e.ifc_global_id for e in entries]
        bases = {str(getattr(e, "weight_basis", "")).lower().strip() for e in entries}
        valid_weights = (
            bool(entries) and len(set(ids)) == len(ids)
            and len(bases) == 1 and next(iter(bases)) in APPROVED_WEIGHTS
            and all(_valid_weight(e.weight) is not None for e in entries)
        )
        if not entries:
            warnings.append("no_mapped_elements")
        if len(ids) != len(set(ids)):
            warnings.append("duplicate_ifc_mapping")
        if not valid_weights:
            warnings.append("mapping_weights_unverified_or_invalid")

        known_total = 0.0
        unknown_total = 0.0
        total = sum(float(e.weight) for e in entries) if valid_weights else None
        visible = 0.0
        quality = 0.0
        n_visible_known = 0
        n_quality_known = 0
        n_verified = 0
        element_evidence: list[dict[str, Any]] = []

        for entry in entries:
            gid = entry.ifc_global_id
            candidates = grouped.get(gid, [])
            weight = float(entry.weight) if valid_weights else None
            if len(candidates) != 1:
                fields = {
                    "physical_progress_estimate": None, "observability_ratio": None,
                    "evidence_confidence": None,
                    "physical_progress_status": "unknown",
                    "geometry_evidence_state": "unknown",
                }
                warnings.append(f"missing_or_duplicate_element_evidence:{gid}")
            else:
                fields = element_scientific_fields(candidates[0])
                # Never trust Stage 9's geometric acceptance alone as a signed
                # quantity. If raw approved quantity evidence wasn't exported,
                # the physical estimate remains undefined.
            fraction = fields["physical_progress_estimate"]
            if valid_weights:
                if fraction is None:
                    unknown_total += weight
                else:
                    known_total += weight * fraction
                    n_verified += 1
                obs = fields["observability_ratio"]
                if obs is not None:
                    visible += weight * obs
                    n_visible_known += 1
                quality_score = fields["evidence_confidence"]
                if quality_score is not None:
                    quality += weight * quality_score
                    n_quality_known += 1
            element_evidence.append({
                "ifc_global_id": gid,
                "mapping_weight": weight,
                "physical_progress_estimate": fraction,
                "observability_ratio": fields["observability_ratio"],
                "evidence_confidence": fields["evidence_confidence"],
                "physical_progress_status": fields["physical_progress_status"],
                "geometry_evidence_state": fields["geometry_evidence_state"],
            })

        if valid_weights and total is not None and total > 0:
            lower = 100.0 * known_total / total
            upper = 100.0 * (known_total + unknown_total) / total
            estimate = lower if n_verified == len(entries) else None
            obs_ratio = visible / total if n_visible_known == len(entries) else None
            conf = quality / total if n_quality_known == len(entries) else None
            known_weight_ratio = (total - unknown_total) / total
        else:
            lower = upper = estimate = obs_ratio = conf = known_weight_ratio = None

        baseline = schedule_lookup.get(aid, {})
        planned = _approved_baseline(baseline)
        provisional = activity.planned_percent_complete_at(data_date)
        variance = (estimate - planned) if estimate is not None and planned is not None else None
        variance_bounds = (
            [lower - planned, upper - planned]
            if lower is not None and planned is not None else None
        )
        status = _classify(planned, estimate, on_schedule_band_pct)
        if estimate is None:
            warnings.append("no_full_verified_quantity_coverage")
        if planned is None:
            warnings.append("approved_planned_baseline_missing")
        activities.append({
            "activity_id": aid,
            "activity_name": activity.activity_name,
            "physical_progress_estimate": estimate,
            "physical_progress_status": "verified_quantity" if estimate is not None else "unknown",
            "physical_progress_lower_bound_pct": lower,
            "physical_progress_upper_bound_pct": upper,
            "progress_interval_type": "partial_identification_not_confidence_interval",
            "observability_ratio": obs_ratio,
            "evidence_confidence": conf,
            "verified_quantity_weight_ratio": known_weight_ratio,
            "planned_percent_complete": planned,
            "provisional_linear_planned_percent": provisional,
            "planner_reported_percent": activity.percent_complete,
            "schedule_variance_percent": variance,
            "schedule_variance_bounds_pct": variance_bounds,
            "schedule_variance_status": status,
            "n_mapped_elements": len(entries),
            "n_verified_quantity_elements": n_verified,
            "n_observability_audited_elements": n_visible_known,
            "n_quality_calibrated_elements": n_quality_known,
            "mapping_weight_basis": next(iter(bases)) if valid_weights else None,
            "warnings": sorted(set(warnings)),
            "elements": element_evidence,
        })

    # Project-level aggregation uses independent, approved cross-activity
    # budget/BOQ weights. Counting IFC objects is never a cost-weight model.
    project_weights: list[float] = []
    project_bases: list[str] = []
    for item in activities:
        row = schedule_lookup.get(item["activity_id"], {})
        w = _valid_weight(row.get("activity_weight"))
        basis = str(row.get("activity_weight_basis") or "").strip().lower()
        if w is None or basis not in APPROVED_WEIGHTS:
            project_weights = []
            break
        project_weights.append(w)
        project_bases.append(basis)
    valid_project_weights = (
        bool(activities) and len(project_weights) == len(activities)
        and len(set(project_bases)) == 1
    )
    if valid_project_weights:
        total_w = sum(project_weights)
        if all(a["physical_progress_lower_bound_pct"] is not None for a in activities):
            proj_lower = sum(w*a["physical_progress_lower_bound_pct"] for w,a in zip(project_weights, activities)) / total_w
            proj_upper = sum(w*a["physical_progress_upper_bound_pct"] for w,a in zip(project_weights, activities)) / total_w
        else:
            proj_lower = proj_upper = None
        project_actual = (
            sum(w*a["physical_progress_estimate"] for w,a in zip(project_weights, activities)) / total_w
            if all(a["physical_progress_estimate"] is not None for a in activities) else None
        )
        project_planned = (
            sum(w*a["planned_percent_complete"] for w,a in zip(project_weights, activities)) / total_w
            if all(a["planned_percent_complete"] is not None for a in activities) else None
        )
    else:
        proj_lower = proj_upper = project_actual = project_planned = None
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "data_date_utc": data_date.isoformat(timespec="seconds"),
        "status": "verified" if project_actual is not None and project_planned is not None else "unknown",
        "units": {"element_progress": "fraction_0_1", "activity_and_project_percent": "percent_0_100"},
        "uncertainty_method": "deterministic_partial_identification_not_wilson_CI",
        "legacy_warning": "v1 actual_percent_complete is a status-acceptance proxy, NOT validated physical progress",
        "n_activities": len(activities),
        "overall": {
            "physical_progress_estimate": project_actual,
            "physical_progress_lower_bound_pct": proj_lower,
            "physical_progress_upper_bound_pct": proj_upper,
            "planned_percent_complete": project_planned,
            "schedule_variance_percent": (
                project_actual-project_planned if project_actual is not None and project_planned is not None else None
            ),
            "schedule_variance_status": _classify(project_planned, project_actual, on_schedule_band_pct),
            "activity_weights_approved": valid_project_weights,
        },
        "activity_status_counts": dict(Counter(a["schedule_variance_status"] for a in activities)),
        "activities": activities,
    }
