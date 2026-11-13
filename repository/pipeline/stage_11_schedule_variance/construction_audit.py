"""Versioned audit payloads for a civil-engineering 4D BIM progress report.

No visual/point coverage can create a construction quantity. Camera SHA and
geometry references are pointers only; immutable image/IFC files stay untouched.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping, Sequence

SCHEMA = "mycon.progress_engineering_audit.v1"


def _json_refs(raw: Any) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


def engineering_audit(
    statuses: Sequence[Mapping[str, Any]],
    scientific: Mapping[str, Any],
    *, geometry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create audit-safe visibility, certified quantity and uncertainty views.

    Counts are audit counts, not inferred as-built percentages. Activity
    weight approval and WBS come from the separately verified v2 aggregator.
    """
    observation_rows = []
    quantity_rows = []
    uncertain_rows = []
    for source in statuses:
        gid = source["global_id"]
        state = source["construction_status"]
        observation = source.get("observation_status") or "unknown"
        images = _json_refs(source.get("image_evidence_refs_json"))
        observation_rows.append({
            "global_id": gid,
            "activity_ids": source.get("activity_ids") or "",
            "wbs_codes": source.get("wbs_codes") or "",
            "construction_status": state,
            "observation_status": observation,
            "occlusion_state": source.get("occlusion_state", "unknown"),
            "observability_ratio": source.get("observability_ratio"),
            "evidence_confidence": source.get("evidence_confidence"),
            "coordinate_frame": source.get("coordinate_frame", "unknown"),
            "units": source.get("coordinate_units", "unknown"),
            "geometry_manifest_sha256": source.get("geometry_manifest_sha256"),
            "source_image_refs": images,
            "geometry_support_sample_ids": source.get("geometry_support_sample_ids_json", "[]"),
            "observation_reason": source.get("observation_reason"),
        })
        quantity_rows.append({
            "global_id": gid,
            "construction_status": state,
            "approved_fraction": source.get("physical_progress_estimate"),
            "quantity_unit": source.get("quantity_unit"),
            "quantity_approval_gate": source.get("quantity_audit_gate"),
            "quantity_record_id": source.get("quantity_record_id"),
            "document_sha256": source.get("quantity_document_sha256"),
            "activity_ids": source.get("activity_ids"),
        })
        # Missing image visibility does not erase a separately signed
        # fraction. Every decision uncertainty is shown to the engineer.
        if state == "unknown" or observation in {"unknown", "not_observable"}:
            uncertain_rows.append({
                "global_id": gid,
                "construction_status": state,
                "observation_status": observation,
                "verified_fraction_or_null": source.get("physical_progress_estimate"),
                "reason_codes": source.get("reason_codes"),
                "observation_reason": source.get("observation_reason"),
                "manual_review_required": state == "unknown",
                "source_image_refs": images,
            })

    act = []
    for a in scientific["activities"]:
        act.append({
            "activity_id": a["activity_id"],
            "wbs_code": a["wbs_code"],
            "approved_weight_basis": a.get("mapping_weight_basis"),
            "approved_activity_weight": a.get("activity_weight_audit"),
            "approved_element_weights": [
                {
                    "ifc_global_id": e["ifc_global_id"],
                    "weight": e["mapping_weight"],
                    "approval": e["weight_audit"],
                    "physical_fraction": e["physical_progress_estimate"],
                } for e in a["elements"]
            ],
            "physical_percent": a["physical_progress_estimate"],
            "progress_bounds_percent": [
                a["physical_progress_lower_bound_pct"],
                a["physical_progress_upper_bound_pct"],
            ],
            "schedule_baseline": a["baseline_audit"],
            "planned_percent": a["planned_percent_complete"],
            "schedule_variance_pp": a["schedule_variance_percent"],
            "schedule_variance_bounds_pp": a["schedule_variance_bounds_pct"],
            "decision_quality_gate": a["schedule_decision_quality_gate"],
        })
    status_counts = Counter(r["observation_status"] for r in observation_rows)
    return {
        "engineering_audit_schema_version": SCHEMA,
        "observability_report": {
            "status_counts": dict(status_counts),
            "geometry_manifest_valid": geometry_summary.get("geometry_adapter_valid") is True,
            "metric_and_ifc_geometry_gate": (
                "pass" if geometry_summary.get("geometry_adapter_valid") is True else "blocked"
            ),
            "elements": observation_rows,
            "note": "Visibility states are independent of quantity-based construction statuses.",
        },
        "approved_quantity_report": {
            "approved_count": sum(x["quantity_approval_gate"] == "pass" for x in quantity_rows),
            "elements": quantity_rows,
            "note": "Fractions are certified only from approved dated measurements; no raw point-coverage converted to work quantity.",
        },
        "wbs_boq_weight_report": {
            "activities": act,
            "wbs_rollup": scientific["wbs"],
            "project": scientific["overall"],
            "note": "Project percent is null when compatible approved project weights or quantities are missing.",
        },
        "uncertain_elements_report": {
            "count": len(uncertain_rows),
            "elements": uncertain_rows,
            "note": "Nonobservable is NOT not built. Numeric signed quantities may remain known even if camera visibility is unknown.",
        },
        "manager_summary": {
            "physical_progress_pct": scientific["overall"]["physical_progress_estimate"],
            "progress_lower_bound_pct": scientific["overall"]["physical_progress_lower_bound_pct"],
            "progress_upper_bound_pct": scientific["overall"]["physical_progress_upper_bound_pct"],
            "planned_percent_complete": scientific["overall"]["planned_percent_complete"],
            "schedule_variance_pp": scientific["overall"]["schedule_variance_percent"],
            "schedule_status": scientific["overall"]["schedule_variance_status"],
            "decision_quality_gate": scientific["overall"]["schedule_decision_quality_gate"],
            "needs_human_review": len(uncertain_rows) > 0,
            "units": {
                "element_fraction": "0_to_1",
                "project_progress": "percent_0_to_100",
                "schedule_variance": "percentage_points",
                "visibility": "fraction_0_to_1_or_null",
            },
        },
    }
