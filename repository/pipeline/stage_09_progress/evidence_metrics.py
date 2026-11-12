"""Scientific Stage 9 fields; scan coverage is NOT construction percentage.

All ratios use [0,1]. Empty CSV fields mean not estimable. No source for
visibility or independently measured quantities => unknown, never 0%.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

APPROVED_QUANTITY_SOURCES = {"approved_measurement", "signed_site_log", "survey_qa"}
AUDITED_VISIBILITY_SOURCES = {"calibrated_camera_frustum", "raycast_zbuffer", "manual_occlusion_audit"}
CALIBRATED_CONFIDENCE_SOURCES = {"heldout_calibration", "audited_quality_model"}


def finite_number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def unit_ratio(value: Any) -> float | None:
    number = finite_number(value)
    return number if number is not None and 0 <= number <= 1 else None


def element_scientific_fields(
    row: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Keep physical quantity, observable design surface, and evidence quality separate.

    Quantity: verified executed quantity / planned quantity of the same unit.
    Observability: independently audited visible/eligible design surface.
    Confidence: independently calibrated quality index, *not* a Wilson CI.
    None means undefined, not zero. Registration errors/overlap block
    any geometry-only interpretation, even when legacy coverage is large.
    """
    meta = metadata or {}
    def get(key: str) -> Any:
        value = row.get(key)
        return value if value not in (None, "") else meta.get(key)

    registration = str(get("registration_confidence") or "").lower().strip()
    synthetic = str(get("synthetic_bim_fallback_used") or "").lower() in {"true", "1", "yes"}
    ambiguous = finite_number(get("ambiguous_scan_point_count")) or 0
    assigned = finite_number(get("assigned_scan_point_count")) or 0

    visibility_src = str(get("observability_source") or "").strip().lower()
    observability = unit_ratio(get("observability_ratio")) if visibility_src in AUDITED_VISIBILITY_SOURCES else None
    # Only independent calibration is defensible as a confidence indicator.
    confidence_src = str(get("evidence_confidence_source") or "").strip().lower()
    confidence = unit_ratio(get("evidence_confidence")) if confidence_src in CALIBRATED_CONFIDENCE_SOURCES else None
    if registration not in {"high", "medium"} or synthetic:
        confidence = None

    qsrc = str(get("quantity_source") or "").strip().lower()
    planned = finite_number(get("planned_quantity"))
    executed = finite_number(get("executed_quantity"))
    verified = str(get("quantity_verified") or "").lower() in {"true", "yes", "1"}
    valid_quantity = (
        verified and qsrc in APPROVED_QUANTITY_SOURCES
        and planned is not None and planned > 0 and executed is not None
        and 0 <= executed <= planned
        and bool(str(get("quantity_unit") or "").strip())
    )
    physical = executed / planned if valid_quantity else None

    if registration == "low" or synthetic:
        state = "uncertain_registration"
    elif ambiguous > 0 and assigned <= 0:
        state = "unknown_overlap"
    elif observability is None:
        state = "unknown_observability"
    elif observability == 0:
        state = "occluded"
    elif assigned <= 0:
        state = "not_observed"
    else:
        coverage = unit_ratio(get("coverage"))
        tolerance = unit_ratio(get("in_tolerance_ratio"))
        if coverage is None or tolerance is None:
            state = "unknown_geometry"
        elif coverage < 0.65 or tolerance < 0.65:
            state = "partial_or_geometric_nonconformance"
        else:
            state = "geometry_consistent_candidate"
    # Independently verified quantities remain valid in a low-registration scan,
    # but this prevents treating geometry as verified execution.
    return {
        "physical_progress_estimate": physical,
        "observability_ratio": observability,
        "evidence_confidence": confidence,
        "physical_progress_status": "verified_quantity" if physical is not None else "unknown",
        "geometry_evidence_state": state,
        "scientific_metrics_schema": "mycon.element_progress.v2",
    }
