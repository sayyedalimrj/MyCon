from __future__ import annotations

from typing import Any


VISIBLE_LABELS = {"high", "medium", "visible", "evaluated"}
LOW_VISIBILITY_LABELS = {"low", "occluded", "poor", "blocked"}
UNKNOWN_VISIBILITY_LABELS = {"", "unknown", "none", "null", "nan", "not_evaluated", "not evaluated"}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_visibility_confidence(value: Any) -> str:
    label = _norm(value)

    if label in UNKNOWN_VISIBILITY_LABELS:
        return "not_evaluated"

    if label in VISIBLE_LABELS:
        return "high" if label == "visible" or label == "evaluated" else label

    if label in LOW_VISIBILITY_LABELS:
        return "low"

    numeric = _float_or_none(value)
    if numeric is not None:
        if numeric >= 0.75:
            return "high"
        if numeric >= 0.45:
            return "medium"
        return "low"

    return "unknown"


def interpret_element_visibility(
    row: dict[str, Any],
    *,
    min_visible_low_coverage: float = 0.05,
    min_partial_coverage: float = 0.65,
) -> dict[str, str]:
    """Add a conservative visibility-aware interpretation.

    This deliberately avoids saying "not built" unless stronger project
    evidence is available. Low observed coverage in a visible area is reported
    as "not_observed_in_visible_area", not as definitive absence.
    """
    raw_visibility = (
        row.get("visibility_confidence")
        or row.get("visibility")
        or row.get("visibility_score")
        or row.get("visible_surface_confidence")
    )

    visibility = normalize_visibility_confidence(raw_visibility)
    completion_state = _norm(row.get("completion_state"))
    acceptable = _norm(row.get("acceptable")) == "true"
    evidence_status = _norm(row.get("evidence_status"))
    coverage = _float_or_none(row.get("observed_surface_ratio", row.get("coverage")))

    risks: list[str] = []
    visibility_status = "not_evaluated"
    construction_state = "not_evidenced"
    visibility_evidence_status = "visibility_not_evaluated"

    if visibility in {"high", "medium"}:
        visibility_status = "visible"
    elif visibility == "low":
        visibility_status = "low_visibility"
        risks.append("visibility_low_or_occluded")
    else:
        visibility_status = "not_evaluated"
        risks.append("visibility_not_evaluated")

    if completion_state == "uncertain_low_registration" or "registration" in evidence_status:
        construction_state = "uncertain_low_registration"
        visibility_evidence_status = "blocked_by_registration"

    elif acceptable:
        if visibility_status == "not_evaluated":
            construction_state = "completed_pending_visibility_check"
            visibility_evidence_status = "metric_evidence_sufficient_visibility_missing"
            risks.append("visibility_not_evaluated_for_completed_element")
        elif visibility_status == "low_visibility":
            construction_state = "completed_metric_candidate_low_visibility"
            visibility_evidence_status = "metric_evidence_sufficient_low_visibility"
            risks.append("visibility_low_for_completed_metric_candidate")
        else:
            construction_state = "completed"
            visibility_evidence_status = "metric_evidence_sufficient"

    elif coverage is None:
        construction_state = "not_evidenced"
        visibility_evidence_status = "coverage_missing"

    elif visibility_status == "visible" and coverage < min_visible_low_coverage:
        construction_state = "not_observed_in_visible_area"
        visibility_evidence_status = "visible_area_low_or_zero_coverage"
        risks.append(
            f"visible_area_coverage_below_observation_threshold:{coverage:.6f}<{min_visible_low_coverage:.6f}"
        )

    elif visibility_status == "visible" and coverage < min_partial_coverage:
        construction_state = "partial_or_deviated"
        visibility_evidence_status = "visible_area_partial_metric_evidence"
        risks.append(
            f"visible_area_coverage_below_completion_threshold:{coverage:.6f}<{min_partial_coverage:.6f}"
        )

    else:
        construction_state = "not_evidenced"
        visibility_evidence_status = "insufficient_visibility_or_metric_evidence"

    return {
        "visibility_confidence": visibility,
        "visibility_status": visibility_status,
        "visibility_evidence_status": visibility_evidence_status,
        "construction_state_interpretation": construction_state,
        "visibility_decision_risks": ";".join(risks),
    }
