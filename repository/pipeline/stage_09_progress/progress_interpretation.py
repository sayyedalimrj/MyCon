from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def infer_visibility_confidence(
    *,
    point_count_evaluated: int,
    observed_surface_ratio: float | None,
    min_points_low: int = 50,
    min_points_high: int = 500,
) -> float:
    """Estimate whether the element is sufficiently observed to interpret progress.

    This is not a geometric visibility solver. It is a conservative evidence
    gate that prevents "not observed" from being interpreted as "not built".
    """
    if point_count_evaluated <= 0:
        return 0.0

    ratio = observed_surface_ratio if observed_surface_ratio is not None else 0.0

    point_score = min(1.0, max(0.0, point_count_evaluated / float(min_points_high)))
    if point_count_evaluated < min_points_low:
        point_score *= 0.5

    ratio_score = min(1.0, max(0.0, ratio / 0.20))
    return max(0.0, min(1.0, 0.5 * point_score + 0.5 * ratio_score))


def infer_completion_state(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    coverage = _safe_float(row.get("coverage"), 0.0)
    in_tolerance = _safe_float(row.get("in_tolerance_ratio"), coverage)
    confidence = _safe_float(row.get("confidence"), 0.0)
    point_count = _safe_int(row.get("point_count_evaluated"), 0)

    raw_status = str(row.get("status", "")).strip().lower()
    registration_confidence = str(row.get("registration_confidence", "")).strip().lower()

    observed_surface_ratio = coverage
    visibility_conf = infer_visibility_confidence(
        point_count_evaluated=point_count,
        observed_surface_ratio=observed_surface_ratio,
    )

    notes: list[str] = []

    if registration_confidence == "low" or "low_registration" in raw_status:
        notes.append("registration_low_progress_interpretation_blocked")
        return "uncertain_low_registration", "uncertain", notes

    if visibility_conf < 0.20:
        notes.append("not_enough_observed_surface_to_infer_progress")
        return "not_evidenced", "not_evidenced", notes

    if coverage is None or in_tolerance is None:
        notes.append("missing_coverage_or_tolerance")
        return "uncertain_missing_metrics", "uncertain", notes

    if coverage >= 0.65 and in_tolerance >= 0.65 and (confidence or 0.0) >= 0.50:
        notes.append("candidate_only_requires_project_acceptance_review")
        return "complete_candidate", "candidate_complete", notes

    if coverage >= 0.20:
        notes.append("partial_evidence_observed")
        return "partial_candidate", "partial", notes

    notes.append("low_coverage_but_visible")
    return "insufficient_coverage", "partial_or_not_built_uncertain", notes


def upgrade_element_row(row: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(row)

    coverage = _safe_float(row.get("coverage"), 0.0)
    point_count = _safe_int(row.get("point_count_evaluated"), 0)
    visibility_conf = infer_visibility_confidence(
        point_count_evaluated=point_count,
        observed_surface_ratio=coverage,
    )
    completion_state, evidence_state, notes = infer_completion_state(row)

    existing_notes = str(upgraded.get("notes", "")).strip()
    joined_notes = ";".join(item for item in [existing_notes, *notes] if item)

    upgraded["observed_surface_ratio"] = _fmt(coverage)
    upgraded["visibility_confidence"] = _fmt(visibility_conf)
    upgraded["completion_state"] = completion_state
    upgraded["evidence_state"] = evidence_state
    upgraded["metric_truth_source"] = "stage8_registration_and_stage9_deterministic_metrics"
    upgraded["interpretation_notes"] = joined_notes
    return upgraded


def upgrade_activity_row(row: dict[str, Any], element_rows: list[dict[str, Any]]) -> dict[str, Any]:
    upgraded = dict(row)
    activity_id = str(row.get("activity_id", "")).strip()

    related = [r for r in element_rows if str(r.get("activity_id", "")).strip() == activity_id]

    if not related:
        upgraded["activity_completion_state"] = "not_evidenced"
        upgraded["activity_visibility_confidence"] = "0.000000"
        upgraded["activity_interpretation_notes"] = "no_related_element_metrics"
        return upgraded

    visibility_values = [_safe_float(r.get("visibility_confidence"), 0.0) or 0.0 for r in related]
    states = [str(r.get("completion_state", "")) for r in related]
    visibility = sum(visibility_values) / max(1, len(visibility_values))

    if any("uncertain_low_registration" == s for s in states):
        state = "uncertain_low_registration"
        notes = "blocked_by_low_registration"
    elif all(s == "complete_candidate" for s in states):
        state = "complete_candidate"
        notes = "all_related_elements_complete_candidates"
    elif any(s in {"partial_candidate", "insufficient_coverage"} for s in states):
        state = "partial_candidate"
        notes = "some_related_elements_partial_or_low_coverage"
    elif all(s == "not_evidenced" for s in states):
        state = "not_evidenced"
        notes = "related_elements_not_sufficiently_observed"
    else:
        state = "uncertain"
        notes = "mixed_or_uncertain_related_element_states"

    upgraded["activity_completion_state"] = state
    upgraded["activity_visibility_confidence"] = _fmt(visibility)
    upgraded["activity_interpretation_notes"] = notes
    return upgraded


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upgrade_progress_outputs(
    *,
    element_metrics_csv: Path,
    activity_progress_csv: Path,
    output_element_csv: Path,
    output_activity_csv: Path,
    output_summary_json: Path,
) -> dict[str, Any]:
    element_rows = [upgrade_element_row(row) for row in _read_csv(element_metrics_csv)]
    activity_rows = [
        upgrade_activity_row(row, element_rows)
        for row in _read_csv(activity_progress_csv)
    ]

    _write_csv(output_element_csv, element_rows)
    _write_csv(output_activity_csv, activity_rows)

    summary = {
        "status": "complete",
        "element_count": len(element_rows),
        "activity_count": len(activity_rows),
        "completion_state_counts": {},
        "notes": [
            "This is a conservative interpretation layer.",
            "not_evidenced means the element was not sufficiently observed; it does not mean not built.",
            "Stage 8 registration confidence remains authoritative for progress interpretation.",
        ],
        "outputs": {
            "element_interpretation_csv": str(output_element_csv),
            "activity_interpretation_csv": str(output_activity_csv),
        },
    }

    counts: dict[str, int] = {}
    for row in element_rows:
        state = str(row.get("completion_state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    summary["completion_state_counts"] = counts

    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
