"""Auditable Stage 09 construction states and independent held-out evaluation.

No scan absence, geometry agreement or legacy acceptance score proves 'not built'.
Missing labels are abstentions, not negative classifications.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from .evidence_metrics import as_utc, element_scientific_fields, finite_number, valid_sha256

SCHEMA = "mycon.element_status.v1"
CLASSES = ("built", "not_built", "in_progress")


def status_for_element(row: Mapping[str, Any], *, data_date: dt.datetime,
                       mapped: bool, unique: bool) -> dict[str, Any]:
    scientific = element_scientific_fields(row, data_date=data_date, require_audit=True)
    fraction = scientific["physical_progress_estimate"] if unique else None
    reason = []
    if not mapped:
        reason.append("ifc_id_not_in_approved_activity_mapping")
    if not unique:
        reason.append("duplicate_or_missing_ifc_global_id")
    if fraction is None:
        reason.extend(filter(None, scientific["physical_progress_reason_codes"].split(";")))
        state = "unknown"
    elif math.isclose(fraction, 1.0, abs_tol=1e-12):
        state = "built"  # verified full measured quantity, not quality acceptance
    elif fraction > 0:
        state = "in_progress"
    else:
        # A signed measurement of zero is a numeric quantity, but independently
        # audited absence is required before asserting the site is unbuilt.
        absence = str(row.get("absence_verified") or "").lower() in {"true", "yes", "1"}
        absence_ref = str(row.get("absence_evidence_ref") or "").strip()
        absence_hash = row.get("absence_document_sha256")
        absence_reviewer = str(row.get("absence_approved_by") or "").strip()
        state = "not_built" if absence and absence_ref and valid_sha256(absence_hash) and absence_reviewer else "unknown"
        if state == "unknown":
            reason.append("zero_quantity_without_independent_absence_audit")
    geometry_valid = str(row.get("geometry_adapter_valid") or "").lower() in {"true", "yes", "1"}
    geometry_state = scientific["geometry_evidence_state"] if geometry_valid else "unknown_unverified_geometry"
    occlusion = ("occluded" if geometry_state == "occluded" else
                 "out_of_fov" if geometry_state == "out_of_fov" else
                 "unknown")
    audit = scientific["quantity_audit"]
    return {
        "schema_version": SCHEMA,
        "global_id": str(row.get("global_id") or "").strip(),
        "construction_status": state,
        "physical_progress_estimate": fraction,
        "quantity_unit": audit["quantity_unit"],
        "quantity_audit_gate": "pass" if fraction is not None else "blocked",
        "quantity_document_sha256": audit["quantity_document_sha256"],
        "quantity_record_id": audit["quantity_record_id"],
        "geometry_evidence_state": geometry_state,
        "geometry_manifest_sha256": row.get("geometry_manifest_sha256") if geometry_valid else None,
        "occlusion_state": occlusion,
        "observability_ratio": scientific["observability_ratio"] if geometry_valid else None,
        "evidence_confidence": scientific["evidence_confidence"] if geometry_valid else None,
        "coordinate_frame": "bim_world" if geometry_valid else "unknown",
        "coordinate_units": "m" if geometry_valid else "unknown",
        "reason_codes": ";".join(sorted(set(reason))),
        "evidence_types": "approved_quantity" if fraction is not None else "insufficient",
    }


def _validate_truth_rows(rows: Iterable[Mapping[str, Any]], *,
                         id_field: str, value_field: str,
                         data_date: dt.datetime) -> list[dict[str, Any]]:
    found: set[str] = set()
    groups: dict[str, set[str]] = {}
    result = []
    for original in rows:
        row = dict(original)
        key = str(row.get(id_field) or "").strip()
        split = str(row.get("split") or "").strip().lower()
        site = str(row.get("site_id") or "").strip()
        session = str(row.get("capture_session_id") or "").strip()
        stamp = as_utc(row.get("verified_at_utc"))
        if (not key or key in found or split not in {"train", "test"}
                or not site or not session or not str(row.get("reviewer") or "").strip()
                or not valid_sha256(row.get("document_sha256"))
                or stamp is None or stamp.date() != data_date.date()
                or stamp > data_date):
            raise ValueError("invalid_or_non_independent_ground_truth_record:" + key)
        found.add(key)
        groups.setdefault(site, set()).add(split)
        if str(row.get(value_field) or "").strip():
            value = finite_number(row[value_field])
            limit = 100 if value_field == "expert_progress_pct" else 1
            if value is None or not 0 <= value <= limit:
                raise ValueError("invalid_ground_truth_fraction:" + key)
        result.append(row)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("train_test_site_leakage")
    return [row for row in result if str(row.get("split")).lower() == "test"]


def evaluate_element_truth(predictions: Iterable[Mapping[str, Any]],
                           truth_rows: Iterable[Mapping[str, Any]],
                           *, data_date: dt.datetime) -> dict[str, Any]:
    truth = _validate_truth_rows(truth_rows, id_field="global_id",
                                 value_field="ground_truth_fraction",
                                 data_date=data_date)
    pred = {str(r.get("global_id")): r for r in predictions}
    confusion = {k: Counter() for k in CLASSES}
    abs_errors = []
    abstentions = 0
    labeled = 0
    for row in truth:
        actual = str(row.get("ground_truth_status") or "").strip().lower()
        if actual and actual not in CLASSES:
            raise ValueError("invalid_ground_truth_status:" + actual)
        matched = pred.get(row["global_id"], {})
        predicted = str(matched.get("construction_status") or "unknown")
        if actual:
            confusion[actual][predicted] += 1
            labeled += 1
            abstentions += predicted not in CLASSES
        number = finite_number(row.get("ground_truth_fraction"))
        prediction = finite_number(matched.get("physical_progress_estimate"))
        if number is not None and prediction is not None:
            abs_errors.append(abs(number - prediction))
    class_scores = {}
    for name in CLASSES:
        tp = confusion[name][name]
        support = sum(confusion[name].values())
        predicted_count = sum(confusion[c][name] for c in CLASSES)
        precision = tp / predicted_count if predicted_count else (0.0 if support else None)
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              (0.0 if support else None))
        class_scores[name] = {
            "support": support, "precision": precision, "recall": recall, "f1": f1,
        }
    valid_f1 = [v["f1"] for v in class_scores.values() if v["f1"] is not None]
    return {
        "evaluation_status": "heldout_truth_provided",
        "ground_truth_rows_test": len(truth),
        "ground_truth_labeled": labeled,
        "unknown_or_missing_predictions": abstentions,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_class": class_scores,
        "macro_f1": sum(valid_f1) / len(valid_f1) if valid_f1 else None,
        "fraction_mae": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "fraction_mae_n": len(abs_errors),
        "unscored_fraction_count": sum(bool(str(r.get("ground_truth_fraction") or "").strip()) for r in truth) - len(abs_errors),
        "note": "Unknown predictions count as false negatives; numeric quantities scored separately. Independent GT required.",
    }


def evaluate_expert_progress(activities: Iterable[Mapping[str, Any]],
                             truth_rows: Iterable[Mapping[str, Any]],
                             *, data_date: dt.datetime) -> dict[str, Any]:
    truth = _validate_truth_rows(truth_rows, id_field="activity_id",
                                 value_field="expert_progress_pct",
                                 data_date=data_date)
    predictions = {str(a.get("activity_id")): a for a in activities}
    comparison = []
    for row in truth:
        value = finite_number(row.get("expert_progress_pct"))
        predicted = finite_number(predictions.get(row["activity_id"], {}).get("physical_progress_estimate"))
        comparison.append({
            "activity_id": row["activity_id"],
            "expert_percent": value,
            "model_percent": predicted,
            "absolute_error_pp": abs(predicted - value) if value is not None and predicted is not None else None,
        })
    valid = [x["absolute_error_pp"] for x in comparison if x["absolute_error_pp"] is not None]
    return {
        "evaluation_status": "independent_expert_report_provided",
        "test_activities": len(truth),
        "matched_known_estimates": len(valid),
        "mae_percentage_points": sum(valid) / len(valid) if valid else None,
        "comparisons": comparison,
    }
