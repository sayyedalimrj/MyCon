from __future__ import annotations

from pipeline.stage_09_progress.visibility_policy import (
    interpret_element_visibility,
    normalize_visibility_confidence,
)


def test_visibility_confidence_normalization() -> None:
    assert normalize_visibility_confidence(None) == "not_evaluated"
    assert normalize_visibility_confidence("visible") == "high"
    assert normalize_visibility_confidence("occluded") == "low"
    assert normalize_visibility_confidence("0.9") == "high"
    assert normalize_visibility_confidence("0.2") == "low"


def test_low_registration_stays_uncertain() -> None:
    result = interpret_element_visibility(
        {
            "completion_state": "uncertain_low_registration",
            "evidence_status": "blocked_by_registration",
            "acceptable": "false",
            "coverage": "1.0",
            "visibility_confidence": "high",
        }
    )

    assert result["construction_state_interpretation"] == "uncertain_low_registration"
    assert result["visibility_evidence_status"] == "blocked_by_registration"


def test_unknown_visibility_low_coverage_is_not_evidenced_not_not_built() -> None:
    result = interpret_element_visibility(
        {
            "completion_state": "not_evidenced",
            "evidence_status": "insufficient_metric_evidence",
            "acceptable": "false",
            "coverage": "0.0",
        }
    )

    assert result["visibility_confidence"] == "not_evaluated"
    assert result["construction_state_interpretation"] == "not_evidenced"
    assert "visibility_not_evaluated" in result["visibility_decision_risks"]


def test_visible_low_coverage_is_not_observed_in_visible_area() -> None:
    result = interpret_element_visibility(
        {
            "completion_state": "not_evidenced",
            "evidence_status": "insufficient_metric_evidence",
            "acceptable": "false",
            "coverage": "0.0",
            "visibility_confidence": "high",
        }
    )

    assert result["construction_state_interpretation"] == "not_observed_in_visible_area"
    assert result["visibility_evidence_status"] == "visible_area_low_or_zero_coverage"


def test_visible_partial_coverage_is_partial_or_deviated() -> None:
    result = interpret_element_visibility(
        {
            "completion_state": "not_evidenced",
            "evidence_status": "insufficient_metric_evidence",
            "acceptable": "false",
            "coverage": "0.2",
            "visibility_confidence": "high",
        }
    )

    assert result["construction_state_interpretation"] == "partial_or_deviated"
    assert result["visibility_evidence_status"] == "visible_area_partial_metric_evidence"


def test_acceptable_without_visibility_stays_pending_visibility_check() -> None:
    result = interpret_element_visibility(
        {
            "completion_state": "complete_candidate",
            "evidence_status": "metric_evidence_sufficient",
            "acceptable": "true",
            "coverage": "0.95",
        }
    )

    assert result["construction_state_interpretation"] == "completed_pending_visibility_check"
    assert result["visibility_evidence_status"] == "metric_evidence_sufficient_visibility_missing"
    assert "visibility_not_evaluated_for_completed_element" in result["visibility_decision_risks"]
