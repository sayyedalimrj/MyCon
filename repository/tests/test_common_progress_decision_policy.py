from __future__ import annotations

from pipeline.common.progress_decision_policy import decide_element_progress


def test_low_registration_blocks_acceptance_even_with_good_metrics() -> None:
    decision = decide_element_progress(
        {
            "coverage": "1.0",
            "in_tolerance_ratio": "1.0",
            "confidence": "1.0",
            "status": "ok",
        },
        registration_confidence="low",
    )

    assert decision.acceptable is False
    assert decision.completion_state == "uncertain_low_registration"
    assert decision.evidence_status == "blocked_by_registration"
    assert "registration_confidence_low" in decision.risks


def test_missing_element_metrics_are_not_evidenced() -> None:
    decision = decide_element_progress(None, registration_confidence="high")

    assert decision.acceptable is False
    assert decision.completion_state == "not_evidenced"
    assert "metric_element_metrics:not_found" in decision.risks


def test_high_quality_metrics_can_be_accepted() -> None:
    decision = decide_element_progress(
        {
            "coverage": "0.9",
            "in_tolerance_ratio": "0.92",
            "confidence": "0.95",
            "status": "ok",
        },
        registration_confidence="high",
    )

    assert decision.acceptable is True
    assert decision.completion_state == "completed"
    assert decision.evidence_status == "metric_evidence_sufficient"


def test_low_coverage_blocks_acceptance() -> None:
    decision = decide_element_progress(
        {
            "coverage": "0.1",
            "in_tolerance_ratio": "0.95",
            "confidence": "0.95",
            "status": "ok",
        },
        registration_confidence="high",
    )

    assert decision.acceptable is False
    assert decision.completion_state == "not_evidenced"
    assert any("element_coverage_below_threshold" in risk for risk in decision.risks)
