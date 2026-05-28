from __future__ import annotations

from pipeline.common.progress_decision_policy import decide_element_progress
from pipeline.stage_10_copilot.answer_validator import validate_copilot_answer_payload


def test_stage8_low_confidence_forces_stage9_uncertain_and_stage10_refusal() -> None:
    stage9_decision = decide_element_progress(
        {
            "global_id": "E1",
            "coverage": "1.0",
            "in_tolerance_ratio": "1.0",
            "confidence": "1.0",
            "status": "ok",
        },
        registration_confidence="low",
    )

    assert stage9_decision.acceptable is False
    assert stage9_decision.completion_state == "uncertain_low_registration"

    bad_stage10 = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: This element is accepted as completed.",
            "confidence": "high",
            "provider": "ollama_local",
            "evidence_used": ["latest_evidence_package.json"],
            "risks_or_uncertainty": stage9_decision.risks,
        }
    )

    assert bad_stage10.passed is False
    assert "high_confidence_with_low_quality_risks" in bad_stage10.failures

    safe_stage10 = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: Do not accept this element because registration confidence is low.",
            "confidence": "low",
            "provider": "ollama_local",
            "evidence_used": ["latest_evidence_package.json"],
            "risks_or_uncertainty": stage9_decision.risks,
        }
    )

    assert safe_stage10.passed is True
