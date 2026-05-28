from __future__ import annotations

from pipeline.stage_10_copilot.answer_validator import validate_copilot_answer_payload


def test_validator_rejects_high_confidence_low_registration_risk() -> None:
    result = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: This element is accepted as completed.",
            "confidence": "high",
            "provider": "ollama_local",
            "evidence_used": ["evidence.json"],
            "risks_or_uncertainty": ["registration_confidence_low"],
        }
    )

    assert result.passed is False
    assert "high_confidence_with_low_quality_risks" in result.failures


def test_validator_allows_conservative_low_confidence_refusal() -> None:
    result = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: Do not accept this element because registration confidence is low.",
            "confidence": "low",
            "provider": "mock",
            "evidence_used": ["evidence.json"],
            "risks_or_uncertainty": ["registration_confidence_low"],
        }
    )

    assert result.passed is True


def test_validator_rejects_missing_evidence() -> None:
    result = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: The current evidence is insufficient.",
            "confidence": "low",
            "provider": "mock",
            "evidence_used": [],
            "risks_or_uncertainty": [],
        }
    )

    assert result.passed is False
    assert "missing_evidence_used" in result.failures


def test_validator_rejects_model_reported_confidence_with_low_quality_risk() -> None:
    result = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: The element can be accepted as complete.",
            "confidence": "model_reported",
            "provider": "ollama_local",
            "evidence_used": ["evidence.json"],
            "risks_or_uncertainty": ["registration_confidence_low"],
        }
    )

    assert result.passed is False
    assert "unverified_confidence_with_low_quality_risks" in result.failures
