from __future__ import annotations

from pipeline.common.progress_decision_policy import decide_element_progress
from pipeline.stage_10_copilot.answer_validator import validate_copilot_answer_payload


def main() -> int:
    low_reg_decision = decide_element_progress(
        {
            "coverage": "1.0",
            "in_tolerance_ratio": "1.0",
            "confidence": "1.0",
            "status": "ok",
        },
        registration_confidence="low",
    )

    if low_reg_decision.acceptable:
        print("DECISION_GATE_FAIL low registration accepted")
        return 1

    bad_answer = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: This element is accepted as completed.",
            "confidence": "high",
            "provider": "ollama_local",
            "evidence_used": ["evidence.json"],
            "risks_or_uncertainty": low_reg_decision.risks,
        }
    )

    if bad_answer.passed:
        print("DECISION_GATE_FAIL copilot accepted low registration")
        return 1

    conservative_answer = validate_copilot_answer_payload(
        {
            "answer": "Direct answer: Do not accept this element because registration confidence is low.",
            "confidence": "low",
            "provider": "mock",
            "evidence_used": ["evidence.json"],
            "risks_or_uncertainty": low_reg_decision.risks,
        }
    )

    if not conservative_answer.passed:
        print("DECISION_GATE_FAIL conservative refusal rejected")
        return 1

    print("DECISION_GATES_OK")
    print("low_registration_state:", low_reg_decision.completion_state)
    print("low_registration_risks:", ",".join(low_reg_decision.risks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
