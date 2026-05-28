from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from pipeline.common.progress_decision_policy import combine_decision_risks


@dataclass(frozen=True)
class CopilotAnswerValidation:
    status: str
    passed: bool
    failures: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOW_QUALITY_RISK_TOKENS = (
    "registration_confidence_low",
    "low_registration",
    "icp_fitness_below",
    "metric_element_metrics:not_found",
    "element_status_not_acceptable",
    "element_coverage_below_threshold",
    "element_in_tolerance_below_threshold",
    "element_confidence_below_threshold",
    "activity_status_risk:uncertain",
    "answer_validation_failed",
)

ACCEPTANCE_PATTERN = re.compile(
    r"\b(accept|accepted|approve|approved|complete|completed|completion|pass|passed)\b",
    re.IGNORECASE,
)

REFUSAL_PATTERN = re.compile(
    r"\b(no|not|do not|don't|cannot|can't|insufficient|not enough|should not|must not|reject|rejected)\b",
    re.IGNORECASE,
)


def _lower_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _normalize_confidence(value: Any) -> str:
    raw = _lower_text(value).strip()
    if raw in {"high", "medium", "low", "low_to_medium"}:
        return raw
    if raw in {"model_reported", "unknown", "", "n/a", "none"}:
        return "unverified"
    return raw


def _has_low_quality_risk(risks: list[str]) -> bool:
    risk_text = " ".join(_lower_text(item) for item in risks)
    return any(token in risk_text for token in LOW_QUALITY_RISK_TOKENS)


def _claims_acceptance(answer: str) -> bool:
    return bool(ACCEPTANCE_PATTERN.search(answer))


def _is_refusal_or_caution(answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(answer))


def validate_copilot_answer_payload(payload: dict[str, Any]) -> CopilotAnswerValidation:
    """Validate Stage 10 response against evidence-only policy.

    This is a deterministic post-check for mock, Ollama, or OpenAI-compatible
    local VLM responses. It prevents a real VLM from overriding deterministic
    Stage 8/9 quality gates.
    """
    failures: list[str] = []
    warnings: list[str] = []

    answer = _lower_text(payload.get("answer"))
    confidence = _normalize_confidence(payload.get("confidence"))
    provider = _lower_text(payload.get("provider"))
    evidence = payload.get("evidence_used") or []
    risks = combine_decision_risks(payload.get("risks_or_uncertainty"))

    has_low_risk = _has_low_quality_risk(risks)
    claims_acceptance = _claims_acceptance(answer)
    is_refusal = _is_refusal_or_caution(answer)

    if not evidence:
        failures.append("missing_evidence_used")

    if confidence == "high" and has_low_risk:
        failures.append("high_confidence_with_low_quality_risks")
    elif confidence not in {"low", "low_to_medium", "medium"} and has_low_risk:
        # B1: was previously a chain of three identical `elif` clauses; only the
        # first could ever fire. Collapsed into a single branch.
        failures.append("unverified_confidence_with_low_quality_risks")

    if claims_acceptance and has_low_risk and not is_refusal:
        failures.append("acceptance_claim_with_low_quality_risks")

    if provider in {"", "unknown"}:
        warnings.append("missing_or_unknown_provider")

    if "direct answer" not in answer:
        warnings.append("answer_missing_direct_answer_section")

    return CopilotAnswerValidation(
        status="pass" if not failures else "fail",
        passed=not failures,
        failures=failures,
        warnings=warnings,
    )
