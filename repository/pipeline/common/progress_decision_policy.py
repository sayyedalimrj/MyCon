from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LOW_REGISTRATION_LABELS = {
    "low",
    "very_low",
    "failed",
    "fail",
    "unknown",
    "skipped",
    "skipped_insufficient_anchors",
    "uncertain_low_registration",
}

UNCERTAIN_STATUS_TOKENS = (
    "uncertain",
    "low_registration",
    "not_evidenced",
    "missing",
    "failed",
    "not_found",
)


@dataclass(frozen=True)
class ProgressDecision:
    completion_state: str
    evidence_status: str
    acceptable: bool
    confidence: str
    risks: list[str]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def is_low_registration_confidence(value: Any) -> bool:
    label = _norm(value)
    if not label:
        return True
    if label in LOW_REGISTRATION_LABELS:
        return True
    return any(token in label for token in ("low", "failed", "uncertain"))


def decide_element_progress(
    element_metrics: dict[str, Any] | None,
    *,
    registration_confidence: Any = None,
    min_coverage: float = 0.65,
    min_in_tolerance: float = 0.65,
    min_element_confidence: float = 0.65,
) -> ProgressDecision:
    """Conservative element decision policy.

    Stage 8/9/10 must never accept an element when registration is weak,
    element evidence is missing, or coverage/tolerance/confidence is below
    acceptance thresholds.
    """
    risks: list[str] = []

    if is_low_registration_confidence(registration_confidence):
        risks.append("registration_confidence_low")
        return ProgressDecision(
            completion_state="uncertain_low_registration",
            evidence_status="blocked_by_registration",
            acceptable=False,
            confidence="low",
            risks=risks,
            recommended_action="Improve metric alignment/registration before accepting element completion.",
        )

    if not element_metrics:
        risks.append("metric_element_metrics:not_found")
        return ProgressDecision(
            completion_state="not_evidenced",
            evidence_status="missing_element_metrics",
            acceptable=False,
            confidence="low",
            risks=risks,
            recommended_action="Provide element metrics before accepting element completion.",
        )

    status = _norm(element_metrics.get("status") or element_metrics.get("completion_state"))
    if any(token in status for token in UNCERTAIN_STATUS_TOKENS):
        risks.append(f"element_status_not_acceptable:{status or 'unknown'}")
        return ProgressDecision(
            completion_state=status or "not_evidenced",
            evidence_status="blocked_by_element_status",
            acceptable=False,
            confidence="low",
            risks=risks,
            recommended_action="Do not accept this element until deterministic metrics become acceptable.",
        )

    coverage = _float_or_none(
        element_metrics.get("observed_surface_ratio", element_metrics.get("coverage"))
    )
    in_tolerance = _float_or_none(element_metrics.get("in_tolerance_ratio"))
    element_confidence = _float_or_none(element_metrics.get("confidence"))

    if coverage is None:
        risks.append("element_coverage_missing")
    elif coverage < min_coverage:
        risks.append(f"element_coverage_below_threshold:{coverage:.6f}<{min_coverage:.6f}")

    if in_tolerance is None:
        risks.append("element_in_tolerance_ratio_missing")
    elif in_tolerance < min_in_tolerance:
        risks.append(
            f"element_in_tolerance_below_threshold:{in_tolerance:.6f}<{min_in_tolerance:.6f}"
        )

    if element_confidence is None:
        risks.append("element_confidence_missing")
    elif element_confidence < min_element_confidence:
        risks.append(
            f"element_confidence_below_threshold:{element_confidence:.6f}<{min_element_confidence:.6f}"
        )

    if risks:
        return ProgressDecision(
            completion_state="not_evidenced",
            evidence_status="insufficient_metric_evidence",
            acceptable=False,
            confidence="low",
            risks=risks,
            recommended_action="Do not accept this element until coverage, tolerance, and confidence pass thresholds.",
        )

    return ProgressDecision(
        completion_state="completed",
        evidence_status="metric_evidence_sufficient",
        acceptable=True,
        confidence="high",
        risks=[],
        recommended_action="Element can be considered acceptable from deterministic metrics, subject to project QA review.",
    )


def combine_decision_risks(*items: Any) -> list[str]:
    risks: list[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            risks.append(item)
        elif isinstance(item, (list, tuple, set)):
            risks.extend(str(x) for x in item)
        elif isinstance(item, dict):
            risks.extend(str(x) for x in item.get("risks", []) or [])
    return risks
