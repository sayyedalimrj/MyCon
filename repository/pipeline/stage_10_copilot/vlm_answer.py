"""VLM answer generation for Stage 10.

The default provider is a deterministic mock to keep tests stable. HTTP providers
can be wired to Ollama, vLLM, llama.cpp server, or a custom Transformers endpoint.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any

from .config_access import cfg_get
from .evidence_builder import EvidencePackage
from .local_vlm_client import LocalVLMError, call_ollama_local, call_openai_compatible_local


@dataclass(frozen=True)
class CopilotAnswer:
    answer: str
    evidence_used: list[str]
    confidence: str
    recommended_action: str
    risks_or_uncertainty: list[str]
    provider: str
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_POLICY = """You are a Construction Copilot. Use only provided evidence.
Do not invent measurements. Distinguish visual observations from metric facts.
If evidence is insufficient, say which artifact is missing. Always return: direct
answer, evidence used, metric facts, visual observations, confidence, risks, and
recommended next action.
"""


def _metric_facts(package: EvidencePackage) -> list[str]:
    facts: list[str] = []
    element = package.metrics.get("element_metrics", {}).get("data", {})
    if element:
        for key in [
            "global_id",
            "name",
            "element_name",
            "ifc_class",
            "coverage",
            "in_tolerance_ratio",
            "mean_deviation_m",
            "median_deviation_m",
            "p95_deviation_m",
            "confidence",
            "status",
            "registration_confidence",
        ]:
            if key in element and element[key] not in {None, ""}:
                facts.append(f"{key}={element[key]}")

    activity = package.metrics.get("activity_progress", {}).get("data", {})
    if activity:
        for key in [
            "activity_id",
            "activity_name",
            "planned_percent",
            "observed_percent",
            "actual_percent",
            "confidence",
            "status",
            "delay_days",
        ]:
            if key in activity and activity[key] not in {None, ""}:
                facts.append(f"{key}={activity[key]}")

    registration = package.metrics.get("registration_quality", {}).get("data", {})
    if registration:
        for key in [
            "fitness",
            "rmse_m",
            "inlier_rmse",
            "confidence_label",
            "confidence_score",
            "registration_confidence",
        ]:
            if key in registration and registration[key] not in {None, ""}:
                facts.append(f"{key}={registration[key]}")
    return facts


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_risks(package: EvidencePackage) -> tuple[list[str], str, str]:
    """Return risk list, calibrated confidence, and acceptance recommendation.

    The mock provider must not become overconfident just because all files exist.
    Low registration confidence, uncertain element status, and low coverage are
    deterministic metric risks and must be reflected in the answer.
    """
    risks: list[str] = list(package.limitations)

    element = package.metrics.get("element_metrics", {}).get("data", {}) or {}
    activity = package.metrics.get("activity_progress", {}).get("data", {}) or {}
    registration = package.metrics.get("registration_quality", {}).get("data", {}) or {}

    coverage = _safe_float(element.get("coverage"))
    element_conf = _safe_float(element.get("confidence"))
    status = str(element.get("status", "")).strip()
    reg_label = str(
        registration.get("confidence_label")
        or registration.get("registration_confidence")
        or element.get("registration_confidence")
        or ""
    ).strip().lower()
    fitness = _safe_float(registration.get("fitness"))

    if reg_label == "low":
        risks.append("registration_confidence_low")
    if fitness is not None and fitness < 0.05:
        risks.append(f"icp_fitness_below_acceptance_threshold:{fitness:.6f}")
    if status and ("uncertain" in status or "low_registration" in status):
        risks.append(f"element_status_not_acceptable:{status}")
    if coverage is not None and coverage < 0.65:
        risks.append(f"element_coverage_below_threshold:{coverage:.6f}<0.650000")
    if element_conf is not None and element_conf < 0.65:
        risks.append(f"element_confidence_below_threshold:{element_conf:.6f}<0.650000")

    activity_status = str(activity.get("status", "")).strip()
    if activity_status and ("uncertain" in activity_status or "behind" in activity_status):
        risks.append(f"activity_status_risk:{activity_status}")

    risks = sorted(set(risks))

    severe = any(
        item.startswith("registration_confidence_low")
        or item.startswith("icp_fitness_below")
        or item.startswith("element_status_not_acceptable")
        for item in risks
    )
    if severe:
        confidence = "low"
        recommendation = (
            "Do not accept this element as complete from the current evidence. "
            "Use a real/project-specific IFC or improve metric alignment with anchors/primitives, "
            "then rerun Stage 8 and Stage 9."
        )
    elif risks:
        confidence = "medium"
        recommendation = "Review the evidence package and resolve listed risks before making an acceptance decision."
    else:
        confidence = "high"
        recommendation = "Evidence is sufficient for the mock copilot; verify against project acceptance criteria."

    return risks, confidence, recommendation

def _merge_metric_risks_for_model_answer(
    package: EvidencePackage,
    provided_risks: list[Any] | None = None,
) -> tuple[list[str], str, str]:
    metric_risks, calibrated_confidence, calibrated_action = _metric_risks(package)
    normalized_extra = [str(item) for item in (provided_risks or []) if str(item).strip()]
    risks = sorted(set(metric_risks + normalized_extra))
    if risks and calibrated_confidence == "high":
        calibrated_confidence = "medium"
    return risks, calibrated_confidence, calibrated_action


def _mock_answer(package: EvidencePackage) -> CopilotAnswer:
    facts = _metric_facts(package)
    risks, confidence, calibrated_action = _metric_risks(package)

    question = package.question.lower()
    asks_acceptance = any(token in question for token in ["accept", "accepted", "approve", "approval", "complete", "completed"])

    if facts and asks_acceptance and confidence == "low":
        direct = "No. The current evidence is not strong enough to accept this element as completed."
    elif facts:
        direct = "Based on the provided metric artifacts, the question can be answered with measurable evidence."
    elif risks:
        direct = "The system cannot give a final metric answer yet because some required evidence is missing or risky."
    else:
        direct = "The available evidence package has been prepared for this question."

    evidence_used = list(package.image_paths.values()) + [package.evidence_path]

    action = calibrated_action
    if any("pointcloud_missing" in item for item in risks):
        action = "Run Stage 5/6/7 on the target server to generate a valid cleaned or DA3-assisted point cloud, then ask again."
    if any("metric_element_metrics" in item for item in risks):
        action = "Run Stage 8/9 or provide element metrics CSV before accepting element-level completion decisions."

    sections = [
        f"Direct answer: {direct}",
        "Evidence used: " + ", ".join(evidence_used[:6]),
        "Metric facts: " + ("; ".join(facts) if facts else "No numeric metric facts were available."),
        "Visual observations: Evidence views were generated as render artifacts; visual interpretation should be confirmed against deterministic metrics.",
        f"Confidence level: {confidence}",
        "Risks or uncertainty: " + ("; ".join(risks) if risks else "No major limitations flagged."),
        f"Recommended next action: {action}",
    ]
    return CopilotAnswer(
        answer="\n".join(sections),
        evidence_used=evidence_used,
        confidence=confidence,
        recommended_action=action,
        risks_or_uncertainty=risks,
        provider="mock",
        raw_response={"metric_facts": facts, "limitations": package.limitations, "metric_risks": risks},
    )

def _call_http_provider(cfg: Any, package: EvidencePackage) -> CopilotAnswer:
    endpoint = str(cfg_get(cfg, "copilot.vlm.endpoint", "")).strip()
    model = str(cfg_get(cfg, "copilot.vlm.model", "qwen3-vl-8b-instruct"))
    if not endpoint:
        return _mock_answer(package)
    prompt = {
        "model": model,
        "system": SYSTEM_POLICY,
        "question": package.question,
        "evidence": package.to_dict(),
    }
    body = json.dumps(prompt).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=float(cfg_get(cfg, "copilot.vlm.timeout_sec", 120))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        fallback = _mock_answer(package)
        return CopilotAnswer(
            answer=fallback.answer + f"\nProvider warning: HTTP VLM unavailable: {exc}",
            evidence_used=fallback.evidence_used,
            confidence="low",
            recommended_action="Start the configured local VLM server or switch copilot.vlm.provider to mock.",
            risks_or_uncertainty=fallback.risks_or_uncertainty + [f"vlm_http_unavailable:{exc}"],
            provider="http_fallback_mock",
            raw_response={"error": str(exc)},
        )
    text = str(payload.get("answer") or payload.get("response") or payload.get("text") or payload)
    provided_risks_raw = payload.get("risks_or_uncertainty", [])
    provided_risks = provided_risks_raw if isinstance(provided_risks_raw, list) else [provided_risks_raw]
    risks, confidence, action = _merge_metric_risks_for_model_answer(package, provided_risks)

    model_confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence == "high" and model_confidence in {"low", "medium", "high"}:
        confidence = model_confidence

    return CopilotAnswer(
        answer=text,
        evidence_used=list(package.image_paths.values()) + [package.evidence_path],
        confidence=confidence,
        recommended_action=str(payload.get("recommended_action") or action),
        risks_or_uncertainty=risks,
        provider="http",
        raw_response=payload,
    )



def _local_result_to_answer(package: EvidencePackage, provider: str, text: str, raw_response: dict[str, Any]) -> CopilotAnswer:
    evidence_used = list(package.image_paths.values()) + [package.evidence_path]
    raw_risks = raw_response.get("risks_or_uncertainty", []) if isinstance(raw_response, dict) else []
    provided_risks = raw_risks if isinstance(raw_risks, list) else [raw_risks]
    risks, confidence, action = _merge_metric_risks_for_model_answer(package, provided_risks)

    raw_confidence = str(raw_response.get("confidence", "")).strip().lower() if isinstance(raw_response, dict) else ""
    if confidence == "high" and raw_confidence in {"low", "medium", "high"}:
        confidence = raw_confidence

    return CopilotAnswer(
        answer=text,
        evidence_used=evidence_used,
        confidence=confidence,
        recommended_action=action,
        risks_or_uncertainty=risks,
        provider=provider,
        raw_response=raw_response,
    )



def _fallback_or_raise(cfg: Any, package: EvidencePackage, provider: str, exc: Exception) -> CopilotAnswer:
    fallback_allowed = bool(cfg_get(cfg, "copilot.vlm.fallback_to_mock_when_unavailable", True))
    require_real = bool(cfg_get(cfg, "copilot.vlm.require_real_vlm", False))
    if require_real or not fallback_allowed:
        raise LocalVLMError(f"Local VLM provider {provider!r} failed and fallback is disabled: {exc}") from exc
    fallback = _mock_answer(package)
    return CopilotAnswer(
        answer=fallback.answer + f"\nProvider warning: local VLM unavailable: {exc}",
        evidence_used=fallback.evidence_used,
        confidence="low",
        recommended_action=(
            "Start the configured offline local VLM server, verify the endpoint, "
            "or set copilot.vlm.provider=mock only for smoke tests."
        ),
        risks_or_uncertainty=fallback.risks_or_uncertainty + [f"local_vlm_unavailable:{provider}:{exc}"],
        provider=f"{provider}_fallback_mock",
        raw_response={"error": str(exc)},
    )


def answer_with_vlm(cfg: Any, package: EvidencePackage) -> CopilotAnswer:
    provider = str(cfg_get(cfg, "copilot.vlm.provider", "mock")).strip().lower()
    if provider in {"mock", "none", "disabled"}:
        return _mock_answer(package)
    if provider in {"ollama", "ollama_local"}:
        try:
            result = call_ollama_local(cfg, package)
        except Exception as exc:
            return _fallback_or_raise(cfg, package, provider, exc)
        return _local_result_to_answer(package, result.provider, result.text, result.raw_response)
    if provider in {"vllm", "vllm_local", "openai_compatible", "openai_compatible_local", "lmstudio", "transformers_server"}:
        try:
            result = call_openai_compatible_local(cfg, package)
        except Exception as exc:
            return _fallback_or_raise(cfg, package, provider, exc)
        return _local_result_to_answer(package, result.provider, result.text, result.raw_response)
    if provider in {"http", "legacy_http"}:
        return _call_http_provider(cfg, package)
    return _mock_answer(package)
