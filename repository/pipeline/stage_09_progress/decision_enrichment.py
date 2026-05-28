from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from pipeline.common.progress_decision_policy import decide_element_progress
from .visibility_policy import interpret_element_visibility


def _cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_config(cfg_or_path: Any) -> dict[str, Any]:
    if isinstance(cfg_or_path, dict):
        return cfg_or_path

    path = Path(str(cfg_or_path))
    if not path.exists():
        return {}

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _project_root(cfg: dict[str, Any]) -> Path:
    root = _cfg_get(cfg, "project.root", ".")
    return Path(str(root)).resolve()


def _resolve(root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _first_path(cfg: dict[str, Any], keys: list[str], fallback: str) -> Path:
    root = _project_root(cfg)
    for key in keys:
        value = _cfg_get(cfg, key)
        if value:
            return _resolve(root, value)
    return _resolve(root, fallback)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _registration_confidence(registration_quality_json: Path) -> str:
    data = _read_json(registration_quality_json)
    candidates = [
        data.get("confidence_label"),
        data.get("registration_confidence"),
        data.get("confidence"),
        data.get("status"),
        data.get("quality_gate", {}).get("confidence") if isinstance(data.get("quality_gate"), dict) else None,
    ]
    for item in candidates:
        if item not in (None, ""):
            return str(item)
    return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def enrich_element_metrics_rows(
    rows: list[dict[str, Any]],
    *,
    registration_confidence: str,
    min_coverage: float = 0.65,
    min_in_tolerance: float = 0.65,
    min_element_confidence: float = 0.65,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    acceptable_count = 0
    state_counts: dict[str, int] = {}

    for row in rows:
        decision = decide_element_progress(
            row,
            registration_confidence=registration_confidence,
            min_coverage=min_coverage,
            min_in_tolerance=min_in_tolerance,
            min_element_confidence=min_element_confidence,
        )

        out = dict(row)
        out["completion_state"] = decision.completion_state
        out["evidence_status"] = decision.evidence_status
        out["acceptable"] = str(decision.acceptable).lower()
        out["decision_confidence"] = decision.confidence
        out["decision_risks"] = ";".join(decision.risks)
        
        out["decision_recommended_action"] = decision.recommended_action
        
        visibility_fields = interpret_element_visibility(out)
        out.update(visibility_fields)

        if visibility_fields.get("visibility_decision_risks"):
            existing_risks = str(out.get("decision_risks", "") or "")
            extra_risks = visibility_fields["visibility_decision_risks"]
            out["decision_risks"] = ";".join(x for x in [existing_risks, extra_risks] if x)

        if decision.acceptable:
            acceptable_count += 1
        state_counts[decision.completion_state] = state_counts.get(decision.completion_state, 0) + 1

        enriched.append(out)

    return enriched, {
        "element_count": len(enriched),
        "acceptable_element_count": acceptable_count,
        "completion_state_counts": state_counts,
        "registration_confidence": registration_confidence,
    }


def enrich_activity_progress_rows(
    rows: list[dict[str, Any]],
    *,
    registration_confidence: str,
    element_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_activity: dict[str, list[dict[str, Any]]] = {}
    for row in element_rows:
        activity_id = str(row.get("activity_id", "")).strip()
        if activity_id:
            by_activity.setdefault(activity_id, []).append(row)

    enriched: list[dict[str, Any]] = []
    acceptable_count = 0
    state_counts: dict[str, int] = {}

    for row in rows:
        out = dict(row)
        activity_id = str(row.get("activity_id", "")).strip()
        elements = by_activity.get(activity_id, [])

        if registration_confidence.lower() != "high":
            completion_state = "uncertain_low_registration"
            evidence_status = "blocked_by_registration"
            acceptable = False
            confidence = "low"
            risks = ["registration_confidence_low"]
            action = "Improve Stage 8 registration before accepting activity progress."
        elif not elements:
            completion_state = "not_evidenced"
            evidence_status = "missing_element_metrics"
            acceptable = False
            confidence = "low"
            risks = ["activity_element_metrics:not_found"]
            action = "Provide mapped element metrics before accepting activity progress."
        else:
            total = len(elements)
            accepted = sum(str(e.get("acceptable", "")).lower() == "true" for e in elements)
            acceptable_ratio = accepted / max(total, 1)

            if acceptable_ratio >= 0.65:
                completion_state = "completed"
                evidence_status = "metric_evidence_sufficient"
                acceptable = True
                confidence = "high"
                risks = []
                action = "Activity can be considered acceptable from mapped element metrics, subject to project QA review."
            elif accepted > 0:
                completion_state = "partial"
                evidence_status = "partial_metric_evidence"
                acceptable = False
                confidence = "medium"
                risks = [f"activity_acceptable_ratio_below_threshold:{acceptable_ratio:.6f}<0.650000"]
                action = "Treat activity as partial until more mapped elements pass acceptance thresholds."
            else:
                completion_state = "not_evidenced"
                evidence_status = "insufficient_metric_evidence"
                acceptable = False
                confidence = "low"
                risks = ["activity_no_acceptable_elements"]
                action = "Do not accept activity completion from current element metrics."

        out["completion_state"] = completion_state
        out["evidence_status"] = evidence_status
        out["acceptable"] = str(acceptable).lower()
        out["decision_confidence"] = confidence
        out["decision_risks"] = ";".join(risks)
        out["decision_recommended_action"] = action

        if acceptable:
            acceptable_count += 1
        state_counts[completion_state] = state_counts.get(completion_state, 0) + 1
        enriched.append(out)

    return enriched, {
        "activity_count": len(enriched),
        "acceptable_activity_count": acceptable_count,
        "completion_state_counts": state_counts,
        "registration_confidence": registration_confidence,
    }


def enrich_progress_decisions_from_files(
    *,
    element_metrics_csv: Path,
    activity_progress_csv: Path,
    registration_quality_json: Path,
    output_summary_json: Path | None = None,
) -> dict[str, Any]:
    registration_confidence = _registration_confidence(registration_quality_json)

    element_rows = _read_csv(element_metrics_csv)
    activity_rows = _read_csv(activity_progress_csv)

    enriched_elements, element_summary = enrich_element_metrics_rows(
        element_rows,
        registration_confidence=registration_confidence,
    )
    enriched_activities, activity_summary = enrich_activity_progress_rows(
        activity_rows,
        registration_confidence=registration_confidence,
        element_rows=enriched_elements,
    )

    if enriched_elements:
        _write_csv(element_metrics_csv, enriched_elements)

    if enriched_activities:
        _write_csv(activity_progress_csv, enriched_activities)

    summary = {
        "stage": "stage_09_progress_decision_enrichment",
        "status": "ok" if enriched_elements or enriched_activities else "skipped_no_progress_rows",
        "registration_confidence": registration_confidence,
        "element_summary": element_summary,
        "activity_summary": activity_summary,
        "element_metrics_csv": str(element_metrics_csv),
        "activity_progress_csv": str(activity_progress_csv),
        "registration_quality_json": str(registration_quality_json),
    }

    if output_summary_json is not None:
        output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def enrich_progress_decisions_from_config(cfg_or_path: Any) -> dict[str, Any]:
    cfg = _load_config(cfg_or_path)

    element_metrics_csv = _first_path(
        cfg,
        [
            "progress.element_metrics_csv",
            "progress.paths.element_metrics_csv",
            "paths.element_metrics_csv",
            "copilot.paths.element_metrics_csv",
        ],
        "data/bim/metrics/site01/element_metrics.csv",
    )

    activity_progress_csv = _first_path(
        cfg,
        [
            "progress.activity_progress_csv",
            "progress.paths.activity_progress_csv",
            "paths.activity_progress_csv",
            "copilot.paths.activity_progress_csv",
        ],
        "data/bim/metrics/site01/activity_progress.csv",
    )

    registration_quality_json = _first_path(
        cfg,
        [
            "progress.registration_quality_json",
            "progress.paths.registration_quality_json",
            "paths.registration_quality_json",
            "copilot.paths.registration_quality_json",
        ],
        "data/bim/metrics/site01/registration_quality.json",
    )

    output_summary_json = _first_path(
        cfg,
        [
            "progress.progress_decision_summary_json",
            "progress.paths.progress_decision_summary_json",
            "paths.progress_decision_summary_json",
        ],
        "runs/2026-04-30_site01_baseline/reports/progress_decision_summary.json",
    )

    return enrich_progress_decisions_from_files(
        element_metrics_csv=element_metrics_csv,
        activity_progress_csv=activity_progress_csv,
        registration_quality_json=registration_quality_json,
        output_summary_json=output_summary_json,
    )