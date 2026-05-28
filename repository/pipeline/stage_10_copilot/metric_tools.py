"""Deterministic metric readers for Stage 10 evidence.

The VLM is not allowed to invent numeric measurements. All numeric facts in the
answer should come from this module or future deterministic geometry tools.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config_access import cfg_get, resolve_path


@dataclass(frozen=True)
class MetricResult:
    name: str
    status: str
    data: dict[str, Any]
    source_path: str | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {}, ["path_not_configured"]
    if not path.exists():
        return {}, [f"missing:{path.as_posix()}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return {}, [f"invalid_json:{path.as_posix()}:{exc}"]


def _read_csv_rows(path: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    if path is None:
        return [], ["path_not_configured"]
    if not path.exists():
        return [], [f"missing:{path.as_posix()}"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), []


def _path(cfg: Any, key: str) -> Path | None:
    return resolve_path(cfg, cfg_get(cfg, key, None))


def get_element_metrics(cfg: Any, global_id: str | None) -> MetricResult:
    path = _path(cfg, "copilot.paths.element_metrics_csv")
    rows, warnings = _read_csv_rows(path)
    if not global_id:
        return MetricResult("element_metrics", "no_element_selected", {}, path.as_posix() if path else None, warnings)
    candidates = [r for r in rows if r.get("global_id") == global_id or r.get("GlobalId") == global_id]
    status = "ok" if candidates else "not_found"
    return MetricResult("element_metrics", status, candidates[0] if candidates else {}, path.as_posix() if path else None, warnings)


def get_activity_progress(cfg: Any, activity_id: str | None) -> MetricResult:
    path = _path(cfg, "copilot.paths.activity_progress_csv")
    rows, warnings = _read_csv_rows(path)
    if not activity_id:
        return MetricResult("activity_progress", "no_activity_selected", {}, path.as_posix() if path else None, warnings)
    candidates = [r for r in rows if r.get("activity_id") == activity_id or r.get("ActivityId") == activity_id]
    return MetricResult("activity_progress", "ok" if candidates else "not_found", candidates[0] if candidates else {}, path.as_posix() if path else None, warnings)


def get_nearest_deviation_summary(cfg: Any, region_or_element: dict[str, Any] | None = None) -> MetricResult:
    path = _path(cfg, "copilot.paths.deviation_summary_json")
    data, warnings = _read_json(path)
    if region_or_element:
        data = {"query_context": region_or_element, "summary": data}
    return MetricResult("deviation_summary", "ok" if data else "missing", data, path.as_posix() if path else None, warnings)


def get_low_confidence_elements(cfg: Any, limit: int | None = None) -> MetricResult:
    path = _path(cfg, "copilot.paths.element_metrics_csv")
    rows, warnings = _read_csv_rows(path)
    threshold = float(cfg_get(cfg, "copilot.low_confidence_threshold", 0.65))
    low: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("confidence") or row.get("confidence_score") or row.get("coverage")
        try:
            score = float(value) if value is not None else 1.0
        except ValueError:
            continue
        if score < threshold:
            low.append(row)
    low = low[: int(limit or cfg_get(cfg, "copilot.max_metric_rows", 20))]
    return MetricResult("low_confidence_elements", "ok" if low else "none_or_missing", {"items": low, "threshold": threshold}, path.as_posix() if path else None, warnings)


def get_undercovered_regions(cfg: Any) -> MetricResult:
    path = _path(cfg, "copilot.paths.coverage_summary_json")
    data, warnings = _read_json(path)
    return MetricResult("undercovered_regions", "ok" if data else "missing", data, path.as_posix() if path else None, warnings)


def get_registration_quality(cfg: Any) -> MetricResult:
    path = _path(cfg, "copilot.paths.registration_quality_json")
    data, warnings = _read_json(path)
    return MetricResult("registration_quality", "ok" if data else "missing", data, path.as_posix() if path else None, warnings)


def collect_metrics(cfg: Any, *, element_global_id: str | None = None, activity_id: str | None = None) -> dict[str, Any]:
    results = [
        get_element_metrics(cfg, element_global_id),
        get_activity_progress(cfg, activity_id),
        get_nearest_deviation_summary(cfg, {"element_global_id": element_global_id, "activity_id": activity_id}),
        get_low_confidence_elements(cfg),
        get_undercovered_regions(cfg),
        get_registration_quality(cfg),
    ]
    return {result.name: result.to_dict() for result in results}
