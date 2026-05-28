"""Build structured evidence packages for the Construction Copilot."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config_access import cfg_get, resolve_path
from .context_resolver import CopilotContext, resolve_context
from .metric_tools import collect_metrics
from .query_router import RoutedQuery, route_query
from .view_renderer import render_views


@dataclass(frozen=True)
class EvidencePackage:
    question: str
    selected_element_id: str | None
    selected_activity_id: str | None
    route: dict[str, Any]
    selected_context: dict[str, Any]
    image_paths: dict[str, str]
    metrics: dict[str, Any]
    schedule_context: dict[str, Any]
    limitations: list[str]
    confidence_flags: list[str]
    evidence_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _schedule_context(metrics: dict[str, Any]) -> dict[str, Any]:
    activity = metrics.get("activity_progress", {})
    return {
        "status": activity.get("status", "missing"),
        "data": activity.get("data", {}),
        "source_path": activity.get("source_path"),
    }


def _limitations(context: CopilotContext, metrics: dict[str, Any], rendered_warnings: list[str]) -> list[str]:
    items: list[str] = list(context.warnings) + rendered_warnings
    for name, result in metrics.items():
        status = result.get("status")
        if status in {"missing", "not_found", "path_not_configured", "no_element_selected", "no_activity_selected"}:
            items.append(f"metric_{name}:{status}")
        for warning in result.get("warnings", []) or []:
            items.append(f"metric_{name}:{warning}")
    return sorted(set(items))


def _confidence_flags(route: RoutedQuery, limitations: list[str]) -> list[str]:
    flags: list[str] = []
    if limitations:
        flags.append("evidence_incomplete")
    if route.needs_metrics and any(item.startswith("metric_") for item in limitations):
        flags.append("metric_artifacts_missing_or_incomplete")
    if route.needs_visuals:
        flags.append("visual_evidence_generated")
    return flags or ["evidence_package_complete_enough_for_mock_answer"]


def build_evidence_package(
    cfg: Any,
    question: str,
    *,
    element_global_id: str | None = None,
    activity_id: str | None = None,
    selected_bbox: Any = None,
    current_view: str | None = None,
    camera_pose: Any = None,
    pointcloud_path: str | Path | None = None,
    ifc_path: str | Path | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> EvidencePackage:
    route = route_query(question)
    context = resolve_context(
        cfg,
        element_global_id=element_global_id,
        activity_id=activity_id,
        selected_bbox=selected_bbox,
        current_view=current_view,
        camera_pose=camera_pose,
        pointcloud_path=pointcloud_path,
        ifc_path=ifc_path,
        artifact_paths=artifact_paths,
    )
    rendered = render_views(cfg, context, question=question, requested_views=route.requested_views)
    metrics = collect_metrics(cfg, element_global_id=element_global_id, activity_id=activity_id)
    limitations = _limitations(context, metrics, rendered.warnings)
    confidence_flags = _confidence_flags(route, limitations)

    evidence_dir = resolve_path(cfg, cfg_get(cfg, "copilot.paths.evidence_dir", "runs/copilot/evidence"), required=True)
    assert evidence_dir is not None
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "latest_evidence_package.json"

    package = EvidencePackage(
        question=question,
        selected_element_id=element_global_id,
        selected_activity_id=activity_id,
        route=route.to_dict(),
        selected_context=context.to_dict(),
        image_paths=rendered.image_paths,
        metrics=metrics,
        schedule_context=_schedule_context(metrics),
        limitations=limitations,
        confidence_flags=confidence_flags,
        evidence_path=evidence_path.as_posix(),
    )
    evidence_path.write_text(json.dumps(_json_ready(package.to_dict()), indent=2, sort_keys=True), encoding="utf-8")
    return package
