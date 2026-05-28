"""Resolve selected element, activity, view and region context for Copilot."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config_access import cfg_get, resolve_path


@dataclass(frozen=True)
class CopilotContext:
    element_global_id: str | None
    activity_id: str | None
    selected_bbox: list[float] | None
    current_view: str
    camera_pose: list[float] | None
    pointcloud_path: str | None
    ifc_path: str | None
    artifact_paths: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float_list(value: Any, *, expected: int | None = None) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(value)
    result = [float(p) for p in parts]
    if expected is not None and len(result) != expected:
        raise ValueError(f"Expected {expected} numeric values, got {len(result)}.")
    return result


def resolve_context(
    cfg: Any,
    *,
    element_global_id: str | None = None,
    activity_id: str | None = None,
    selected_bbox: Any = None,
    current_view: str | None = None,
    camera_pose: Any = None,
    pointcloud_path: str | Path | None = None,
    ifc_path: str | Path | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> CopilotContext:
    warnings: list[str] = []
    view = current_view or str(cfg_get(cfg, "copilot.default_view", "front"))

    if pointcloud_path is None:
        pointcloud_path = cfg_get(cfg, "copilot.paths.default_pointcloud", cfg_get(cfg, "paths.clean_cloud", None))
    if ifc_path is None:
        ifc_path = cfg_get(cfg, "inputs.ifc", None)

    resolved_pointcloud = resolve_path(cfg, pointcloud_path)
    resolved_ifc = resolve_path(cfg, ifc_path)
    if resolved_pointcloud and not resolved_pointcloud.exists():
        warnings.append(f"pointcloud_missing:{resolved_pointcloud.as_posix()}")
    if resolved_ifc and not resolved_ifc.exists():
        warnings.append(f"ifc_missing:{resolved_ifc.as_posix()}")

    artifacts: dict[str, str] = {}
    for key, value in (artifact_paths or {}).items():
        path = resolve_path(cfg, value)
        artifacts[key] = path.as_posix() if path else str(value)

    return CopilotContext(
        element_global_id=element_global_id,
        activity_id=activity_id,
        selected_bbox=_as_float_list(selected_bbox, expected=6) if selected_bbox is not None else None,
        current_view=view,
        camera_pose=_as_float_list(camera_pose) if camera_pose is not None else None,
        pointcloud_path=resolved_pointcloud.as_posix() if resolved_pointcloud else None,
        ifc_path=resolved_ifc.as_posix() if resolved_ifc else None,
        artifact_paths=artifacts,
        warnings=warnings,
    )
