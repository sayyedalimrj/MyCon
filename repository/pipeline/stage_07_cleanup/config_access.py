from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class Stage7ConfigError(ValueError):
    """Raised for invalid Stage 7 configuration."""


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    data = getattr(cfg, "data", cfg)

    if isinstance(data, dict):
        cur: Any = data
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass

    return default


def cfg_require(cfg: Any, dotted_key: str) -> Any:
    value = cfg_get(cfg, dotted_key, None)
    if value is None:
        raise Stage7ConfigError(f"Missing required config key: {dotted_key}")
    return value


def project_root(cfg: Any) -> Path:
    root = Path(str(cfg_get(cfg, "project.root", ".")))
    return root.expanduser().resolve()


def resolve_path(cfg: Any, dotted_key: str, default: str | None = None) -> Path:
    value = cfg_get(cfg, dotted_key, default)
    if value is None:
        raise Stage7ConfigError(f"Missing required path config key: {dotted_key}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root(cfg) / path


def resolve_relative_or_abs(cfg: Any, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root(cfg) / path


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "always"}


def list_value(value: Any, default: Iterable[Any] = ()) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def float_list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none"}:
            return None
        parts = [part.strip() for part in stripped.replace(";", ",").split(",")]
        return [float(part) for part in parts if part]
    values = [float(v) for v in value]
    return values


def stage7_paths(cfg: Any) -> dict[str, Path]:
    return {
        "clean_dir": resolve_path(cfg, "paths.clean_dir", "data/clean/site01"),
        "downsampled_cloud": resolve_path(cfg, "paths.clean_downsampled_cloud", "data/clean/site01/downsampled_cloud.ply"),
        "cleaned_cloud": resolve_path(cfg, "paths.clean_cloud", "data/clean/site01/cleaned_cloud.ply"),
        "mesh_ply": resolve_path(cfg, "paths.clean_mesh", "data/clean/site01/mesh.ply"),
        "planes_json": resolve_path(cfg, "paths.clean_planes_json", "data/clean/site01/planes.json"),
        "plane_clouds_dir": resolve_path(cfg, "paths.clean_plane_clouds_dir", "data/clean/site01/plane_clouds"),
        "report_json": resolve_path(cfg, "paths.cleanup_report_json", "runs/2026-04-30_site01_baseline/reports/cleanup_summary.json"),
        "dense_fused_ply": resolve_path(cfg, "paths.fused_ply", "data/dense/site01/fused.ply"),
        "da3_assisted_ply": resolve_path(cfg, "paths.da3_assisted_ply", "data/da3/site01/da3_assisted_points.ply"),
        "yolo_detections_jsonl": resolve_path(cfg, "cleanup.yolo_detections_jsonl", "data/semantics/site01/yolo_detections.jsonl"),
        "yolo_summary_json": resolve_path(cfg, "cleanup.yolo_summary_json", "data/semantics/site01/yolo_summary.json"),
        "vlm_scene_report_json": resolve_path(cfg, "cleanup.vlm_scene_report_json", "data/semantics/site01/vlm_scene_report.json"),
        "vlm_summary_json": resolve_path(cfg, "cleanup.vlm_summary_json", "data/semantics/site01/vlm_summary.json"),
    }


def input_candidates(cfg: Any) -> list[Path]:
    configured = list_value(cfg_get(cfg, "cleanup.input_candidates", None), [])
    if configured:
        return [resolve_relative_or_abs(cfg, p) for p in configured]
    paths = stage7_paths(cfg)
    candidates = [paths["da3_assisted_ply"], paths["dense_fused_ply"]]
    if not bool_value(cfg_get(cfg, "cleanup.prefer_da3_assisted", True)):
        candidates = [paths["dense_fused_ply"], paths["da3_assisted_ply"]]
    return candidates
