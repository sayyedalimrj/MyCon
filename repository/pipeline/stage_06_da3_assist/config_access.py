from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class Stage6ConfigError(ValueError):
    """Raised for invalid Stage 6 configuration."""


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass
    data = getattr(cfg, "data", cfg)
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_require(cfg: Any, dotted_key: str) -> Any:
    value = cfg_get(cfg, dotted_key, None)
    if value is None:
        raise Stage6ConfigError(f"Missing required config key: {dotted_key}")
    return value


def project_root(cfg: Any) -> Path:
    root = Path(str(cfg_get(cfg, "project.root", ".")))
    return root.expanduser().resolve()


def resolve_path(cfg: Any, dotted_key: str, default: str | None = None) -> Path:
    value = cfg_get(cfg, dotted_key, default)
    if value is None:
        raise Stage6ConfigError(f"Missing required path config key: {dotted_key}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root(cfg) / path


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def list_value(value: Any, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def da3_enabled(cfg: Any) -> str:
    value = str(cfg_get(cfg, "da3.enabled", "auto")).strip().lower()
    if value not in {"auto", "true", "false", "on", "off", "always", "never"}:
        raise Stage6ConfigError("da3.enabled must be one of: auto, true, false, on, off, always, never.")
    return value


def stage6_paths(cfg: Any) -> dict[str, Path]:
    return {
        "da3_dir": resolve_path(cfg, "paths.da3_dir", "data/da3/site01"),
        "decision_json": resolve_path(cfg, "paths.da3_decision_json", "data/da3/site01/decision.json"),
        "depth_manifest_csv": resolve_path(cfg, "paths.da3_depth_manifest_csv", "data/da3/site01/depth_manifest.csv"),
        "alignment_manifest_csv": resolve_path(cfg, "paths.da3_alignment_manifest_csv", "data/da3/site01/alignment_manifest.csv"),
        "fusion_plan_json": resolve_path(cfg, "paths.da3_fusion_plan_json", "data/da3/site01/fusion_plan.json"),
        "assisted_ply": resolve_path(cfg, "paths.da3_assisted_ply", "data/da3/site01/da3_assisted_points.ply"),
        "report_json": resolve_path(cfg, "paths.da3_report_json", "runs/2026-04-30_site01_baseline/reports/da3_summary.json"),
        "dense_summary_json": resolve_path(cfg, "paths.dense_summary_json", "runs/2026-04-30_site01_baseline/reports/dense_summary.json"),
        "sparse_refined_dir": resolve_path(cfg, "paths.sparse_refined_dir", "data/sparse_refined/site01/0"),
        "sparse_text_dir": resolve_path(cfg, "da3.sparse_text_dir", "data/da3/site01/sparse_txt"),
        "image_dir": resolve_path(cfg, "paths.sfm_images_dir", "data/sfm/site01/images"),
        "keyframes_dir": resolve_path(cfg, "paths.keyframes_dir", "data/frames/key/site01"),
        "depth_input_dir": resolve_path(cfg, "da3.depth_input_dir", "data/da3/site01/raw_depth"),
        "depth_output_dir": resolve_path(cfg, "da3.depth_output_dir", "data/da3/site01/raw_depth"),
        "aligned_depth_dir": resolve_path(cfg, "da3.aligned_depth_dir", "data/da3/site01/aligned_depth"),
    }
