from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class Stage45ConfigError(ValueError):
    """Raised for Stage 4.5 configuration errors."""


def _walk_mapping(data: Mapping[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    data = getattr(cfg, "data", cfg)

    if isinstance(data, Mapping):
        try:
            return _walk_mapping(data, dotted_key)
        except KeyError:
            return default

    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass

    return default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "always"}


def project_root(cfg: Any) -> Path:
    return Path(str(cfg_get(cfg, "project.root", "."))).expanduser().resolve()


def project_name(cfg: Any) -> str:
    return str(cfg_get(cfg, "project.name", "site01"))


def run_id(cfg: Any) -> str:
    return str(cfg_get(cfg, "project.run_id", "2026-04-30_site01_baseline"))


def resolve_path(cfg: Any, dotted_key: str, default: str) -> Path:
    raw = cfg_get(cfg, dotted_key, default)
    path = Path(str(raw))
    return path if path.is_absolute() else project_root(cfg) / path


def stage45_paths(cfg: Any) -> dict[str, Path]:
    return {
        "output_dir": resolve_path(cfg, "cams_gs.output_dir", "data/cams_gs/site01"),
        "nerfstudio_dataset_dir": resolve_path(cfg, "cams_gs.nerfstudio_dataset_dir", "data/cams_gs/site01/nerfstudio_dataset"),
        "manifest_json": resolve_path(cfg, "cams_gs.manifest_json", "data/cams_gs/site01/train_manifest.json"),
        "training_status_json": resolve_path(cfg, "cams_gs.training_status_json", "data/cams_gs/site01/training_status.json"),
        "report_json": resolve_path(cfg, "cams_gs.report_json", "runs/2026-04-30_site01_baseline/reports/cams_gs_prepare_summary.json"),
        "source_images_dir": resolve_path(cfg, "cams_gs.source_images_dir", "data/sfm/site01/images"),
        "source_sparse_model_dir": resolve_path(cfg, "cams_gs.source_sparse_model_dir", "data/sparse_refined/site01/0"),
        "source_colmap_workspace": resolve_path(cfg, "cams_gs.source_colmap_workspace", "data/sfm/site01"),
    }
