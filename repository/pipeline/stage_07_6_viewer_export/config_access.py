from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class Stage76ConfigError(ValueError):
    """Raised for Stage 7.6 configuration errors."""


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


def stage76_paths(cfg: Any) -> dict[str, Path]:
    return {
        "output_dir": resolve_path(cfg, "viewer_export.output_dir", "exports/viewer/site01"),
        "artifacts_dir": resolve_path(cfg, "viewer_export.artifacts_dir", "exports/viewer/site01/artifacts"),
        "manifest_json": resolve_path(cfg, "viewer_export.manifest_json", "exports/viewer/site01/viewer_manifest.json"),
        "index_html": resolve_path(cfg, "viewer_export.index_html", "exports/viewer/site01/index.html"),
    }
