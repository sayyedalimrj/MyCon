from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class Stage9ConfigError(RuntimeError):
    pass


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


def cfg_float(cfg: Any, dotted_key: str, default: float) -> float:
    value = cfg_get(cfg, dotted_key, default)
    return float(value)


def cfg_bool(cfg: Any, dotted_key: str, default: bool = False) -> bool:
    value = cfg_get(cfg, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def project_root(cfg: Any) -> Path:
    return Path(str(cfg_get(cfg, "project.root", "/workspace"))).expanduser().resolve()


def project_name(cfg: Any) -> str:
    return str(cfg_get(cfg, "project.name", "site01"))


def run_id(cfg: Any) -> str:
    return str(cfg_get(cfg, "project.run_id", "2026-04-30_site01_baseline"))


def resolve_path(cfg: Any, value: str | Path | None, *, required: bool = False) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise Stage9ConfigError("Required path is missing.")
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root(cfg) / path
