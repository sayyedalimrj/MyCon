"""Small compatibility helpers for reading the project YAML config.

The project already has ``pipeline.common.config.PipelineConfig`` with ``get`` and
``require`` helpers. These functions keep Stage 3 robust if tests use a plain
``dict`` or a lightweight stub config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class Stage3ConfigError(RuntimeError):
    """Raised when Stage 3 configuration or file contracts are invalid."""


def _walk_mapping(data: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from PipelineConfig-like objects or dictionaries."""
    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass
    data = getattr(cfg, "data", cfg)
    if isinstance(data, Mapping):
        try:
            return _walk_mapping(data, dotted_key)
        except KeyError:
            return default
    return default


def cfg_require(cfg: Any, dotted_key: str) -> Any:
    """Read a required dotted key and raise a clear Stage 3 error if missing."""
    if hasattr(cfg, "require"):
        try:
            return cfg.require(dotted_key)
        except Exception as exc:  # noqa: BLE001 - preserve source while normalizing message
            raise Stage3ConfigError(f"Missing required config key for Stage 3: {dotted_key}") from exc
    value = cfg_get(cfg, dotted_key, None)
    if value is None:
        raise Stage3ConfigError(f"Missing required config key for Stage 3: {dotted_key}")
    return value


def cfg_bool(cfg: Any, dotted_key: str, default: bool = False) -> bool:
    value = cfg_get(cfg, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def cfg_int(cfg: Any, dotted_key: str, default: int) -> int:
    value = cfg_get(cfg, dotted_key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Stage3ConfigError(f"Config key {dotted_key} must be an integer, got {value!r}") from exc


def cfg_float(cfg: Any, dotted_key: str, default: float) -> float:
    value = cfg_get(cfg, dotted_key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise Stage3ConfigError(f"Config key {dotted_key} must be a float, got {value!r}") from exc


def project_root(cfg: Any) -> Path:
    root = Path(str(cfg_require(cfg, "project.root")))
    if not root.is_absolute():
        raise Stage3ConfigError(f"project.root must be absolute inside Docker/WSL, got: {root}")
    return root


def resolve_project_path(cfg: Any, dotted_key: str, default: str | None = None) -> Path:
    raw = cfg_get(cfg, dotted_key, default)
    if raw is None:
        raise Stage3ConfigError(f"Missing path config key for Stage 3: {dotted_key}")
    path = Path(str(raw))
    return path if path.is_absolute() else project_root(cfg) / path


def run_id(cfg: Any) -> str:
    value = str(cfg_require(cfg, "project.run_id"))
    if not value.strip():
        raise Stage3ConfigError("project.run_id must not be empty")
    return value


def project_name(cfg: Any) -> str:
    value = str(cfg_require(cfg, "project.name"))
    if not value.strip():
        raise Stage3ConfigError("project.name must not be empty")
    return value


def bool_to_colmap(value: bool) -> str:
    return "1" if value else "0"
