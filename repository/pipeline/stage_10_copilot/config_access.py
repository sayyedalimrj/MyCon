"""Config helpers for Stage 10.

These helpers work with both the project's PipelineConfig object and plain dicts
used by tests and smoke scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class Stage10ConfigError(RuntimeError):
    """Raised when a Stage 10 config value is invalid."""


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Read a dotted config key from PipelineConfig or dict-like objects."""
    data = getattr(cfg, "data", cfg)

    if isinstance(data, dict):
        current: Any = data
        for part in dotted_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    if hasattr(cfg, "get"):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass
        except Exception:
            pass

    return default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on", "enabled"}


def project_root(cfg: Any) -> Path:
    root = cfg_get(cfg, "project.root", "/workspace")
    return Path(str(root)).expanduser().resolve()


def resolve_path(cfg: Any, value: str | Path | None, *, required: bool = False) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise Stage10ConfigError("Required path value is missing.")
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (project_root(cfg) / path).resolve()


def first_existing_path(cfg: Any, values: Iterable[str | Path | None]) -> Path | None:
    for value in values:
        path = resolve_path(cfg, value)
        if path and path.exists() and path.stat().st_size > 0:
            return path
    return None
