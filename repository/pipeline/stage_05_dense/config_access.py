"""Configuration helpers for Stage 5 dense stereo."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class Stage5ConfigError(ValueError):
    """Raised when Stage 5 configuration is invalid."""


class ConfigOverlay:
    """Read-through config wrapper with dotted-key runtime overrides.

    This keeps YAML file contracts stable while allowing Stage 5 to adapt dense
    parameters to the GPU visible inside the runtime container.
    """

    def __init__(self, base: Any, overrides: dict[str, Any]) -> None:
        self.base = base
        self.overrides = overrides

    def get(self, dotted_key: str, default: Any = None) -> Any:
        if dotted_key in self.overrides:
            return self.overrides[dotted_key]
        return cfg_get(self.base, dotted_key, default)

    @property
    def data(self) -> Any:
        return getattr(self.base, "data", self.base)


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Read a dotted config key from PipelineConfig-like objects or dictionaries.

    PipelineConfig exposes ``get("section.key")``; plain dictionaries should be
    traversed as nested mappings instead of using ``dict.get`` on the full dotted
    string.
    """
    if hasattr(cfg, "get") and not isinstance(cfg, dict):
        try:
            return cfg.get(dotted_key, default)
        except TypeError:
            pass
    current: Any = getattr(cfg, "data", cfg)
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cfg_bool(cfg: Any, dotted_key: str, default: bool = False) -> bool:
    value = cfg_get(cfg, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def cfg_int(cfg: Any, dotted_key: str, default: int) -> int:
    try:
        return int(cfg_get(cfg, dotted_key, default))
    except (TypeError, ValueError) as exc:
        raise Stage5ConfigError(f"Config key {dotted_key} must be an integer.") from exc


def cfg_float(cfg: Any, dotted_key: str, default: float) -> float:
    try:
        return float(cfg_get(cfg, dotted_key, default))
    except (TypeError, ValueError) as exc:
        raise Stage5ConfigError(f"Config key {dotted_key} must be a float.") from exc


def project_name(cfg: Any) -> str:
    name = str(cfg_get(cfg, "project.name", "site01")).strip()
    if not name:
        raise Stage5ConfigError("project.name must not be empty")
    return name


def run_id(cfg: Any) -> str:
    rid = str(cfg_get(cfg, "project.run_id", "run")).strip()
    if not rid:
        raise Stage5ConfigError("project.run_id must not be empty")
    return rid


def project_root(cfg: Any) -> Path:
    root = Path(str(cfg_get(cfg, "project.root", "/workspace")))
    return root


def resolve_project_path(cfg: Any, dotted_key: str, default: str | Path) -> Path:
    value = cfg_get(cfg, dotted_key, str(default))
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root(cfg) / path
