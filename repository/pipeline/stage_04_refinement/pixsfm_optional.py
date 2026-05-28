"""Optional PixSfM hook for Stage 4.

PixSfM is intentionally disabled by default. The baseline thesis pipeline must
not be blocked by this optional dependency. This module records whether PixSfM
is available and returns a structured skip result unless explicitly enabled.
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config_access import Stage4ConfigError, cfg_bool


@dataclass(slots=True)
class PixsfmResult:
    enabled: bool
    status: str
    message: str
    input_model_dir: str | None = None
    output_model_dir: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        return asdict(self)


def run_pixsfm_optional(
    cfg: Any,
    input_model_dir: Path,
    output_model_dir: Path,
    logger: logging.Logger,
) -> PixsfmResult:
    enabled = cfg_bool(cfg, "refinement.pixsfm_enabled", False)
    if not enabled:
        return PixsfmResult(False, "disabled", "PixSfM disabled by config; baseline COLMAP BA remains mandatory.")

    if importlib.util.find_spec("pixsfm") is None:
        message = "PixSfM requested but Python package 'pixsfm' is not installed in this environment."
        if cfg_bool(cfg, "refinement.pixsfm_allow_missing", True):
            logger.warning("%s Continuing with COLMAP bundle adjustment only.", message)
            return PixsfmResult(True, "skipped_missing_dependency", message)
        raise Stage4ConfigError(message)

    # The operational baseline does not run PixSfM yet because its CLI/API and
    # feature inputs need project-specific integration. Keeping this as a clear
    # hook prevents accidental hidden dependency on PixSfM while preserving the
    # thesis research path.
    message = "PixSfM is installed, but automatic PixSfM refinement is not wired into the baseline Stage 4 run."
    logger.warning("%s Continuing with COLMAP bundle adjustment only.", message)
    return PixsfmResult(True, "skipped_not_wired", message, str(input_model_dir), str(output_model_dir))
