"""Input selection for Stage 8."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import cfg_get, cfg_list, resolve_project_path, resolve_raw_project_path


class Stage8InputError(RuntimeError):
    """Raised when a required Stage 8 input is missing."""


@dataclass(frozen=True)
class SelectedInput:
    path: Path
    source: str
    reason: str
    size_bytes: int


def _source_name(path: Path) -> str:
    parts = set(path.parts)
    if "da3" in parts:
        return "da3_assisted_clean_geometry"
    if "clean" in parts:
        return "stage_07_clean_geometry"
    if "dense" in parts:
        return "stage_05_dense_geometry"
    return "custom"


def scan_input_candidates(cfg: Any) -> list[Path]:
    configured = cfg_list(cfg, "bim.scan_input_candidates", [])
    if not configured:
        configured = [
            str(cfg_get(cfg, "paths.clean_ply", "data/clean/site01/cleaned_cloud.ply")),
            str(cfg_get(cfg, "paths.cleaned_cloud", "data/clean/site01/cleaned_cloud.ply")),
            str(cfg_get(cfg, "paths.clean_point_cloud", "data/clean/site01/cleaned_cloud.ply")),
            str(cfg_get(cfg, "paths.fused_da3_ply", "data/da3/site01/da3_assisted_points.ply")),
            str(cfg_get(cfg, "paths.fused_ply", "data/dense/site01/fused.ply")),
        ]
    candidates: list[Path] = []
    seen: set[str] = set()
    for raw in configured:
        path = resolve_raw_project_path(cfg, raw)
        key = path.as_posix()
        if key not in seen:
            candidates.append(path)
            seen.add(key)
    return candidates


def select_scan_input(cfg: Any) -> SelectedInput:
    checked: list[str] = []
    for path in scan_input_candidates(cfg):
        checked.append(path.as_posix())
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return SelectedInput(
                path=path,
                source=_source_name(path),
                reason="first_existing_candidate",
                size_bytes=int(path.stat().st_size),
            )
    raise Stage8InputError("No valid Stage 8 scan point cloud found. Checked: " + ", ".join(checked))


def select_ifc_input(cfg: Any) -> SelectedInput:
    ifc_path = resolve_project_path(cfg, "inputs.ifc", "data/bim/design/model.ifc")
    if ifc_path.exists() and ifc_path.is_file() and ifc_path.stat().st_size > 0:
        return SelectedInput(
            path=ifc_path,
            source="design_ifc",
            reason="inputs.ifc",
            size_bytes=int(ifc_path.stat().st_size),
        )
    if bool(cfg_get(cfg, "bim.allow_synthetic_ifc_fallback_for_tests", False)):
        return SelectedInput(path=ifc_path, source="synthetic_test_ifc", reason="test_fallback", size_bytes=0)
    raise Stage8InputError(f"IFC file is missing or empty: {ifc_path}")
