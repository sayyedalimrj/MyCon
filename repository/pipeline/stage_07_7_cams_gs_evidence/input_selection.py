from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import stage77_paths
from .io_utils import read_json


@dataclass(frozen=True)
class CamsGsEvidenceInputs:
    stage45_manifest_path: Path
    stage45_training_status_path: Path
    dataset_dir: Path
    viewer_export_manifest_path: Path
    stage45_manifest: dict[str, Any]
    training_status: dict[str, Any]
    viewer_export_manifest: dict[str, Any]
    warnings: list[str]


def select_inputs(cfg: Any) -> CamsGsEvidenceInputs:
    paths = stage77_paths(cfg)
    warnings: list[str] = []

    if not paths["stage45_manifest_json"].exists():
        warnings.append(f"missing_stage45_manifest:{paths['stage45_manifest_json'].as_posix()}")
    if not paths["stage45_training_status_json"].exists():
        warnings.append(f"missing_stage45_training_status:{paths['stage45_training_status_json'].as_posix()}")
    if not paths["stage45_dataset_dir"].exists():
        warnings.append(f"missing_stage45_dataset_dir:{paths['stage45_dataset_dir'].as_posix()}")
    if not paths["viewer_export_manifest_json"].exists():
        warnings.append(f"missing_viewer_export_manifest:{paths['viewer_export_manifest_json'].as_posix()}")

    return CamsGsEvidenceInputs(
        stage45_manifest_path=paths["stage45_manifest_json"],
        stage45_training_status_path=paths["stage45_training_status_json"],
        dataset_dir=paths["stage45_dataset_dir"],
        viewer_export_manifest_path=paths["viewer_export_manifest_json"],
        stage45_manifest=read_json(paths["stage45_manifest_json"]),
        training_status=read_json(paths["stage45_training_status_json"]),
        viewer_export_manifest=read_json(paths["viewer_export_manifest_json"]),
        warnings=warnings,
    )
