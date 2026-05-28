from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import cfg_get, stage45_paths


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CamsGsInputs:
    images_dir: Path
    sparse_model_dir: Path
    colmap_workspace: Path
    image_count: int
    selected_images: list[Path]
    warnings: list[str]


def _image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def select_inputs(cfg: Any) -> CamsGsInputs:
    paths = stage45_paths(cfg)
    max_images = int(cfg_get(cfg, "cams_gs.max_training_images", 300) or 300)

    images = _image_files(paths["source_images_dir"])
    selected = images[:max_images] if max_images > 0 else images

    warnings: list[str] = []
    if not images:
        warnings.append(f"missing_or_empty_images_dir:{paths['source_images_dir'].as_posix()}")
    if not paths["source_sparse_model_dir"].exists():
        warnings.append(f"missing_sparse_model_dir:{paths['source_sparse_model_dir'].as_posix()}")
    if len(images) > len(selected):
        warnings.append(f"image_limit_applied:{len(selected)}/{len(images)}")

    return CamsGsInputs(
        images_dir=paths["source_images_dir"],
        sparse_model_dir=paths["source_sparse_model_dir"],
        colmap_workspace=paths["source_colmap_workspace"],
        image_count=len(images),
        selected_images=selected,
        warnings=warnings,
    )
