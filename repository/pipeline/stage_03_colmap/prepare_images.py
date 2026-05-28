"""Prepare Stage 2 keyframes for COLMAP Stage 3."""
from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import Stage3ConfigError, cfg_get, cfg_int, project_name, resolve_project_path
from .io_utils import clean_dir


@dataclass(frozen=True, slots=True)
class PreparedSparseInputs:
    project_name: str
    source_keyframes_dir: Path
    manifest_csv: Path
    sfm_dir: Path
    stage_images_dir: Path
    active_manifest_csv: Path
    image_list_txt: Path
    rows: list[dict[str, str]]


def _truthy(value: str | bool | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_manifest_image(root: Path, row: dict[str, str]) -> Path:
    raw = row.get("image_path", "")
    if not raw:
        raise Stage3ConfigError("Manifest row is missing image_path")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _stage_image(source: Path, dest_dir: Path) -> Path:
    target = dest_dir / source.name
    shutil.copy2(source, target)
    return target


def prepare_sparse_inputs(cfg: Any, force: bool, logger: logging.Logger | None = None) -> PreparedSparseInputs:
    logger = logger or logging.getLogger(__name__)
    root = resolve_project_path(cfg, "project.root", "/workspace")
    name = project_name(cfg)
    keyframes_dir = resolve_project_path(cfg, "paths.keyframes_dir")
    manifest_csv = resolve_project_path(cfg, "paths.manifest_csv")
    sfm_dir = resolve_project_path(cfg, "paths.sfm_dir", f"data/sfm/{name}")
    stage_images_dir = sfm_dir / "images"
    active_manifest_csv = sfm_dir / "active_manifest.csv"
    image_list_txt = sfm_dir / "image_list.txt"
    min_images = cfg_int(cfg, "colmap.min_input_images", 2)
    stage_mode = str(cfg_get(cfg, "colmap.stage_images_mode", "copy")).lower()
    if stage_mode != "copy":
        # Symlinks across Windows/WSL bind mounts can be fragile for C++ tools.
        # Keep Stage 3 deterministic by downgrading to copy instead of failing.
        logger.warning("Ignoring colmap.stage_images_mode=%s; Stage 3 always copies images for Docker/WSL safety.", stage_mode)
    if not keyframes_dir.exists():
        raise Stage3ConfigError(f"Stage 3 keyframes directory does not exist: {keyframes_dir}")
    if not manifest_csv.exists():
        raise Stage3ConfigError(f"Stage 3 manifest CSV does not exist: {manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise Stage3ConfigError(f"Stage 3 manifest is empty: {manifest_csv}")
    missing_columns = {"image_path", "keep_sparse"} - set(rows[0].keys())
    if missing_columns:
        raise Stage3ConfigError(f"Stage 3 manifest missing required columns: {sorted(missing_columns)}")

    active: list[dict[str, str]] = []
    missing_images: list[str] = []
    for row in rows:
        if not _truthy(row.get("keep_sparse", "true")):
            continue
        source = _resolve_manifest_image(root, row)
        if not source.exists():
            missing_images.append(str(source))
            continue
        active.append(dict(row))
    if missing_images:
        preview = "\n".join(missing_images[:20])
        raise Stage3ConfigError(f"Manifest references missing keyframe images. First missing files:\n{preview}")
    if len(active) < min_images:
        raise Stage3ConfigError(
            f"Stage 3 needs at least {min_images} keep_sparse keyframes, found {len(active)} in {manifest_csv}"
        )

    sfm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir(stage_images_dir, force=force)
    staged_rows: list[dict[str, str]] = []
    image_names: list[str] = []
    for row in active:
        source = _resolve_manifest_image(root, row)
        staged = _stage_image(source, stage_images_dir)
        staged_row = dict(row)
        try:
            staged_row["stage_image_path"] = str(staged.relative_to(root))
        except ValueError:
            staged_row["stage_image_path"] = str(staged)
        staged_rows.append(staged_row)
        image_names.append(staged.name)

    fieldnames = list(staged_rows[0].keys())
    with active_manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(staged_rows)
    image_list_txt.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    logger.info("Prepared %d Stage 3 images at %s", len(staged_rows), stage_images_dir)
    return PreparedSparseInputs(
        project_name=name,
        source_keyframes_dir=keyframes_dir,
        manifest_csv=manifest_csv,
        sfm_dir=sfm_dir,
        stage_images_dir=stage_images_dir,
        active_manifest_csv=active_manifest_csv,
        image_list_txt=image_list_txt,
        rows=staged_rows,
    )
