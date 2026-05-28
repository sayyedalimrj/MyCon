"""Run COLMAP mapper and promote the best sparse component."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .colmap_cli import ColmapRunner
from .config_access import bool_to_colmap, cfg_bool, cfg_int

SPARSE_BIN_NAMES = ("cameras.bin", "images.bin", "points3D.bin")


def build_mapper_args(
    cfg: Any,
    database_path: Path,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--Mapper.min_num_matches",
        str(cfg_int(cfg, "colmap.mapper_min_num_matches", 15)),
        "--Mapper.multiple_models",
        bool_to_colmap(cfg_bool(cfg, "colmap.mapper_multiple_models", True)),
        "--Mapper.extract_colors",
        bool_to_colmap(cfg_bool(cfg, "colmap.mapper_extract_colors", False)),
    ]


def run_mapper(
    runner: ColmapRunner,
    cfg: Any,
    database_path: Path,
    image_path: Path,
    output_path: Path,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    runner.run(build_mapper_args(cfg, database_path, image_path, output_path), name="mapper")


def has_sparse_model(path: Path) -> bool:
    return all((path / name).exists() and (path / name).stat().st_size > 0 for name in SPARSE_BIN_NAMES)


def _safe_model_score(path: Path) -> tuple[int, int, str]:
    """Rank mapper components without in-process PyCOLMAP.

    Reading a corrupt COLMAP binary model through pycolmap can segfault the
    Python process because the parser lives in C++. For choosing a component,
    file sizes are a safe and deterministic proxy; detailed stats are collected
    later through COLMAP subprocess tools.
    """
    images_size = (path / "images.bin").stat().st_size if (path / "images.bin").exists() else 0
    points_size = (path / "points3D.bin").stat().st_size if (path / "points3D.bin").exists() else 0
    return (images_size, points_size, path.name)


def find_best_sparse_model(sparse_output_dir: Path) -> Path:
    candidates = [p for p in sparse_output_dir.iterdir() if p.is_dir() and has_sparse_model(p)] if sparse_output_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(f"COLMAP mapper did not produce a valid sparse model under: {sparse_output_dir}")
    candidates.sort(key=_safe_model_score, reverse=True)
    return candidates[0]


def promote_sparse_model(best_model_dir: Path, final_sparse_dir: Path, force: bool, logger: logging.Logger | None = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    final_model_dir = final_sparse_dir / "0"
    if final_model_dir.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing sparse model without --force: {final_model_dir}")
        shutil.rmtree(final_model_dir)
    final_sparse_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_model_dir, final_model_dir)
    logger.info("Promoted sparse model %s -> %s", best_model_dir, final_model_dir)
    return final_model_dir
