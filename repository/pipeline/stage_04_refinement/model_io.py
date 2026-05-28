"""COLMAP sparse model file-contract helpers for Stage 4."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

SPARSE_BIN_NAMES = ("cameras.bin", "images.bin", "points3D.bin")


class SparseModelContractError(RuntimeError):
    """Raised when a sparse model does not satisfy the expected file contract."""


def has_sparse_model(path: Path) -> bool:
    return all((path / name).exists() and (path / name).stat().st_size > 0 for name in SPARSE_BIN_NAMES)


def validate_sparse_model(path: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    ok = path.exists() and path.is_dir()
    for name in SPARSE_BIN_NAMES:
        file_path = path / name
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        files[name] = {"exists": exists, "size_bytes": size, "nonempty": size > 0}
        ok = ok and exists and size > 0
    return {"model_dir": str(path), "valid_binary_contract": bool(ok), "files": files}


def _safe_model_score(path: Path) -> tuple[int, int, str]:
    """Score a COLMAP sparse component without parsing binaries in-process.

    points3D.bin size is the safest proxy for useful sparse content here;
    images.bin is used only as a tie breaker.
    """
    points_size = (path / "points3D.bin").stat().st_size if (path / "points3D.bin").exists() else 0
    images_size = (path / "images.bin").stat().st_size if (path / "images.bin").exists() else 0
    return (points_size, images_size, path.name)


def resolve_sparse_component_dir(path: Path) -> Path:
    """Resolve either a component dir or a parent sparse dir to the best component.

    COLMAP may create multiple numbered sparse components. We do not blindly
    prefer component ``0`` because the largest/most useful component can be a
    different child. The choice is based on safe file-size proxies only, avoiding
    in-process binary parsing.
    """
    if has_sparse_model(path):
        return path
    candidates = [child for child in path.iterdir() if child.is_dir() and has_sparse_model(child)] if path.exists() else []
    if not candidates:
        raise SparseModelContractError(
            f"No valid COLMAP sparse component found at {path}. Expected cameras.bin, images.bin, points3D.bin."
        )
    candidates.sort(key=_safe_model_score, reverse=True)
    return candidates[0]


def copy_sparse_model(src: Path, dst: Path, force: bool) -> Path:
    """Copy a sparse model directory safely into the destination component directory."""
    src = resolve_sparse_component_dir(src)
    validation = validate_sparse_model(src)
    if not validation["valid_binary_contract"]:
        raise SparseModelContractError(f"Invalid source sparse model: {src}")
    if dst.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing sparse model without --force: {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    tmp.replace(dst)
    return dst
